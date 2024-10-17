import torch
import math
from torch import nn, einsum
import torch.nn.functional as F
import vision_transformer as vits
from einops import rearrange, repeat
from vision_transformer import DINOHead
from einops_exts import rearrange_many, repeat_many
from torch.utils import data
import numpy as np
import pandas as pd
import open_clip

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        media,
    ):
        "media (b,n,d)"
        h = self.heads
        x = self.norm(x)
        q = self.to_q(x)

        k, v = self.to_kv(media).chunk(2, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        gated = True,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
    ):
        super().__init__()
        self.gated =gated
        self.attn = CrossAttention(dim = dim, dim_head = dim_head, heads = heads)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult = ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(
        self,
        x,
        media,                  # media tensor, encoded by perceiver resample - (batch,latents, dim)
    ):
        if self.gated:
            x = self.attn(x, media) * self.attn_gate.tanh() + x
            x = self.ff(x) * self.ff_gate.tanh() + x
        else:
            x = self.attn(x, media) + x
            x = self.ff(x) + x
        return x


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head, in_dim=384, hidden_dim=2048, nlayers=2, output_dim=256, nmb_prototypes=1024, teacher=False):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.teacher = teacher
        self.backbone = backbone
        self.head = head
        self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)  # change
        layers = [nn.Linear(in_dim, hidden_dim)]
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.projection = nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        x = self.prototypes(x)
        return x


def numstep_embedding(timesteps, dim, max_period=10000):

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding



def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


class fusionblock(nn.Module):
    def __init__(self, prototype_all, text_model, disease_model, depth=1, noise_ratio=0.2, gated=True, dim_in=256, num_embed_dim=256, text_dim=768, dim_out=512, logit_scale_init_value=0.07):
        super().__init__()
        self.prototype = prototype_all
        self.text_model = text_model
        self.disease_model = disease_model
        self.ratio = noise_ratio

        self.dim_in = dim_in
        self.dim_media = dim_in * 4
        self.num_embed = nn.Sequential(
            linear(dim_in, num_embed_dim),
            nn.SiLU(),
            linear(num_embed_dim, num_embed_dim),
        )
        self.project_text_to_img = nn.Sequential(
            linear(text_dim, dim_out, bias=False),
            nn.GELU(),
            linear(dim_out, dim_in, bias=False),
        )
        self.cross_img_text = GatedCrossAttentionBlock(dim=dim_in, gated=gated)
        self.cross_text_img = GatedCrossAttentionBlock(dim=dim_in, gated=gated)
        # self.transformer = GatedCrossAttentionBlock(dim=dim_in)
        self.transformer = nn.ModuleList([
            GatedCrossAttentionBlock(dim=dim_in, gated=gated)
            for i in range(depth)])
        self.linear = linear(dim_in, dim_out, bias=False)
        self.emmeder = nn.Sequential(
            linear(dim_in, self.dim_media, bias=False),
            nn.GELU(),
            linear(self.dim_media, dim_out, bias=False),
        )
        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / logit_scale_init_value)))

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def encode_image(self,img_p):
        with torch.no_grad():
            img_p = self.prototype[img_p]  # (b,n,d)
            if self.training:
                #print('image training')
                epsilon_p = torch.randn_like(img_p) * self.ratio
                img_p = img_p + epsilon_p.to(device=img_p.device)
            else:
                img_p = img_p
            img_p = F.normalize(img_p, dim=-1)

        return img_p
    def encode_text(self,text,disease):
        with torch.no_grad():
            text = self.text_model.transformer(text)["last_hidden_state"]
            disease = self.disease_model.get_text_features(**disease)
            if self.training:
                #print('text training')
                epsilon_t = torch.randn_like(text) * self.ratio
                text = text + epsilon_t.to(device=text.device)
            else:
                text =text
            text = F.normalize(text, dim=-1)
            disease = F.normalize(disease, dim=-1)
        return text, disease

    def forward(self, img_p, text, img_num, disease):
        text,disease = self.encode_text(text,disease)
        img_p = self.encode_image(img_p)
        disease = disease.detach()
        img_p = img_p.detach()
        text = text.detach()
        num_embedding = torch.stack([numstep_embedding(img_num[i], dim=self.dim_in) for i in range(img_num.shape[0])])
        num_embedding = self.num_embed(num_embedding)
        img_p_ori = img_p + num_embedding
        text_ori = self.project_text_to_img(text)
        img_p = self.cross_img_text(img_p_ori, text_ori)
        text = self.cross_text_img(text_ori, img_p_ori)
        img_text = torch.cat((img_p, text), dim=1)
        for blk in self.transformer:
            img_text = blk(img_text, img_text)
        img_text_one = self.emmeder(img_text)
        img_text_two = self.linear(img_text)
        img_text = img_text_one + img_text_two  # (b,n,d)
        disease = disease.unsqueeze(dim=1)
        disease = F.normalize(disease, dim=-1)
        img_text = F.normalize(img_text, dim=-1)
        sim = einsum('... i d, ... j d -> ... i j', disease, img_text)
        attention_score = F.softmax(sim, dim=2)  # (b,1,n)
        img_text_out = einsum('... i n, ... n d -> ... i d', attention_score, img_text)
        img_text_out = img_text_out.squeeze(1)
        disease = disease.squeeze(dim=1)
        img_text_out = F.normalize(img_text_out, dim=-1)
        disease = F.normalize(disease, dim=-1)
        logits_per_image = self.compute_logits(img_text_out, disease)
        logits_per_text = logits_per_image.t()
        loss_cosine = 2 - 2 * (img_text_out * disease).sum(dim=-1)
        loss_clip = self.clip_loss(logits_per_text)
        loss = 0.5*loss_clip + loss_cosine.mean()

        return loss, loss_cosine.mean(), loss_clip*0.5, img_text_out, disease, attention_score


class fusionblock2(nn.Module):
    def __init__(self, prototype_all, text_model, disease_model, depth=1, noise_ratio=0.2, gated=True, dim_in=256, num_embed_dim=256, text_dim=512, dim_out=512, logit_scale_init_value=0.07):
        super().__init__()
        self.prototype = prototype_all
        self.text_model = text_model
        self.disease_model = disease_model
        self.ratio = noise_ratio

        self.dim_in = dim_in
        self.dim_media = dim_in * 4
        self.num_embed = nn.Sequential(
            linear(dim_in, num_embed_dim),
            nn.SiLU(),
            linear(num_embed_dim, num_embed_dim),
        )
        self.project_text_to_img = nn.Sequential(
            linear(text_dim, dim_out, bias=False),
            nn.GELU(),
            linear(dim_out, dim_in, bias=False),
        )
        self.cross_img_text = GatedCrossAttentionBlock(dim=dim_in, gated=gated)
        self.cross_text_img = GatedCrossAttentionBlock(dim=dim_in, gated=gated)
        # self.transformer = GatedCrossAttentionBlock(dim=dim_in)
        self.transformer = nn.ModuleList([
            GatedCrossAttentionBlock(dim=dim_in, gated=gated)
            for i in range(depth)])
        self.linear = linear(dim_in, dim_out, bias=False)
        self.emmeder = nn.Sequential(
            linear(dim_in, self.dim_media, bias=False),
            nn.GELU(),
            linear(self.dim_media, dim_out, bias=False),
        )
        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / logit_scale_init_value)))

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def encode_image(self,img_p):
        with torch.no_grad():
            img_p = self.prototype[img_p]  # (b,n,d)
            if self.training:
                #print('image training')
                epsilon_p = torch.randn_like(img_p) * self.ratio
                img_p = img_p + epsilon_p.to(device=img_p.device)
            else:
                img_p = img_p
            img_p = F.normalize(img_p, dim=-1)

        return img_p
    def encode_text(self,text,disease):
        with torch.no_grad():
            text = self.text_model.text_model(**text)["last_hidden_state"]

            disease = self.disease_model.get_text_features(**disease)
            if self.training:
                #print('text training')
                epsilon_t = torch.randn_like(text) * self.ratio
                text = text + epsilon_t.to(device=text.device)
            else:
                text =text
            text = F.normalize(text, dim=-1)
            disease = F.normalize(disease, dim=-1)
        return text, disease

    def forward(self, img_p, text, img_num, disease):
        text,disease = self.encode_text(text,disease)
        img_p = self.encode_image(img_p)
        disease = disease.detach()
        img_p = img_p.detach()
        text = text.detach()
        num_embedding = torch.stack([numstep_embedding(img_num[i], dim=self.dim_in) for i in range(img_num.shape[0])])
        num_embedding = self.num_embed(num_embedding)
        img_p_ori = img_p + num_embedding
        text_ori = self.project_text_to_img(text)
        img_p = self.cross_img_text(img_p_ori, text_ori)
        text = self.cross_text_img(text_ori, img_p_ori)
        img_text = torch.cat((img_p, text), dim=1)
        for blk in self.transformer:
            img_text = blk(img_text, img_text)
        img_text_one = self.emmeder(img_text)
        img_text_two = self.linear(img_text)
        img_text = img_text_one + img_text_two  # (b,n,d)
        disease = disease.unsqueeze(dim=1)
        disease = F.normalize(disease, dim=-1)
        img_text = F.normalize(img_text, dim=-1)
        sim = einsum('... i d, ... j d -> ... i j', disease, img_text)
        attention_score = F.softmax(sim, dim=2)  # (b,1,n)
        img_text_out = einsum('... i n, ... n d -> ... i d', attention_score, img_text)
        img_text_out = img_text_out.squeeze(1)
        disease = disease.squeeze(dim=1)
        img_text_out = F.normalize(img_text_out, dim=-1)
        disease = F.normalize(disease, dim=-1)
        logits_per_image = self.compute_logits(img_text_out, disease)
        logits_per_text = logits_per_image.t()
        loss_cosine = 2 - 2 * (img_text_out * disease).sum(dim=-1)
        loss_clip = self.clip_loss(logits_per_text)
        loss = 0.5*loss_clip + loss_cosine.mean()

        return loss, loss_cosine.mean(), loss_clip*0.5, img_text_out, disease, attention_score


class fusionblock_wonum(nn.Module):
    def __init__(self, prototype_all, text_model, disease_model, depth=1, noise_ratio=0.2, gated=True, dim_in=256, num_embed_dim=256, text_dim=512, dim_out=512, logit_scale_init_value=0.07):
        super().__init__()
        self.prototype = prototype_all
        self.text_model = text_model
        self.disease_model = disease_model
        self.ratio = noise_ratio

        self.dim_in = dim_in
        self.dim_media = dim_in * 4
        # self.num_embed = nn.Sequential(
        #     linear(dim_in, num_embed_dim),
        #     nn.SiLU(),
        #     linear(num_embed_dim, num_embed_dim),
        # )
        self.project_text_to_img = nn.Sequential(
            linear(text_dim, dim_out, bias=False),
            nn.GELU(),
            linear(dim_out, dim_in, bias=False),
        )
        self.cross_img_text = GatedCrossAttentionBlock(dim=dim_in, gated=gated)
        self.cross_text_img = GatedCrossAttentionBlock(dim=dim_in, gated=gated)
        # self.transformer = GatedCrossAttentionBlock(dim=dim_in)
        self.transformer = nn.ModuleList([
            GatedCrossAttentionBlock(dim=dim_in, gated=gated)
            for i in range(depth)])
        self.linear = linear(dim_in, dim_out, bias=False)
        self.emmeder = nn.Sequential(
            linear(dim_in, self.dim_media, bias=False),
            nn.GELU(),
            linear(self.dim_media, dim_out, bias=False),
        )
        # learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / logit_scale_init_value)))

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text.t()

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.T)
        return (caption_loss + image_loss) / 2.0

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def encode_image(self,img_p):
        with torch.no_grad():
            img_p = self.prototype[img_p]  # (b,n,d)
            if self.training:
                #print('image training')
                epsilon_p = torch.randn_like(img_p) * self.ratio
                img_p = img_p + epsilon_p.to(device=img_p.device)
            else:
                img_p = img_p
            img_p = F.normalize(img_p, dim=-1)

        return img_p
    def encode_text(self,text,disease):
        with torch.no_grad():
            text = self.text_model.text_model(**text)["last_hidden_state"]

            disease = self.disease_model.get_text_features(**disease)
            if self.training:
                #print('text training')
                epsilon_t = torch.randn_like(text) * self.ratio
                text = text + epsilon_t.to(device=text.device)
            else:
                text =text
            text = F.normalize(text, dim=-1)
            disease = F.normalize(disease, dim=-1)
        return text, disease

    def forward(self, img_p, text, img_num, disease):
        text,disease = self.encode_text(text,disease)
        img_p = self.encode_image(img_p)
        disease = disease.detach()
        img_p_ori = img_p.detach()
        text = text.detach()
        #num_embedding = torch.stack([numstep_embedding(img_num[i], dim=self.dim_in) for i in range(img_num.shape[0])])
        #num_embedding = self.num_embed(num_embedding)
        #img_p_ori = img_p + num_embedding
        text_ori = self.project_text_to_img(text)
        img_p = self.cross_img_text(img_p_ori, text_ori)
        text = self.cross_text_img(text_ori, img_p_ori)
        img_text = torch.cat((img_p, text), dim=1)
        for blk in self.transformer:
            img_text = blk(img_text, img_text)
        img_text_one = self.emmeder(img_text)
        img_text_two = self.linear(img_text)
        img_text = img_text_one + img_text_two  # (b,n,d)
        disease = disease.unsqueeze(dim=1)
        disease = F.normalize(disease, dim=-1)
        img_text = F.normalize(img_text, dim=-1)
        sim = einsum('... i d, ... j d -> ... i j', disease, img_text)
        attention_score = F.softmax(sim, dim=2)  # (b,1,n)
        img_text_out = einsum('... i n, ... n d -> ... i d', attention_score, img_text)
        img_text_out = img_text_out.squeeze(1)
        disease = disease.squeeze(dim=1)
        img_text_out = F.normalize(img_text_out, dim=-1)
        disease = F.normalize(disease, dim=-1)
        logits_per_image = self.compute_logits(img_text_out, disease)
        logits_per_text = logits_per_image.t()
        loss_cosine = 2 - 2 * (img_text_out * disease).sum(dim=-1)
        loss_clip = self.clip_loss(logits_per_text)
        loss = 0.5*loss_clip + loss_cosine.mean()

        return loss, loss_cosine.mean(), loss_clip*0.5, img_text_out, disease, attention_score
