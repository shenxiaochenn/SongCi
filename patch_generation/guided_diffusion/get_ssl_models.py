# generation based on prototype
import torch
import torch.nn as nn
from guided_diffusion import vision_transformer as vits
from guided_diffusion.vision_transformer import DINOHead


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
        _, pro_ind = x.max(1)
        x =self.prototypes.weight[pro_ind]
        return x



def get_model(model="my_model", use_head=False, pretrained_weights=None):
    if model == "my_model":
        model = vits.__dict__['vit_small'](patch_size=16, num_classes=0)
        state_dict = torch.load("/home/wangzhenyuan/pathology/multi_modality/checkpoint_1018_220.pth", map_location="cpu")
        head = DINOHead(384,
                        65536,
                        use_bn=True,
                        norm_last_layer=False,
                        predictor=True, )

        student = MultiCropWrapper(model, head)
        state_dict_new = state_dict["student"]
        state_dict_new = {k.replace("module.", ""): v for k, v in state_dict_new.items()}
        student.load_state_dict(state_dict_new, strict=True)

        for p in student.parameters():
            p.requires_grad = False

        student.eval()
        with torch.no_grad():
            w = student.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            student.prototypes.weight.copy_(w)

        return student
