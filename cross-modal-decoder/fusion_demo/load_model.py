from model_fusion_plip import MultiCropWrapper,fusionblock2,fusionblock_wonum
import torch
from torch import nn
from transformers import  CLIPModel

def model_fusion(depth=2,noise_ratio=0.5, gate=True,num_em=True):
    prototype_all = torch.load("songci_prototype.pt",map_location="cuda") # from patch-level pretraining and have already padding !
    disease_model = CLIPModel.from_pretrained("vinid/plip")
    disease_model.eval()

    if num_em == True:

        model_fusion = fusionblock2(prototype_all=prototype_all, text_model=disease_model, disease_model=disease_model, depth=depth, noise_ratio=noise_ratio, gated=gate)
    else:
        model_fusion = fusionblock_wonum(prototype_all=prototype_all, text_model=disease_model, disease_model=disease_model,
                                    depth=depth, noise_ratio=noise_ratio, gated=gate)

    return model_fusion

# model = model_fusion()

# model.load_state_dict(torch.load("fusion_checkpoint.pth",map_location="cpu"))
# print("finish!!!!")
