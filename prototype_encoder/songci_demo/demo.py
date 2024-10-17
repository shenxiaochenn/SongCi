import vision_former as vits
import torch

model = vits.__dict__['vit_small'](patch_size=16, num_classes=0)
model.load_state_dict(torch.load("./songci.pth"))


for p in model.parameters():

    p.requires_grad = False


model.eval()

aa=torch.randn((10,3,224,224))

print(model(aa).shape)
