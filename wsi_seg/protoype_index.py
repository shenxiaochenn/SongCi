import os
from PIL import Image
import re
import torch
import torch.nn as nn
import vision_transformer as vits
from vision_transformer import DINOHead
from torchvision import  transforms
import pandas as pd
from tqdm import tqdm
# build model
device = torch.device("cuda")
model = vits.__dict__['vit_small'](patch_size=16, num_classes=0)
state_dict = torch.load("./checkpoint_1018_220.pth", map_location="cpu")
head = DINOHead(384,
        65536,
        use_bn=True,
        norm_last_layer=False,
        predictor=True,)

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

student = MultiCropWrapper(model,head)
state_dict_new = state_dict["student"]
state_dict_new = {k.replace("module.", ""): v for k, v in state_dict_new.items()}
student.load_state_dict(state_dict_new,strict=True)

for p in student.parameters():
    p.requires_grad = False


normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
trans = transforms.Compose([transforms.Resize((224,224)),normalize])

data_dir ="/home/hpcstack/data/human_patch_10x/"
student = student.to(device)
student.eval()
with torch.no_grad():
    w = student.prototypes.weight.data.clone()
    w = nn.functional.normalize(w, dim=1, p=2)
    student.prototypes.weight.copy_(w)

for i in os.listdir(data_dir):
    if os.path.exists(f"/home/wangzhenyuan/pathology/index/prototype_xian1_1018_220/{i}.csv"):
        continue
    else:
        print(i)
        data = pd.DataFrame({"name": [], "x_axis": [], "y_axis": [], "pro_index": [], "sim_value": []})
        for j in tqdm(os.listdir(os.path.join(data_dir, i))):
            ll = re.split("-|_", j)
            dirr = os.path.join(data_dir, i, j)
            pil_image = Image.open(dirr)
            pil_image = trans(pil_image)
            img = torch.unsqueeze(pil_image, 0)
            #print(img.shape)
            with torch.no_grad():
                ind = student(img.to(device))
                v, pro_ind = ind.max(1)
            #print(v)
            #print(pro_ind)
            data.loc[len(data.index)] = [i, int(ll[-3]), int(ll[-2]), pro_ind.item(), v.item()]
        print(data["pro_index"].value_counts())
        data.to_csv(f"/home/wangzhenyuan/pathology/index/prototype_xian1_1018_220/{i}.csv")

print("finish!")

