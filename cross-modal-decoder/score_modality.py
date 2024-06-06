import argparse
import os
import sys
import json
from pathlib import Path
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch
from torch import nn
from datasets import Mydataset
import vision_transformer as vits
from vision_transformer import DINOHead
from model_fusion_plip import MultiCropWrapper,fusionblock2
import utils
import math
import time
import datetime
from datasets import Mydataset_plip2
from torch.utils import data
import numpy as np
import pandas as pd
import random
import open_clip
from torch.utils.data import DataLoader
from operator import itemgetter
from transformers import CLIPProcessor, CLIPModel

myparse = argparse.ArgumentParser(description="score multi modality")
myparse.add_argument('--checkpoint',default='checkpoint_1018_220.pth', type=str, help='prototype checkpoint path')
myparse.add_argument("--depth", default=2, type=int, help="Number of transformer encoder.")
myparse.add_argument('--gate', default=True, type=utils.bool_flag, help="Whether to use gated transformer encoder layer (Default: True)")
myparse.add_argument('--fusion_checkpoint',default="./log_new/log_1000_depth2_1018_220_plip2/checkpoint.pth", type=str, help='model fusion checkpoint path')
myparse.add_argument('--data_path',default="/home/wangzhenyuan/pathology/multi_modality/xian_all_220_change.csv", type=str, help='data(csv) path')
myparse.add_argument("--threshold", default=0.89, type=float, help="cut off threshold")
myparse.add_argument('--out_name',default="all_model_log/plip2_xian_220.csv", type=str, help='data(csv) name')

args = myparse.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1412)

model = vits.__dict__['vit_small'](patch_size=16, num_classes=0)
state_dict = torch.load(args.checkpoint, map_location="cpu")  ### checkpoint
head = DINOHead(384,
                65536,
                use_bn=True,
                norm_last_layer=False,
                predictor=True, )
student = MultiCropWrapper(model, head)
state_dict_new = state_dict["student"]
state_dict_new = {k.replace("module.", ""): v for k, v in state_dict_new.items()}
student.load_state_dict(state_dict_new, strict=True)
student.eval()
with torch.no_grad():
    w = student.prototypes.weight.data.clone()
    w = nn.functional.normalize(w, dim=1, p=2)
    student.prototypes.weight.copy_(w)
    prototypes = student.prototypes.weight
    p = torch.zeros(prototypes.shape[1])
    p = p.unsqueeze(0)
    prototype_all = torch.cat((prototypes, p), 0).cuda()

# model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
#     'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', )
# text_model = model.text
# text_model.eval()

disease_model = CLIPModel.from_pretrained("vinid/plip")
disease_model.eval()

model_fusion = fusionblock2(prototype_all=prototype_all.cuda(), text_model=disease_model, disease_model=disease_model, depth=args.depth, noise_ratio=0, gated=args.gate)

model_fusion.load_state_dict(torch.load(args.fusion_checkpoint)["model_fusion"])

for name, param in model_fusion.named_parameters():
    param.requires_grad = False

# for name, param in text_model.named_parameters():
#     param.requires_grad = False

for name, param in disease_model.named_parameters():
    param.requires_grad = False


model_fusion.eval()
model_fusion.cuda()


dd = Mydataset_plip2(args.data_path, randomm=False)
trainloader = DataLoader(dd, batch_size=1, shuffle=False,drop_last=False)
data_xian = pd.read_csv(args.data_path)
data_pd = pd.read_csv("/home/wangzhenyuan/pathology/multi_modality/all_data.csv") ## xian and institute and shanghai
data_xian["predict"] = ""
data_xian["case"] = ""
d_disease = []
for i in data_pd.loc[:,"disease"]:
    for j in i.split("/"):
        d_disease.append(j.strip().capitalize())
d_disease = list(set(d_disease))
print("all disease proposal:",len(d_disease))

tokenizer = CLIPProcessor.from_pretrained("vinid/plip").tokenizer
token_disease = tokenizer(d_disease, padding="max_length", max_length=77, return_tensors="pt")
token_disease["input_ids"] = token_disease["input_ids"].cuda(non_blocking=True)
token_disease["attention_mask"] = token_disease["attention_mask"].cuda(non_blocking=True)

disease_rep = disease_model.get_text_features(**token_disease)
disease_rep = F.normalize(disease_rep, dim=-1).cpu()

with open("disease.json","r", encoding='utf-8') as f:
    di_dic = json.load(f)
print("all disease proposal:",len(di_dic))
num_wrong_1 = 0
num_wrong_2 = 0
num_wrong_3 = 0
ll_iou = []
ll_recall = []
ll_precision = []
for it, (tokens_disease,tokens_description,img,img_num) in enumerate(trainloader):
    # print(img_num)
    score_list=[]
    model_fusion.eval()
    with torch.no_grad():
        tokens_description["input_ids"] = tokens_description["input_ids"].squeeze(1).cuda(non_blocking=True)
        tokens_description["attention_mask"] = tokens_description["attention_mask"].squeeze(1).cuda(non_blocking=True)
        for ind, (t_disease,t_attention) in enumerate(zip(token_disease["input_ids"],token_disease["attention_mask"])):
            _, _, _, img_text_out, disease, attention_score = model_fusion(img_p=img.cuda(non_blocking=True),
                                                                           text=tokens_description,
                                                                           img_num=img_num.cuda(non_blocking=True),
                                                                           disease={"input_ids":t_disease.unsqueeze(0),"attention_mask":t_attention.unsqueeze(0)})
            score = img_text_out.cpu().squeeze() @ disease_rep[ind].t()
            score_list.append(score)

        score_array = np.array(score_list)


        disease_name_max = [d_disease[i] for i in score_array.argsort()[-1:][::-1].tolist()]  # 预测的唯一的解
        disease_name_top2 = [d_disease[i] for i in score_array.argsort()[-2:][::-1].tolist()] # top2 的解
        disease_name_top3 = [d_disease[i] for i in score_array.argsort()[-3:][::-1].tolist()]  # top3 的解


        disease_name = [d_disease[i] for i, ss in enumerate(score_list) if ss >args.threshold] # 预测的最大的几个解

        diagnois = [i.strip().capitalize() for i in data_xian.loc[it, "disease"].split("/")]  ## 诊断的label..


        if not disease_name:
            disease_name = disease_name_max



        #print("disease_name:", disease_name)
        #print("diagnois:", diagnois)
        #print("top2:",disease_name_top2)
        if len(disease_name) > 1:
            disease_name_num = list(itemgetter(*disease_name)(di_dic))
        else:
            disease_name_num = [di_dic[disease_name[0]]]

        if len(diagnois) > 1:
            diagnois_num = list(itemgetter(*diagnois)(di_dic))
        else:
            diagnois_num = [di_dic[diagnois[0]]]



        disease_name_max_num = [di_dic[disease_name_max[0]]]
        disease_name_top2_num = list(itemgetter(*disease_name_top2)(di_dic))
        disease_name_top3_num = list(itemgetter(*disease_name_top3)(di_dic))


        iou = len(list(set(disease_name_num) & set(diagnois_num))) / len(list(set(disease_name_num) | set(diagnois_num)))

        data_xian.loc[it,"predict"] = "/".join(disease_name)

        recall = len(list(set(disease_name_num) & set(diagnois_num))) / len(set(diagnois_num))
        precision = len(list(set(disease_name_num) & set(diagnois_num))) / len(set(disease_name_num))
        ll_iou.append(iou)
        ll_precision.append(precision)
        ll_recall.append(recall)

        if not any(_ in diagnois_num for _ in disease_name_top2_num):
            num_wrong_2 += 1
        if not any(_ in diagnois_num for _ in disease_name_max_num):
            num_wrong_1 += 1
        if not any(_ in diagnois_num for _ in disease_name_top3_num):
            num_wrong_3 += 1





mean_iou = sum(ll_iou)/len(ll_iou)
mean_precision = sum(ll_precision)/len(ll_precision)
mean_recall = sum(ll_recall)/len(ll_recall)


print("top1 Pointing Game：",num_wrong_1)
print("top2 Pointing Game：",num_wrong_2)
print("top3 Pointing Game：",num_wrong_3)
print("miou:",mean_iou)
print("mprecision:",mean_precision)
print("mrecall:",mean_recall)


data_xian.to_csv(f"./{args.out_name}",index=0)


