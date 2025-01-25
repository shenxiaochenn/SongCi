import sys
#sys.path.append('/home/wangzhenyuan/CONCH/CONCH/conch/')
import torch
from torch.utils import data
import numpy as np
import pandas as pd
import random
import open_clip
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer
#from open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize  #### FOR CONCH
class Mydataset(data.Dataset):
    def __init__(self, root="./tidy_data.csv",randomm=True):
        self.data = pd.read_csv(root)
        self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.img_length = 128
        self.randomm = randomm
    def __getitem__(self, index):
        if self.randomm:
            disease = random.choice(self.data.iloc[index,]["disease"].split("/"))
        else:
            disease = self.data.iloc[index,]["disease"].split("/")[0]
        description = self.data.iloc[index,]["description"]
        img = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).keys()))
        img_num = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).values()))
        img_num = torch.where(img_num > 10000, 10000, img_num)
        tokens_disease = self.tokenizer(
            disease,
            context_length=77,
        )
        tokens_description = self.tokenizer(
            description,
            context_length=77,
        )
        if len(img)<self.img_length:
            pad = self.img_length-len(img)
            padding = torch.tensor(1024).repeat(pad)
            padding_num = torch.tensor(0).repeat(pad)
            img = torch.cat((img,padding))
            img_num = torch.cat((img_num,padding_num))
        if len(img)>self.img_length:
            img = img[:self.img_length]
            img_num = img_num[:self.img_length]

        return tokens_disease,tokens_description,img,img_num

    def __len__(self):
        return len(self.data)


class Mydataset_plip(data.Dataset):
    def __init__(self, root="./tidy_data.csv",randomm=True):
        self.data = pd.read_csv(root)
        self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer_2 = CLIPProcessor.from_pretrained("vinid/plip").tokenizer
        self.img_length = 128
        self.randomm = randomm
    def __getitem__(self, index):
        if self.randomm:
            disease = random.choice(self.data.iloc[index,]["disease"].split("/"))
        else:
            disease = self.data.iloc[index,]["disease"].split("/")[0]
        description = self.data.iloc[index,]["description"]
        img = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).keys()))
        img_num = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).values()))
        img_num = torch.where(img_num > 10000, 10000, img_num)
        tokens_disease = self.tokenizer_2(disease, padding="max_length", max_length=77, return_tensors="pt")
        tokens_description = self.tokenizer(
            description,
            context_length=77,
        )
        if len(img)<self.img_length:
            pad = self.img_length-len(img)
            padding = torch.tensor(1024).repeat(pad)
            padding_num = torch.tensor(0).repeat(pad)
            img = torch.cat((img,padding))
            img_num = torch.cat((img_num,padding_num))
        if len(img)>self.img_length:
            img = img[:self.img_length]
            img_num = img_num[:self.img_length]

        return tokens_disease,tokens_description,img,img_num

    def __len__(self):
        return len(self.data)


class Mydataset_plip2(data.Dataset):
    def __init__(self, root="./tidy_data.csv",randomm=True):
        self.data = pd.read_csv(root)
        #self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer_2 = CLIPProcessor.from_pretrained("vinid/plip").tokenizer
        self.img_length = 128
        self.randomm = randomm
    def __getitem__(self, index):
        if self.randomm:
            disease = random.choice(self.data.iloc[index,]["disease"].split("/"))
        else:
            disease = self.data.iloc[index,]["disease"].split("/")[0]
        description = self.data.iloc[index,]["description"]
        img = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).keys()))
        img_num = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).values()))
        img_num = torch.where(img_num > 10000, 10000, img_num)
        tokens_disease = self.tokenizer_2(disease, padding="max_length", max_length=77, return_tensors="pt")
        tokens_description = self.tokenizer_2(description, padding="max_length", max_length=77, return_tensors="pt", truncation=True)
        if len(img)<self.img_length:
            pad = self.img_length-len(img)
            padding = torch.tensor(1024).repeat(pad)
            padding_num = torch.tensor(0).repeat(pad)
            img = torch.cat((img,padding))
            img_num = torch.cat((img_num,padding_num))
        if len(img)>self.img_length:
            img = img[:self.img_length]
            img_num = img_num[:self.img_length]

        return tokens_disease,tokens_description,img,img_num

    def __len__(self):
        return len(self.data)


class Mydataset_bert(data.Dataset):
    def __init__(self, root="./tidy_data.csv",randomm=True):
        self.data = pd.read_csv(root)
        self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer_2 = AutoTokenizer.from_pretrained("/home/wangzhenyuan/pathology/multi_modality/pubmedbert/")
        self.img_length = 128
        self.randomm = randomm
    def __getitem__(self, index):
        if self.randomm:
            disease = random.choice(self.data.iloc[index,]["disease"].split("/"))
        else:
            disease = self.data.iloc[index,]["disease"].split("/")[0]
        description = self.data.iloc[index,]["description"]
        img = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).keys()))
        img_num = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).values()))
        img_num = torch.where(img_num > 10000, 10000, img_num)
        tokens_disease = self.tokenizer(disease, context_length=77)
        tokens_description = self.tokenizer_2(description, padding="max_length", max_length=128, return_tensors="pt", truncation=True)
        if len(img)<self.img_length:
            pad = self.img_length-len(img)
            padding = torch.tensor(1024).repeat(pad)
            padding_num = torch.tensor(0).repeat(pad)
            img = torch.cat((img,padding))
            img_num = torch.cat((img_num,padding_num))
        if len(img)>self.img_length:
            img = img[:self.img_length]
            img_num = img_num[:self.img_length]

        return tokens_disease,tokens_description,img,img_num

    def __len__(self):
        return len(self.data)

class Mydataset_bert2(data.Dataset):
    def __init__(self, root="./tidy_data.csv",randomm=True):
        self.data = pd.read_csv(root)
        #self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer_2 = AutoTokenizer.from_pretrained("/home/wangzhenyuan/pathology/multi_modality/pubmedbert/")
        self.img_length = 128
        self.randomm = randomm
    def __getitem__(self, index):
        if self.randomm:
            disease = random.choice(self.data.iloc[index,]["disease"].split("/"))
        else:
            disease = self.data.iloc[index,]["disease"].split("/")[0]
        description = self.data.iloc[index,]["description"]
        img = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).keys()))
        img_num = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).values()))
        img_num = torch.where(img_num > 10000, 10000, img_num)
        tokens_disease = self.tokenizer_2(disease, padding="max_length", max_length=77, return_tensors="pt")
        tokens_description = self.tokenizer_2(description, padding="max_length", max_length=128, return_tensors="pt", truncation=True)
        if len(img)<self.img_length:
            pad = self.img_length-len(img)
            padding = torch.tensor(1024).repeat(pad)
            padding_num = torch.tensor(0).repeat(pad)
            img = torch.cat((img,padding))
            img_num = torch.cat((img_num,padding_num))
        if len(img)>self.img_length:
            img = img[:self.img_length]
            img_num = img_num[:self.img_length]

        return tokens_disease,tokens_description,img,img_num

    def __len__(self):
        return len(self.data)


class Mydataset_quilt2(data.Dataset):
    def __init__(self, root="./tidy_data.csv",randomm=True):
        self.data = pd.read_csv(root)
        #self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer_2 = CLIPProcessor.from_pretrained("/home/wangzhenyuan/pathology/multi_modality/quilt").tokenizer
        self.img_length = 128
        self.randomm = randomm
    def __getitem__(self, index):
        if self.randomm:
            disease = random.choice(self.data.iloc[index,]["disease"].split("/"))
        else:
            disease = self.data.iloc[index,]["disease"].split("/")[0]
        description = self.data.iloc[index,]["description"]
        img = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).keys()))
        img_num = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).values()))
        img_num = torch.where(img_num > 10000, 10000, img_num)
        tokens_disease = self.tokenizer_2(disease, padding="max_length", max_length=77, return_tensors="pt")
        tokens_description = self.tokenizer_2(description, padding="max_length", max_length=77, return_tensors="pt", truncation=True)
        if len(img)<self.img_length:
            pad = self.img_length-len(img)
            padding = torch.tensor(1024).repeat(pad)
            padding_num = torch.tensor(0).repeat(pad)
            img = torch.cat((img,padding))
            img_num = torch.cat((img_num,padding_num))
        if len(img)>self.img_length:
            img = img[:self.img_length]
            img_num = img_num[:self.img_length]

        return tokens_disease,tokens_description,img,img_num

    def __len__(self):
        return len(self.data)



class Mydataset_clipp(data.Dataset):
    def __init__(self, root="./tidy_data.csv",randomm=True):
        self.data = pd.read_csv(root)
        #self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer_2 = CLIPProcessor.from_pretrained("/home/wangzhenyuan/pathology/multi_modality/clipp").tokenizer
        self.img_length = 128
        self.randomm = randomm
    def __getitem__(self, index):
        if self.randomm:
            disease = random.choice(self.data.iloc[index,]["disease"].split("/"))
        else:
            disease = self.data.iloc[index,]["disease"].split("/")[0]
        description = self.data.iloc[index,]["description"]
        img = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).keys()))
        img_num = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).values()))
        img_num = torch.where(img_num > 10000, 10000, img_num)
        tokens_disease = self.tokenizer_2(disease, padding="max_length", max_length=77, return_tensors="pt")
        tokens_description = self.tokenizer_2(description, padding="max_length", max_length=77, return_tensors="pt", truncation=True)
        if len(img)<self.img_length:
            pad = self.img_length-len(img)
            padding = torch.tensor(1024).repeat(pad)
            padding_num = torch.tensor(0).repeat(pad)
            img = torch.cat((img,padding))
            img_num = torch.cat((img_num,padding_num))
        if len(img)>self.img_length:
            img = img[:self.img_length]
            img_num = img_num[:self.img_length]

        return tokens_disease,tokens_description,img,img_num

    def __len__(self):
        return len(self.data)


# class Mydataset_conch(data.Dataset):
#     def __init__(self, root="./tidy_data.csv",randomm=True):
#         self.data = pd.read_csv(root)
#         #self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
#         self.tokenizer_2 = get_tokenizer()
#         self.img_length = 128
#         self.randomm = randomm
#     def __getitem__(self, index):
#         if self.randomm:
#             disease = random.choice(self.data.iloc[index,]["disease"].split("/"))
#         else:
#             disease = self.data.iloc[index,]["disease"].split("/")[0]
#         description = self.data.iloc[index,]["description"]
#         img = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).keys()))
#         img_num = torch.tensor(list(eval(self.data.iloc[index,]["prototype"]).values()))
#         img_num = torch.where(img_num > 10000, 10000, img_num)
#         tokens_disease = tokenize(texts=[disease], tokenizer=self.tokenizer_2)
#         tokens_description = tokenize(texts=[description], tokenizer=self.tokenizer_2)
#         if len(img)<self.img_length:
#             pad = self.img_length-len(img)
#             padding = torch.tensor(1024).repeat(pad)
#             padding_num = torch.tensor(0).repeat(pad)
#             img = torch.cat((img,padding))
#             img_num = torch.cat((img_num,padding_num))
#         if len(img)>self.img_length:
#             img = img[:self.img_length]
#             img_num = img_num[:self.img_length]

        return tokens_disease,tokens_description,img,img_num

    def __len__(self):
        return len(self.data)
