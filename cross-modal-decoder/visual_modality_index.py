import os

import numpy as np

from PIL import Image

import pandas as pd
import heapq
from tqdm import tqdm


colors =[[0,0,128],[244,111,68],[255,223,146],[255,182,193],[144,190,224],[230,241,243]] # 橙红 淡黄,粉色，淡蓝，蛋白

def label2color(img,colors):
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(colors[label])
    return img_color

data = pd.read_csv("/home/wangzhenyuan/pathology/multi_modality/visual_220_plip2_all.csv",index_col=0)

level_3090 = pd.read_csv("/home/wangzhenyuan/pathology/path_wsi/wsi_level_3090.csv",index_col=0)
level_4090 = pd.read_csv("/home/wangzhenyuan/pathology/path_wsi/wsi_level_4090.csv",index_col=0)
level_ins = pd.read_csv("/home/wangzhenyuan/pathology/path_wsi/wsi_level_ins.csv",index_col=0)

def topn_dict(d, n):
    return heapq.nlargest(n, d, key=lambda k: d[k])

#[2637,2638,2639,2640,1143,1144,1145,1665,1666,1667,1275,1276,1210,1211,1212]
for i in tqdm(range(len(data))):
    name = data.loc[i, "name"]
    name_ = name[:-4]
    img_name = name_ + "_ori.png"
    csv = pd.read_csv(f"/home/wangzhenyuan/pathology/index/check_220/prototype_all/{name}", index_col=0)

    if os.path.exists(f"/home/wangzhenyuan/pathology/path_wsi/ori_image_3090/{img_name}"):
        name_svs = name[:-4] + ".svs"
        if name_svs in list(level_3090["name"]):
            level = level_3090.loc[level_3090["name"] == name_svs, "level"].iloc[0]
        else:
            name_svs = name[:-5] + ".svs"
            level = level_3090.loc[level_3090["name"] == name_svs, "level"].iloc[0]

        patch_size = int(8 * (2 ** (7 - float(level))))

        img = Image.open(f"/home/wangzhenyuan/pathology/path_wsi/ori_image_3090/{img_name}")  # 原图

        text_dict = data.loc[i, "text_dict"]
        img_dict = data.loc[i, "img_dict"]
        disease = data.loc[i, "disease"]
        dd = "".join(disease)

        top5_img = topn_dict(eval(img_dict), 5)  # a list ####################################
        top5_text = topn_dict(eval(text_dict), 5)

        ss = csv["pro_index"].value_counts().to_dict()

        for k, v in ss.items():
            if k == top5_img[0]:
                ss[k] = 1
            elif k == top5_img[1]:
                ss[k] = 2
            elif k == top5_img[2]:
                ss[k] = 3
            elif k == top5_img[3]:
                ss[k] = 4
            elif k == top5_img[4]:
                ss[k] = 5
            else:
                ss[k] = 0

        csv['color'] = csv['pro_index'].apply(lambda x: ss[x])

        mask_oo = np.full_like(np.array(img)[:,:,0], 0)

        for i in range(len(csv)):
            w = csv.loc[i, "x_axis"]
            h = csv.loc[i, "y_axis"]
            color = csv.loc[i, "color"]
            mask_oo[(h * patch_size):((h + 1) * patch_size), (w * patch_size):((w + 1) * patch_size)] = mask_oo[(h * patch_size):((h + 1) * patch_size),(w * patch_size):(( w + 1) * patch_size)] + color

        mask = label2color(mask_oo, colors)

        mask_img = Image.fromarray(np.uint8(mask))
        image = Image.blend(img, mask_img, 0.5)  # 生成的图

        text_str = "_".join(top5_text)

        image.save(f"/home/wangzhenyuan/pathology/path_wsi/mm_visual/{name_}_{text_str}_con-{dd}.png", dpi=(900, 900))



    elif os.path.exists(f"/home/wangzhenyuan/pathology/path_wsi/ori_img_4090/{img_name}"):
        name_svs = name[:-4] + ".svs"
        if name_svs in list(level_4090["name"]):
            level = level_4090.loc[level_4090["name"] == name_svs, "level"].iloc[0]
        else:
            name_svs = name[:-5] + ".svs"
            level = level_4090.loc[level_4090["name"] == name_svs, "level"].iloc[0]

        patch_size = int(8 * (2 ** (7 - float(level))))

        img = Image.open(f"/home/wangzhenyuan/pathology/path_wsi/ori_img_4090/{img_name}")  # 原图

        text_dict = data.loc[i, "text_dict"]
        img_dict = data.loc[i, "img_dict"]
        disease = data.loc[i, "disease"]
        dd = "".join(disease)

        top5_img = topn_dict(eval(img_dict), 5)  # a list ####################################
        top5_text = topn_dict(eval(text_dict), 5)

        ss = csv["pro_index"].value_counts().to_dict()

        for k, v in ss.items():
            if k == top5_img[0]:
                ss[k] = 1
            elif k == top5_img[1]:
                ss[k] = 2
            elif k == top5_img[2]:
                ss[k] = 3
            elif k == top5_img[3]:
                ss[k] = 4
            elif k == top5_img[4]:
                ss[k] = 5
            else:
                ss[k] = 0

        csv['color'] = csv['pro_index'].apply(lambda x: ss[x])

        mask_oo = np.full_like(np.array(img)[:,:,0], 0)

        for i in range(len(csv)):
            w = csv.loc[i, "x_axis"]
            h = csv.loc[i, "y_axis"]
            color = csv.loc[i, "color"]
            mask_oo[(h * patch_size):((h + 1) * patch_size), (w * patch_size):((w + 1) * patch_size)] = mask_oo[(h * patch_size):((h + 1) * patch_size),(w * patch_size):(( w + 1) * patch_size)] + color

        mask = label2color(mask_oo, colors)

        mask_img = Image.fromarray(np.uint8(mask))
        image = Image.blend(img, mask_img, 0.5)  # 生成的图

        text_str = "_".join(top5_text)

        image.save(f"/home/wangzhenyuan/pathology/path_wsi/mm_visual/{name_}_{text_str}_con-{dd}.png", dpi=(900, 900))

    elif os.path.exists(f"/home/wangzhenyuan/pathology/path_wsi/ori_img_ins/{img_name}"):
        name_svs = name[:-4] + ".svs"
        if name_svs in list(level_ins["name"]):
            level = level_ins.loc[level_ins["name"] == name_svs, "level"].iloc[0]
        else:
            name_svs = name[:-5] + ".svs"
            level = level_ins.loc[level_ins["name"] == name_svs, "level"].iloc[0]

        patch_size = int(8 * (2 ** (7 - float(level))))

        img = Image.open(f"/home/wangzhenyuan/pathology/path_wsi/ori_img_ins/{img_name}")  # 原图

        text_dict = data.loc[i, "text_dict"]
        img_dict = data.loc[i, "img_dict"]
        disease = data.loc[i, "disease"]
        dd = "".join(disease)

        top5_img = topn_dict(eval(img_dict), 5)  # a list ####################################
        top5_text = topn_dict(eval(text_dict), 5)

        ss = csv["pro_index"].value_counts().to_dict()

        for k, v in ss.items():
            if k == top5_img[0]:
                ss[k] = 1
            elif k == top5_img[1]:
                ss[k] = 2
            elif k == top5_img[2]:
                ss[k] = 3
            elif k == top5_img[3]:
                ss[k] = 4
            elif k == top5_img[4]:
                ss[k] = 5
            else:
                ss[k] = 0

        csv['color'] = csv['pro_index'].apply(lambda x: ss[x])

        mask_oo = np.full_like(np.array(img)[:,:,0], 0)

        for i in range(len(csv)):
            w = csv.loc[i, "x_axis"]
            h = csv.loc[i, "y_axis"]
            color = csv.loc[i, "color"]
            mask_oo[(h * patch_size):((h + 1) * patch_size), (w * patch_size):((w + 1) * patch_size)] = mask_oo[(h * patch_size):((h + 1) * patch_size),(w * patch_size):(( w + 1) * patch_size)] + color

        mask = label2color(mask_oo, colors)

        mask_img = Image.fromarray(np.uint8(mask))
        image = Image.blend(img, mask_img, 0.5)  # 生成的图

        text_str = "_".join(top5_text)

        image.save(f"/home/wangzhenyuan/pathology/path_wsi/mm_visual/{name_}_{text_str}_con-{dd}.png", dpi=(900, 900))

    else:
        print(img_name)
        continue

