import os
import openslide
import json
import numpy as np
from tissue_utils import get_tissue
from PIL import Image
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
#colors_nn =[[0,0,128],[144,190,224],[230,241,243],[255,223,146],[244,111,68],[255,182,193],[219,49,36],[138,43,226]]# 浅蓝色 淡白色 淡黄色 橙红色 粉色 深红色 紫色
#[152,251,152]白绿
colors_nn =[[0,0,128],[244,111,68],[255,223,146],[255,182,193],[144,190,224],[230,241,243],[152,251,152],[219,49,36]] # 将每个WSI分成7类。类别数量可调整！

def label2color(img,colors):
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(colors[label])
    return img_color


level_3090 = pd.read_csv("/home/wangzhenyuan/pathology/path_wsi/wsi_level_3090.csv",index_col=0)
level_4090 = pd.read_csv("/home/wangzhenyuan/pathology/path_wsi/wsi_level_4090.csv",index_col=0)
level_ins = pd.read_csv("/home/wangzhenyuan/pathology/path_wsi/wsi_level_ins.csv",index_col=0)
prototype = np.load("../multi_modality/prototype_220.npy") # the prototype matrix !

for svs in os.listdir("/home/wangzhenyuan/pathology/path_wsi/ori_image_3090/"):
    name = svs[:-8]

    colors = colors_nn

    name_ = name[:-1]

    if (level_3090["name"] == (name + ".svs")).sum()==1:
        level = level_3090.loc[level_3090["name"] == (name + ".svs"), "level"].iloc[0]
    else:
        level = level_3090.loc[level_3090["name"] == (name_ + ".svs"), "level"].iloc[0]


    img = Image.open(f"/home/wangzhenyuan/pathology/path_wsi/ori_image_3090/{svs}") # 原图
    ll = pd.read_csv(f"/home/wangzhenyuan/pathology/index/check_220/prototype_xian_all/{name}.csv", index_col=0) # csv
    ss = ll["pro_index"].value_counts().to_dict()
    if len(ss) > 7:
        model = KMeans(random_state=420, n_clusters=7,n_init="auto")
    else:
        model = KMeans(random_state=420, n_clusters=len(ss), n_init="auto")

    labell = model.fit_predict(prototype[list(ss.keys()),:])

    for ind, (k, v) in enumerate(ss.items()):
        ss[k] = labell[ind]+1

    ll['color'] = ll['pro_index'].apply(lambda x: ss[x])
    ## change 给color 排序
    color_new = ll["color"].value_counts()
    for ind, (k, v) in enumerate(color_new.items()):
        color_new[k] = ind + 1
    ll["color_new"] = ll['color'].apply(lambda x: color_new[x])

    patch_size = int(8 * (2 ** (7 - float(level))))

    mask = np.full_like(np.array(img)[:,:,0], 0)
    for i in range(len(ll)):
        w = ll.loc[i, "x_axis"]
        h = ll.loc[i, "y_axis"]
        color = ll.loc[i, "color_new"]
        mask[(h * patch_size):((h + 1) * patch_size), (w * patch_size):((w + 1) * patch_size)] = mask[(h * patch_size):((h + 1) * patch_size),(w * patch_size):((w + 1) * patch_size)] + color
    mask = label2color(mask,colors)
    mask_img = Image.fromarray(np.uint8(mask))
    image = Image.blend(img, mask_img, 0.5)
    image.save(f"/home/wangzhenyuan/pathology/path_wsi/concat_image_0103/{name}_con.png", dpi=(900, 900))

for svs in os.listdir("/home/wangzhenyuan/pathology/path_wsi/ori_img_4090/"):
    name = svs[:-8]

    colors = colors_nn
    name_ = name[:-1]

    if (level_4090["name"] == (name + ".svs")).sum()==1:
        level = level_4090.loc[level_4090["name"] == (name + ".svs"), "level"].iloc[0]
    else:
        level = level_4090.loc[level_4090["name"] == (name_ + ".svs"), "level"].iloc[0]

    img = Image.open(f"/home/wangzhenyuan/pathology/path_wsi/ori_img_4090/{svs}")  # 原图
    ll = pd.read_csv(f"/home/wangzhenyuan/pathology/index/check_220/prototype_xian_all/{name}.csv",
                     index_col=0)  # csv
    ss = ll["pro_index"].value_counts().to_dict()

    if len(ss) > 7:
        model = KMeans(random_state=420, n_clusters=7,n_init="auto")
    else:
        model = KMeans(random_state=420, n_clusters=len(ss), n_init="auto")

    labell = model.fit_predict(prototype[list(ss.keys()), :])

    for ind, (k, v) in enumerate(ss.items()):
        ss[k] = labell[ind] + 1

    ll['color'] = ll['pro_index'].apply(lambda x: ss[x])
    ## change 给color 排序
    color_new = ll["color"].value_counts()
    for ind, (k, v) in enumerate(color_new.items()):
        color_new[k] = ind + 1
    ll["color_new"] = ll['color'].apply(lambda x: color_new[x])

    patch_size = int(8 * (2 ** (7 - float(level))))

    mask = np.full_like(np.array(img)[:, :, 0], 0)
    for i in range(len(ll)):
        w = ll.loc[i, "x_axis"]
        h = ll.loc[i, "y_axis"]
        color = ll.loc[i, "color_new"]
        mask[(h * patch_size):((h + 1) * patch_size), (w * patch_size):((w + 1) * patch_size)] = mask[
                                                                                                 (h * patch_size):((
                                                                                                                               h + 1) * patch_size),
                                                                                                 (w * patch_size):((
                                                                                                                               w + 1) * patch_size)] + color
    mask = label2color(mask, colors)
    mask_img = Image.fromarray(np.uint8(mask))
    image = Image.blend(img, mask_img, 0.5)
    image.save(f"/home/wangzhenyuan/pathology/path_wsi/concat_image_0103/{name}_con.png", dpi=(900, 900))

# break
for svs in os.listdir("/home/wangzhenyuan/pathology/path_wsi/ori_img_ins/"):
    name = svs[:-8]

    colors = colors_nn
    name_ = name[:-1]

    if (level_ins["name"] == (name + ".svs")).sum()==1:
        level = level_ins.loc[level_ins["name"] == (name + ".svs"), "level"].iloc[0]
    else:
        level = level_ins.loc[level_ins["name"] == (name_ + ".svs"), "level"].iloc[0]



    img = Image.open(f"/home/wangzhenyuan/pathology/path_wsi/ori_img_ins/{svs}")  # 原图
    ll = pd.read_csv(f"/home/wangzhenyuan/pathology/index/check_220/prototype_institute_1018_220/{name}.csv",
                     index_col=0)  # csv
    ss = ll["pro_index"].value_counts().to_dict()

    if len(ss) > 7:
        model = KMeans(random_state=420, n_clusters=7,n_init="auto")
    else:
        model = KMeans(random_state=420, n_clusters=len(ss), n_init="auto")

    labell = model.fit_predict(prototype[list(ss.keys()), :])

    for ind, (k, v) in enumerate(ss.items()):
        ss[k] = labell[ind] + 1

    ll['color'] = ll['pro_index'].apply(lambda x: ss[x])
    ## change 给color 排序
    color_new = ll["color"].value_counts()
    for ind, (k, v) in enumerate(color_new.items()):
        color_new[k] = ind + 1
    ll["color_new"] = ll['color'].apply(lambda x: color_new[x])

    patch_size = int(8 * (2 ** (7 - float(level))))

    mask = np.full_like(np.array(img)[:, :, 0], 0)
    for i in range(len(ll)):
        w = ll.loc[i, "x_axis"]
        h = ll.loc[i, "y_axis"]
        color = ll.loc[i, "color_new"]
        mask[(h * patch_size):((h + 1) * patch_size), (w * patch_size):((w + 1) * patch_size)] = mask[
                                                                                                 (h * patch_size):((
                                                                                                                               h + 1) * patch_size),
                                                                                                 (w * patch_size):((
                                                                                                                               w + 1) * patch_size)] + color
    mask = label2color(mask, colors)
    mask_img = Image.fromarray(np.uint8(mask))
    image = Image.blend(img, mask_img, 0.5)
    image.save(f"/home/wangzhenyuan/pathology/path_wsi/concat_image_0103/{name}_con.png", dpi=(900, 900))
