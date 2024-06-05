import numpy as np
import os
from PIL import Image
import openslide
from tissue_utils import get_tissue
import json


def wsi_patch(imgname,patch_size,level):
    dir = imgname.split(".")[0]
    os.makedirs(dir)
    slide = openslide.OpenSlide(f"institute_20/{imgname}")
    Wh = np.zeros((len(slide.level_dimensions), 2))
    for i in range(len(slide.level_dimensions)):
        Wh[i, :] = slide.level_dimensions[i]
    w_count = int(Wh[level, 0] // patch_size)
    h_count = int(Wh[level, 1] // patch_size)
    slide_thumbnail = slide.get_thumbnail(slide.level_dimensions[level])
    mask, _ = get_tissue(np.array(slide_thumbnail), contour_area_threshold=100000)
    mask_img = Image.fromarray(mask)
    get_cut = 0
    for w in range(0, w_count):
        for h in range(0, h_count):
            m_crop = np.array(mask_img.crop((w * patch_size, h * patch_size, ((w + 1) * patch_size), ((h + 1) * patch_size))))
            if np.sum(m_crop > 0) > 0.4*(patch_size)**2:
                fig = np.array(slide.read_region((w * patch_size * np.power(2, level), h * patch_size * np.power(2, level)), level, (patch_size, patch_size)))[:, :, :3]
                img = Image.fromarray(fig.astype('uint8')).convert('RGB')
                img.save(os.path.join(dir, f"{dir}-{w}_{h}_.png"), quality=95)
                get_cut += 1
            else:
                continue
    print(f"there are {get_cut} patches!")
    ff = f"{dir}-finish!"
    return print(ff)


def wsi_patch_multi(filename,imgname,patch_size,level):
    f = open(f"institute_20/{filename}", 'r')
    content = f.read()
    label = json.loads(content)
    slide = openslide.OpenSlide(f"institute_20/{imgname}")
    slide_thumbnail = slide.get_thumbnail(slide.level_dimensions[level])
    mask, _ = get_tissue(np.array(slide_thumbnail), contour_area_threshold=100000)
    mask_img = Image.fromarray(mask)
    for ind, i in enumerate(label['GroupModel']["Labels"]):
        dir_ = filename.split("/")[-1].split(".")[0] + str(ind)
        os.makedirs(dir_)
        ax = i['Coordinates']
        x1 = int(float(ax[0].split(",")[0]))
        y1 = int(float(ax[0].split(",")[1]))
        x2 = int(float(ax[1].split(",")[0]))
        y2 = int(float(ax[1].split(",")[1]))
        patch_level = patch_size*np.power(2,level)
        w_count = int((x2-x1) // patch_level)
        h_count = int((y2-y1) // patch_level)
        x_crop = int(float(ax[0].split(",")[0])/np.power(2,level))
        y_crop = int(float(ax[0].split(",")[1])/np.power(2,level))
        get_cut = 0
        for w in range(0, w_count):
            for h in range(0, h_count):
                m_crop = np.array(mask_img.crop(((w * patch_size + x_crop), (h * patch_size + y_crop), ((w + 1) * patch_size + x_crop), ((h + 1) * patch_size + y_crop))))
                if np.sum(m_crop > 0) > 0.4*(patch_size)**2:
                    fig = np.array(slide.read_region((w * patch_size * np.power(2, level)+x1, h * patch_size * np.power(2, level)+y1), level, (patch_size, patch_size)))[:, :, :3]
                    img = Image.fromarray(fig.astype('uint8')).convert('RGB')
                    img.save(os.path.join(dir_, f"{dir_}-{w}_{h}_.png"), quality=95)
                    get_cut += 1
                else:
                    continue
        print(f"there are {get_cut} patches!")
    return print("finish!")
