from mask_patch import wsi_patch
from mask_patch import wsi_patch_multi
# filename   pwd of json
# imgname    pwd of svs
# patch_size the size of a patch
# level 0:40x  1:20x   2:10x
# savepath: pwd of your path
from tqdm import tqdm
import os

for i in tqdm(os.listdir("/mnt/data/shenxiaochen_data/datasets/")):
    if i.endswith("svs"):
        name = i[:-4]
        if os.path.exists(f"/mnt/data/shenxiaochen_data/datasets/{name}.json"):
            print(name)
            path = wsi_patch_multi(filename=f"{name}.json",imgname=f"{name}.svs",patch_size=256,level=2)
        else:
            path = wsi_patch(imgname=f"{name}.svs",patch_size=256,level=2)
    else:
        continue





    


