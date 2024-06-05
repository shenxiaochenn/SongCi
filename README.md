SongCi :dragon_face: <img src="docs/AI_songci.png" width="200px" align="right" />
===========

SongCi is a multi-modal deep learning model tailored for forensic pathological analyses.
The architecture consists of three main parts, i.e., an imaging encoder for WSI feature extraction, a text encoder for the embedding of gross key findings as well as diagnostic queries, and a multi-modal fusion block that integrates the embeddings of WSI and gross key findings to align with those of the diagnostic queries.


[中文](https://github.com/shenxiaochenn/SongCi/blob/master/README_CN.md) ｜ English

## Large-vocabulary forensic pathological analyses via prototypical cross-modal contrastive learning


<div align=center>
<img src="docs/songci.png" width="800px" />
  
 ### The framework of SongCi and studied large-vocabulary, multi-center datasets.
</div>


## Updates:
* 05/06/2024: We are working on refining the code updates for the SongCi model.
## Installation:

**Pre-requisites**:
```bash
python 3.9+
CUDA 12.1
pip
ANACONDA
```


After activating the virtual environment, you can install specific package requirements as follows:
```python
pip install -r requirements.txt
```

**Optional**: Conda Environment Setup For those who prefer using Conda:
```bash
conda create --name songci python=3.9.7
conda activate songci

```

## WSI preprocessing and the content of text(gross key findings & forensic pathology diagnosis)

### WSI
**NOTE**: In practical scenarios, a single slide can encompass a variety of tissue types. To reduce the labeling time required by forensic scientists, we have adopted a straightforward approach by delineating the area with a simple rectangular boundary. Conversely, regions comprising a single tissue type are segmented without the need for explicit labeling.


```bash
svs_datasets/
  ├── slide_1.svs
  ├── slide_2.svs
  ├── slide_3.svs
  ├── slide_3.json 
  ├── slide_4.svs
  └── ...

```
Here we give an example.
```bash
python patch_tmp.py
```

This will split each WSI at the specifwied magnification by looping through it, while the **JSON** file in this is an annotation file (containing the 4 coordinates of the annotation box).
Finally, we will get the patch-level datasets!

```bash
patch_datasets/
  ├── slide_1/
    ├── slide_1-0_1_.png
    ├── slide_1-0_2_.png
    ├── slide_1-0_3_.png
    └── ...
  ├── slide_2/
    ├── slide_2-0_1_.png
    ├── slide_2-0_2_.png
    ├── slide_2-0_3_.png
    └── ...
  ├── slide_3/
  ├── slide_4/
  └── ...

```
### gross key findings & forensic pathology diagnosis

We provide sample text here in one of our cohorts.

The gross key finding is a paragraph  and forensic pathology diagnosis are text segments delineated by `/`.

```bash
text_xianjiaotong.csv
```
slide_name    | gross key findings | forensic pathology diagnosis
-------- | ----- | -----
slide_1  | The mucosa is smooth, complete and pink, there is no bleeding, ulceration or perforation. | Gastrointestinal congestion/Gastrointestinal tissue autolysis
slide_2  | There is a tear in the bottom of the heart, which leads inward to the left ventricle, the myocardium is dark red, and the coronary artery is stiff.  | Coronary atherosclerotic heart disease/Myocardial infarction with heart rupture/Pericardial tamponade
slide_3  | The envelope of both kidneys is complete and easy to peel, the surface and section are brown red, and the boundary between skin and medulla is clear. | Renal autolysis/Congestion of kidney 

##  train of prototypical WSI encoder
**NOTE**: In our study, the CUDA version is 12.1 and python is 3.9. The computational experiments should be conducted on a system equipped with a minimum of eight NVIDIA GeForce RTX 3090 graphics cards. If you use fp16 for training,  in our study, it's unstable.
```python
python -m torch.distributed.launch --nproc_per_node=8  prototype_encoder/main_prototype.py   --use_bn_in_head True  --use_pre_in_head True  --use_fp16 False  --batch_size_per_gpu 96 --data_path /path/to/WSI_patch/train --output_dir /path/to/saving_dir
```
results:
```bash
/path/to/saving_dir/
  ├──  log.txt 
  ├── checkpoint.pth
  ├── queue.pth
  └── ...
```
###   WSI patch generator both prototype-based & instance-based

If you implement prototype-based generation, use the `patch_generation/guided_diffusion/get_ssl_models.py` file.

If you implement instance-based generation, use the `patch_generation/guided_diffusion/get_sl_models.py` file.

default: prototype-based

**train**

IN the `patch_generation` folder, just run:

```bash
sh train.sh 
```
**sampling**:

* prototype-based : the default loop iterates over all prototypes

```bash
sh sample_prototype.sh
```

* instance-based: choose the instance what you like 


```bash
sh sample.sh
```

## Connection

:v: If you have a keen interest in forensic pathology and wish to contribute to this field, whether it be through data provision, inquiries about algorithm implementation, innovative suggestions, or a desire for comprehensive communication and collaboration, we encourage you to contact us. We eagerly anticipate engaging in discussions with you!:laughing: :laughing: :laughing:

* **Zhenyuan Wang**  Key Laboratory of National Ministry of Health for Forensic Sciences, School of Medicine \& Forensics, Health Science Center,Xi’an Jiaotong University  Email: wzy218@xjtu.edu.cn
* **Chunfeng Lian** School of Mathematics and Statistics, Xi'an Jiaotong University  Email: chunfeng.lian@xjtu.edu.cn
## Citation

## Related Projects

[dino](https://github.com/facebookresearch/dino)

[guided-diffusion](https://github.com/openai/guided-diffusion)


