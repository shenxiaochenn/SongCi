SongCi :dragon_face: <img src="docs/AI_songci.png" width="200px" align="right" />
===========

SongCi is a multi-modal deep learning model tailored for forensic pathological analyses.
The architecture consists of three main parts, i.e., an imaging encoder for WSI feature extraction, a text encoder for the embedding of gross key findings as well as diagnostic queries, and a multi-modal fusion block that integrates the embeddings of WSI and gross key findings to align with those of the diagnostic queries.


[中文](https://github.com/shenxiaochenn/SongCi/blob/master/README_CN.md) ｜ English

## Large-vocabulary forensic pathological analyses via prototypical cross-modal contrastive learning


<div align=center>
<img src="docs/demo.png" width="800px" />
  
 ### The framework of SongCi and studied large-vocabulary, multi-center datasets.
</div>


## Updates:
* 05/06/2024: We are working on refining the code updates for the SongCi model.
* 28/10/2024： We’ve provide checkpoints for each section, which you can now access by simply requesting them from us [hugging face](https://huggingface.co/shenxiaochen/SongCi) .
* 26/01/2025：quick start use this vision-language model for forensic pathology diagnosis. [Demo of how to use SongCi](https://github.com/shenxiaochenn/SongCi/blob/master/cross-modal-decoder/fusion_demo/multi_modality_diagnosis.ipynb) And how to get the prototype embedding of a WSI. [Demo of how to get the prototype of each WSI](https://github.com/shenxiaochenn/SongCi/blob/master/prototype_encoder/songci_demo/how%20to%20get%20prototype%20csv.ipynb). (Note: For safety! please get the checkpoint through the [hugging face](https://huggingface.co/shenxiaochen/SongCi) !!!)
* 04/02/2025：We show the forensic pathology diagnostic results for each sample in the internal cohort, as well as the final numerical assessment results.[Reproduced results of Internal Cohort](https://github.com/shenxiaochenn/SongCi/blob/master/cross-modal-decoder/fusion_demo/Demo_of_the_internal_cohort(Xian%20Jiaotong).ipynb) and [Demo of how to use SongCi](https://github.com/shenxiaochenn/SongCi/blob/master/cross-modal-decoder/fusion_demo/multi_modality_diagnosis.ipynb)
* 18/02/2025： We show some [post-mortem WSIs of forensic pathology](https://drive.google.com/drive/folders/1fz7BMuUBH2L4U8Pk0jtyA50TavYef5Gx?usp=drive_link) (the same deceased and public available) so that you can visualize the differences between forensic pathology WSIs and clinical oncology WSIs. If you have a strong interest in forensic pathology, we welcome you to contact us at once, and we will provide more real forensic cases.🤗🤗🤗


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
git clone https://github.com/shenxiaochenn/SongCi.git
cd SongCi
pip install -r requirements.txt
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
python patch_tmp.py #https://github.com/shenxiaochenn/SongCi/blob/master/WSI_preprocessing/patch_tmp.py
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
text_xianjiaotong.csv # in the WSI_preprocessing folder
```
slide_name    | gross key findings | forensic pathology diagnosis
-------- | ----- | -----
slide_1  | The mucosa is smooth, complete and pink, there is no bleeding, ulceration or perforation. | Gastrointestinal congestion/Gastrointestinal tissue autolysis
slide_2  | There is a tear in the bottom of the heart, which leads inward to the left ventricle, the myocardium is dark red, and the coronary artery is stiff.  | Coronary atherosclerotic heart disease/Myocardial infarction with heart rupture/Pericardial tamponade
slide_3  | The envelope of both kidneys is complete and easy to peel, the surface and section are brown red, and the boundary between skin and medulla is clear. | Renal autolysis/Congestion of kidney 

##  prototypical contrastive learning

* how to train the prototypical self-supervised contrastive learning?
  
**NOTE**: In our study, the CUDA version is 12.1 and python is 3.9. The computational experiments should be conducted on a system equipped with a minimum of eight NVIDIA GeForce RTX 3090 graphics cards. If you use fp16 for training,  in our study, it's unstable. 

Here we use 3149743 high-level patches to train the encoder , it needs almost 2 weeks - 3 weeks.
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

time : 4 gpu cards  1 week !

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

###   WSI segmentaion

First we convert each WSI into a table. In the tabel, we are able to know which prototype each patch belongs to, the exact value of similarity and the coordinates of this patch in the WSI.

```bash
python wsi_seg/prototype_index.py
```
you will get the  WSI table.

For example:

patch_name   | WSI_name | x_axis | y_axis |pro_index | sim_value
-------- | ----- | -----| ----- | -----| ----- 
patch_1  | WSI_1 | 0| 0 | 2| 0.9623
patch_2  | WSI_1 | 1| 0 | 56| 0.8958
patch_3  | WSI_1 | 1| 2 | 3| 0.9703

then just run,  and you will get the final  segmentation results

```bash
python wsi_seg/wsi_seg_prototype.py
```

## cross-modality contrastive learning

how to train the modality fusion block
 
* train

time : 1 gpu card 1 -2 days ! 


```bash
python main_fusion.py  --data_path xxx  --depth 2 --checkpoint xxx(prototype-encoder) --output_dir xxx --gate True --noise_ratio 0.5 --saveckp_freq 100 --warmup_epochs 50
```
At the inferrence time, a `csv` file will be returned containing the  forensic diagnostic results predicted by the model for the samples provided.

* inference 
```bash
python score_modality.py  --checkpoint xxx(prototype-encoder)  --fusion_checkpoint xxx(fusion block)   --data_path xxx --threshold 0.88  --out_name xx
```

###  Multi-modality explainability

We will count the scores for each prototype and each word and turn them into a table.

 WSI_name | disease | img_dict |text_dict 
 ----- | -----| ----- | -----
 WSI_1 | The hemorrhage under the scalp| {prototype:score} |  {word:score} 
 WSI_2 | Gastrointestinal congestion| {prototype:score} | {word:score} 
 WSI_3 | Gastrointestinal tissue autolysis| {prototype:score} | {word:score} 

* For a list of samples ~
```bash
python visual_modality_index.py
```
Examples: here we show the top 5 prototypes and top 5 words

<div align=center>
<img src="docs/explain.png" width="600px" />
  
 **Multi-modality attention visualization of SongCi**
</div>

## Connection

:v: If you have a keen interest in forensic pathology and wish to contribute to this field, whether it be through data provision, inquiries about algorithm implementation, innovative suggestions, or a desire for comprehensive communication and collaboration, we encourage you to contact us. We eagerly anticipate engaging in discussions with you!:laughing: :laughing: :laughing:

* **Zhenyuan Wang**  Key Laboratory of National Ministry of Health for Forensic Sciences, School of Medicine \& Forensics, Health Science Center,Xi’an Jiaotong University  Email: wzy218@xjtu.edu.cn
* **Chunfeng Lian** School of Mathematics and Statistics, Xi'an Jiaotong University  Email: chunfeng.lian@xjtu.edu.cn
*  **Chen Shen** Xi'an Jiaotong University   Email: shenxiaochen@stu.xjtu.edu.cn
## Citation
If you find SongCi useful for your your research and applications, please cite using this BibTeX:

```bibtex
@misc{shen2024largevocabularyforensicpathologicalanalyses,
      title={Large-vocabulary forensic pathological analyses via prototypical cross-modal contrastive learning}, 
      author={Chen Shen and Chunfeng Lian and Wanqing Zhang and Fan Wang and Jianhua Zhang and Shuanliang Fan and Xin Wei and Gongji Wang and Kehan Li and Hongshu Mu and Hao Wu and Xinggong Liang and Jianhua Ma and Zhenyuan Wang},
      year={2024},
      eprint={2407.14904},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2407.14904}, 
}
```
## Related Projects

[dino](https://github.com/facebookresearch/dino)

[guided-diffusion](https://github.com/openai/guided-diffusion)

[flamingo](https://github.com/lucidrains/flamingo-pytorch)
