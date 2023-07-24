# Jurassic World Remake: Bringing Ancient Fossils Back to Life via Zero-Shot Long Image-to-Image Translation </h1>
[Project Page](https://tinyurl.com/skull2animal) | [ACM MM]() | [arxiv]() | [Dataset](https://drive.google.com/drive/folders/1feHrsMNokvXYao_8UkjuJRfaAgmj_FhQ?usp=sharing)

This repo is a fork of a modified version of Stable Diffuion, optimized to use less VRAM than the original. The original repo can be found [here](https://github.com/basujindal/stable-diffusion). 


## Installation
<!--  bash -->
```bash
git clone https://github.com/alexmartin1722/Revive-2I.git
cd Revive-2I
conda env create -f environment.yaml
conda activate ldm
pip install transformers==4.19.2 diffusers invisible-watermark
```

## Weights
The weights used in this paper are the `stable-diffusion-v1-4` downloadable from [HuggingFace](https://huggingface.co/CompVis). 

Once the weights are downloaded they can be placed in `models/ldm/stable-diffusion-v1/model.ckpt` or you can specify the path to the weights using the `--ckpt` argument.

## Dataset
The dataset can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1feHrsMNokvXYao_8UkjuJRfaAgmj_FhQ?usp=sharing). When downloaded, place inside this directory in `data/`. For example the dog dataset would be in `data/skull2dog`. 

## Usage
All code can be run out of optimized_txt_guid_i2i.py. There are two options for running the code 
1. Single image translation
```bash
python optimized_txt_guid_i2i.py "prompt" --source-img <IMG> 
```
2. Batch image translation
```bash
python optimized_txt_guid_i2i.py "prompt" --source-img-dir <DIR>
```

The code used to generate the results in the paper is:
```bash
python optimized_txt_guid_i2i.py "class" --source-img-dir data/skull2dog/testA/ --ddim_steps 100 --strength 0.95 --seed 42
```

## Evaluation
To evaluate the code, first classifier the images in the ouput folder

``bash
python classification/classifier.py --image-dir <DIR> --output <DIR>/labels.csv
```

Then run the evaluation script with the DIR

```bash
python eval/scores.py --generated-dir <DIR> --target-dir data/skull2dog/testB --class-csv <DIR>
```

You can also skip the classification step by providing your own HuggingFace API key, but if you have rate limits, use the classification script. 

```bash
python eval/scores.py --generated-dir <DIR> --target-dir data/skull2dog/testB --api-key XXX
```

## Citation 
When citing this dataset please cite the following papers:

The living animals:
```
@misc{choi2018stargan,
      title={StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation}, 
      author={Yunjey Choi and Minje Choi and Munyoung Kim and Jung-Woo Ha and Sunghun Kim and Jaegul Choo},
      year={2018},
      eprint={1711.09020},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

The skulls and anything else: 
```
@inproceedings{
anonymous2023jurassic,
title={Jurassic World Remake: Bringing Ancient Fossils Back to Life via Zero-Shot Long Image-to-Image Translation},
author={Anonymous},
booktitle={31st ACM International Conference on Multimedia (ACM Multimedia) Brave New Ideas Track},
year={2023},
url={https://openreview.net/forum?id=xwVMFxYjrz}
}
```
