
<h2 align="center">MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting </h2>
<p align="center">
  <a href="https://www.pnu-cvsp.com/members/inhwan"><strong>In-Hwan Jin</strong></a>
  ·  
  <a href="https://www.pnu-cvsp.com/members/hyeong-ju"><strong>Hyeongju Mun</strong></a>
  ·  
  <strong>Joonsoo Kim</strong></a>
  ·    
  <strong>Kugjin Yun</strong>
  ·  
  <a href="https://www.pnu-cvsp.com/prof"><strong>Kyeongbo Kong</strong></a>
  <br>
</p>


## Contents

1. [Setup](#-Setup)
2. [Preprocess Datasets](#-Preprocess-Datasets)
3. [Stage 1: Expert Training](#-Stage-1)
4. [Stage 2: Router Training](#-Stage-2)
5. [Lightfield Rendering](#-Lightfield-Rendering)

<br><br>

## Setup

### Environment Setup
Installation through pip is recommended. First, set up your Python environment:
```shell
conda create -n MoE-GS python=3.9
conda activate MoE-GS
```
Make sure to install CUDA and PyTorch versions that match your CUDA environment. We've tested on NVIDIA RTX A6000 with PyTorch  version 2.0.1.
Please refer https://pytorch.org/ for further information.

```shell
pip install torch
```

The remaining packages can be installed with:

```shell
pip install --upgrade setuptools cython wheel
pip install -r requirements.txt
```

<br><br>

## Preprocess Datasets

For dataset preprocessing, we follow [STG](https://github.com/oppo-us-research/SpacetimeGaussians) for both the [N3V](https://github.com/facebookresearch/Neural_3D_Video) and Technicolor datasets.

### Neural 3D Video Dataset
First, download the dataset from [here](https://github.com/facebookresearch/Neural_3D_Video). You will need colmap environment for preprocess.
To setup dataset preprocessing environment, run scrips:
```shell
./scripts/env_setup.sh
```

To preprocess dataset, run script:
```shell
./scripts/preprocess_all_n3v.sh <path to dataset>
```

### Technicolor dataset
Download the dataset from [here](https://www.interdigital.com/data_sets/light-field-dataset).
To setup dataset preprocessing environment, run scrips:

```shell
./scripts/preprocess_all_techni.sh <path to dataset>
```

Please refer [STG](https://github.com/oppo-us-research/SpacetimeGaussians.git) for further information.

<br><br>

## Stage 1: Expert Training
We use **[Ex4DGS](https://github.com/juno181/Ex4DGS)**, **[E-D3DGS](https://github.com/JeongminB/E-D3DGS)**, **[4DGaussians](https://github.com/hustvl/4DGaussians)**, and **[STG](https://github.com/oppo-us-research/SpacetimeGaussians)** as candidate experts for **MoE-GS**.  
All experts are pretrained using their original configurations, except for **STG**.  
For **STG** on N3V Dataset, we split the frames into **0–149** and **150–299** to handle GPU memory limits,  
and modify the **feature splatting** process to use **spherical harmonics (SHs)** with the **Ex4DGS rasterizer**.

### STG Model Directory Structure

The pretrained **STG** models are organized as follows:
```
<Path to STG model>
|---<scene>/
|   |---<scene>_0to149/
|   |---<scene>_150to299/
```

<br><br>

## Stage 2: Router Training

### N3V Dataset

You can train MoE-GS(n=3,4) by running the following command:

```
python train_E4.py --config "configs/N3V/<scene>.json" \
    --source_path <location>/<scene> \
    --model_path <path to Ex4DGS model>/<scene> \
    --emb_path <path to E-D3DGS model>/<scene> \
    --stg_path <path to STG model>/<scene> \
    --fgaussian_path <path to 4DGaussians model>/<scene> \
    --save_path <path to save model>

python train_E3.py --config "configs/N3V/<scene>.json" \
    --source_path <location>/<scene> \
    --model_path <path to Ex4DGS model>/<scene> \
    --emb_path <path to E-D3DGS model>/<scene> \
    --fgaussian_path <path to 4DGaussians model>/<scene> \
    --save_path <path to save model>

```

You can render MoE-GS(n=3,4) by running the following command:

```

python render_E4.py --skip_train \
    --source_path <location>/<scene> \
    --save_path <path to save model> \
    --iteration <2000|5000>

python render_E3.py --skip_train \
    --source_path <location>/<scene> \
    --save_path <path to save model> \
    --iteration <2000|5000>

```

### Technicolor Dataset
You can train MoE-GS(n=3) by running the following command:

```

python train_E3_tech.py --config "configs/techni/<scene>.json" \
    --source_path <location>/<scene> \
    --model_path <path to Ex4DGS model>/<scene> \
    --emb_path <path to E-D3DGS model>/<scene> \
    --stg_path <path to STG model>/<scene> \
    --save_path <path to save model>

```

You can render MoE-GS(n=3) by running the following command:

```
python render_E3_tech.py --skip_train \
    --source_path <location>/<scene> \
    --save_path <path to save model> \
    --iteration <2000|5000>

```