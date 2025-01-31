# Task-Agnostic Language Model Watermarking via High Entropy Passthrough Layers
This is the notebook for reproducing the results of the "Task-Agnostic Language Model Watermarking via High Entropy Passthrough Layers" paper accepted for publication in AAAI2025.

<p align="center">
<center>
<img src="https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/ptl/cover.png" alt="alt text" width="1000">
</center>
</p> 

## Download and extract the code


```python
!wget https://vbdai-notebooks.obs.cn-north-4.myhuaweicloud.com/ptl/code.zip 
!unzip -qo code.zip
```

## Dependencies
To use this package, please install the following dependencies using your
favourite package manager:
```
torch
datasets
transformers
scikit-learn
scipy
pandas
numpy
safetensors
matplotlib
ipython
tqdm
```

## Getting Started

First download the required models and datasets


```python
!chmod u+x download.sh
!./download.sh
```

This may take a while as both the dataset and model are very large.

To watermark a Bert Passthrough model, use the following command:


```python
!python watermark_passthrough.py --dataset_name=processed_book_corpus_full --max_steps=10000 --eval_steps=2000 --eval_beginning=False --run_name=working-bert-passthrough-2468-layer-10k-steps-train --watermark_layers="1 3 5 7 9" --watermark_multipliers="1 1 1 1 1"
```

Note: GPU training is strongly recommended. 

## Weights & Biases

This package uses Weights & Biases to track training and evaluation metrics.
You can get setup on Weights & Biases at: https://docs.wandb.ai/quickstart
