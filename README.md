# Sparse Gradient Merging for Test-Time Adaptation of VLMs
This is the implementation for WACV Submission \#1672.

## Requirements 
Python 3.9 

Pytorch 2.0.1

## Usage

### Dataset Installation

Corruptions datasets (CIFAR-10-C, CIFAR-100-C and Tiny-ImageNet-C) can be downloaded using the following command:
```bash
python download_datasets.py --data_dir ./data/
```

Fine-grained classifications datasets can be downloaded following the procedure from [CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA/blob/main/DATASETS.md) repository.

The path to the directory containing the datasets should be added in `./utils/datasets.py`.

### Experiments

First, the environment can be setup with the following:
```bash
conda env create -f environment.yaml
```

Files in `./bash` directory can then be used to reproduce the results found in the paper.