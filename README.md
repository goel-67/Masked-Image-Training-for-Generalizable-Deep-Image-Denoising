# Masked Image Training for Generalizable Deep Image Denoising

## Overview
Image denoising is essential in computer vision for enhancing downstream applications like object recognition and medical diagnostics. Traditional models trained on synthetic noise (e.g., Gaussian) often fail to generalize to complex real-world noise. This project builds upon the masked image training framework by Chen et al., using a SwinIR transformer-based architecture with input and attention masking strategies to improve model robustness and generalization.

## Masked Image Training
The core idea involves minimal modifications to the original SwinIR architecture, specifically the **input mask** operation and **attention masks**.

## Instructions for My Custom Version
### Cloning the Repository
Clone my modified version of the repository:
```bash
git clone https://github.com/yourusername/MaskedDenoising.git
```

### Installation
The installation of required packages remains the same as the original project:
```bash
pip install -r requirement.txt
```

### Key Changes
My version builds upon the original by modifying the following files:
- **main_test_swinir.py**
- **main_train_psnr.py**
- **dataset_masked_denoising.py**
- **utils_image.py**

All significant changes have been commented in the code for clarity.

### Configuration Details
Before running or training, ensure the `options` JSON file is configured as needed:
- Set `"gpu_ids": [0,1,2,3]` if 4 GPUs are used.
- Set `"dataroot_H": "trainsets/trainH"` if the high-quality dataset path is `trainsets/trainH`.
- **Input mask**: Set `"if_mask"` and `"mask1"`, `"mask2"` (lines 32-34). The masking ratio will be randomly sampled between `mask1` and `mask2`.
- **Attention mask**: Set `"use_mask"` and `"mask_ratio1"`, `"mask_ratio2"` (lines 68-70). The attention mask ratio can be specified as a range or a fixed value.

### Running My Experimental Model
Use the following command to run the new experimental model:
```bash
!python main_test_swinir.py \
    --model_path masked_denoising/input_80_90/models/new.pth \
    --name input_mask_80_90/McM_poisson_20_mymodel \
    --opt model_zoo/input_mask_80_90.json \
    --folder_gt testset/McM/HR \
    --folder_lq testset/McM/McM_poisson_20
```

### Running the Original Model
To run the original model provided in the repository, use the following command:
```bash
python main_test_swinir.py \
    --model_path model_zoo/input_mask_80_90.pth \
    --name input_mask_80_90/McM_poisson_20 \
    --opt model_zoo/input_mask_80_90.json \
    --folder_gt testset/McM/HR \
    --folder_lq testset/McM/McM_poisson_20
```

### Training
**Note**: Training from scratch is not recommended as it is a complex and computationally expensive process. It is highly encouraged to use the provided pretrained models for most use cases. However, if you wish to train the model, follow the instructions below:

To train the model with `DataParallel`:
```bash
python main_train_psnr.py --opt options/masked_denoising/input_mask_80_90.json
```

To train the model with `DistributedDataParallel` (using 4 GPUs):
```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_train_psnr.py --opt options/masked_denoising/input_mask_80_90.json --dist True
```

Ensure you have configured the `options` JSON file with your specific settings before training.

### Notebook
For reference, a notebook named `project.ipynb` has also been included. This notebook contains some test runs that I conducted locally, which can help illustrate how to use the model and understand its performance.

---
**Acknowledgements**
This project builds on the code and concepts from the original [Masked Image Training for Generalizable Deep Image Denoising](https://github.com/haoyuc/MaskedDenoising) by Chen et al. (2023). We extend our gratitude to the authors for making their work accessible.

### References
```BibTex
@InProceedings{Chen_2023_CVPR,
    author    = {Chen, Haoyu and Gu, Jinjin and Liu, Yihao and Magid, Salma Abdel and Dong, Chao and Wang, Qiong and Pfister, Hanspeter and Zhu, Lei},
    title     = {Masked Image Training for Generalizable Deep Image Denoising},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {1692-1703}
}
```

