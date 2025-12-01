# YOLOv8 Implementation

Minimal PyTorch implementation of YOLOv8 architecture, training, and evaluation.

## 1. Environment Setup

### Create Conda Environment

```bash
# Create a new conda environment named 'yolov8' with Python 3.8
conda create -n yolov8 python=3.8 -y

# Activate the environment
conda activate yolov8
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Usage

### Training

Train the model on your dataset. The data directory should contain `images` and `labels` subdirectories.

```bash
python train.py --data /path/to/data --epochs 100 --batch-size 16 --img-size 640
```

### Evaluation

Evaluate the trained model.

```bash
python val.py --data /path/to/data --weights runs/train/exp/weights/epoch_100.pt
```

## Features

1. **Model**: YOLOv8n backbone (CSPDarknet), neck (PANet), and decoupled head.
2. **Loss**: TaskAlignedAssigner, CIoU loss, and DFL (Distribution Focal Loss).
3. **Training**: Custom training loop with cosine learning rate scheduler and mosaic augmentation (partial).
4. **Evaluation**: mAP calculation with COCO-style metrics.
