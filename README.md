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

### Training on COCO Dataset

Train the model on the COCO2017 dataset. The script expects data at `/home/libing/dataset/coco2017` with the following structure:
```
/home/libing/dataset/coco2017/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

```bash
python train_coco.py --epochs 100 --batch-size 16 --img-size 640
```

The training script will automatically run validation at the end of each epoch and print mAP@0.5.

### Evaluation on COCO Dataset

Evaluate the trained model on the COCO validation set:

```bash
python val_coco.py --weights runs/train/coco/weights/epoch_100.pt
```

## Features

1. **Model**: YOLOv8n backbone (CSPDarknet), neck (PANet), and decoupled head.
2. **Loss**: TaskAlignedAssigner, CIoU loss, and DFL (Distribution Focal Loss).
3. **Training**: Custom training loop with cosine learning rate scheduler and mosaic augmentation (partial).
4. **Evaluation**: mAP calculation with COCO-style metrics.
