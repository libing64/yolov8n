import argparse
import os
import torch
import torch.optim as optim
from tqdm import tqdm
from yolov8.model import YOLOv8n
from yolov8.loss import v8DetectionLoss
from yolov8.coco_dataset import create_coco_dataloader

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    # COCO has 80 classes
    model = YOLOv8n(nc=80).to(device)
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.937, weight_decay=0.0005)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)

    # Loss
    compute_loss = v8DetectionLoss(model, device=device)
    
    # Dataloader
    # Hardcoded paths based on user request /home/libing/dataset/coco2017
    data_root = '/home/libing/dataset/coco2017'
    train_img_dir = os.path.join(data_root, 'train2017')
    train_ann_file = os.path.join(data_root, 'annotations/instances_train2017.json')
    
    print(f"Loading data from {train_img_dir} and {train_ann_file}")
    
    train_loader = create_coco_dataloader(train_img_dir, 
                                          train_ann_file,
                                          batch_size=args.batch_size, 
                                          img_size=args.img_size,
                                          workers=args.workers)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0
        
        for batch in pbar:
            imgs = batch['img'].to(device)
            batch['cls'] = batch['cls'].to(device)
            batch['bboxes'] = batch['bboxes'].to(device)
            
            # Forward
            preds = model(imgs)
            
            # Loss
            loss, loss_items = compute_loss(preds, batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'box': f"{loss_items[0].item():.4f}", 'cls': f"{loss_items[1].item():.4f}"})
            
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            ckpt_path = f"runs/train/coco/weights/epoch_{epoch+1}.pt"
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    
    train(args)

