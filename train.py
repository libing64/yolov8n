import argparse
import os
import torch
import torch.optim as optim
from tqdm import tqdm
import yaml

from yolov8.model import YOLOv8n
from yolov8.loss import v8DetectionLoss
from yolov8.dataset import create_dataloader

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = YOLOv8n(nc=args.nc).to(device)
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.937, weight_decay=0.0005)
    
    # Scheduler
    # linear_lr = lambda x: (1 - x / args.epochs) * (1.0 - 0.01) + 0.01
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_lr)
    # Cosine is standard for YOLOv8
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)

    # Loss
    compute_loss = v8DetectionLoss(model, device=device)
    
    # Dataloader
    train_loader = create_dataloader(os.path.join(args.data, 'images'), 
                                     os.path.join(args.data, 'labels'), 
                                     batch_size=args.batch_size, 
                                     img_size=args.img_size)
    
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
            ckpt_path = f"runs/train/exp/weights/epoch_{epoch+1}.pt"
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='path to dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--nc', type=int, default=80)
    args = parser.parse_args()
    
    train(args)

