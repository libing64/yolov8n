import argparse
import os
import torch
import torch.optim as optim
from tqdm import tqdm
from yolov8.model import YOLOv8n
from yolov8.loss import v8DetectionLoss
from yolov8.coco_dataset import create_coco_dataloader
import numpy as np
from yolov8.utils import non_max_suppression, xywh2xyxy, bbox_iou

def compute_ap(recall, precision):
    """Compute average precision."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return ap

def validate_model(model, val_loader, device):
    """Run validation and return mAP@0.5."""
    model.eval()
    stats = []
    iou_thres = 0.5
    conf_thres = 0.001
    
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['img'].to(device)
            batch['cls'] = batch['cls'].to(device)
            batch['bboxes'] = batch['bboxes'].to(device)
            
            # Reconstruct targets
            targets = []
            h, w = imgs.shape[2:]
            for i in range(imgs.shape[0]):
                t_cls = batch['cls'][i].view(-1)
                t_bbox = batch['bboxes'][i].view(-1, 4)
                
                valid = t_cls > -1
                t_cls = t_cls[valid]
                t_bbox = t_bbox[valid]
                
                t_bbox = t_bbox * torch.tensor([w, h, w, h], device=t_bbox.device)
                t_bbox = xywh2xyxy(t_bbox)
                
                targets.append({'cls': t_cls, 'bboxes': t_bbox})
            
            # Forward
            preds = model(imgs)
            preds = preds[0] if isinstance(preds, tuple) else preds
            preds = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=0.6)
            
            # Match predictions to ground truth
            for i, pred in enumerate(preds):
                t_cls = targets[i]['cls']
                t_bbox = targets[i]['bboxes']
                
                if len(pred) == 0:
                    if len(t_cls):
                        stats.append((torch.zeros(0, dtype=torch.bool), torch.Tensor(), torch.Tensor(), t_cls.cpu()))
                    continue
                
                if len(t_cls):
                    # Compute IoU matrix (n_pred x n_gt)
                    # bbox_iou now handles shapes correctly with minimum/maximum
                    iou = bbox_iou(pred[:, :4], t_bbox, xywh=False).squeeze(-1)
                    
                    # Ensure iou is 2D
                    if iou.dim() == 0:
                        iou = iou.view(1, 1)
                    elif iou.dim() == 1:
                        if len(pred) == 1:
                            iou = iou.unsqueeze(0)
                        else:
                            iou = iou.unsqueeze(1)
                    
                    # Match
                    matches = []
                    if iou.numel() > 0 and iou.max() > iou_thres:
                        x = torch.where(iou > iou_thres)
                        if x[0].shape[0]:
                            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
                            if x[0].shape[0] > 1:
                                matches = matches[matches[:, 2].argsort()[::-1]]
                                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                                matches = matches[matches[:, 2].argsort()[::-1]]
                                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    else:
                        matches = np.zeros((0, 3))
                    
                    if len(matches) == 0:
                        matches = np.zeros((0, 3))
                    
                    matches = torch.from_numpy(matches) if isinstance(matches, np.ndarray) else matches
                    
                    detected = []
                    tps = torch.zeros(len(pred), dtype=torch.bool)
                    
                    if matches.shape[0]:
                        for m in matches:
                            pred_idx = int(m[0])
                            gt_idx = int(m[1])
                            if gt_idx not in detected:
                                if pred[pred_idx, 5] == t_cls[gt_idx]:
                                    tps[pred_idx] = True
                                    detected.append(gt_idx)
                    
                    stats.append((tps.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), t_cls.cpu()))
                else:
                    stats.append((torch.zeros(len(pred), dtype=torch.bool), pred[:, 4].cpu(), pred[:, 5].cpu(), torch.Tensor()))
    
    # Compute mAP
    if len(stats) == 0:
        return 0.0
        
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        tp, conf, pred_cls, target_cls = stats
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
        
        unique_classes = np.unique(target_cls)
        ap = []
        for c in unique_classes:
            i = pred_cls == c
            n_gt = (target_cls == c).sum()
            n_p = i.sum()
            
            if n_p == 0 and n_gt == 0:
                continue
            elif n_p == 0 or n_gt == 0:
                ap.append(0)
            else:
                fpc = (1 - tp[i]).cumsum()
                tpc = (tp[i]).cumsum()
                
                recall = tpc / (n_gt + 1e-16)
                precision = tpc / (tpc + fpc)
                
                ap.append(compute_ap(recall, precision))
        
        return np.mean(ap) if len(ap) > 0 else 0.0
    else:
        return 0.0

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = YOLOv8n(nc=80).to(device)
    
    # Optimizer - lower learning rate to prevent gradient explosion
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.937, weight_decay=0.0005)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0001)

    # Loss
    compute_loss = v8DetectionLoss(model, device=device)
    
    # Dataloader
    # data_root = '/home/libing/dataset/coco2017'
    data_root = '/home/libing/dataset/tiny_coco_dataset/tiny_coco'#tiny_coco
    train_img_dir = os.path.join(data_root, 'train2017')
    train_ann_file = os.path.join(data_root, 'annotations/instances_train2017.json')
    val_img_dir = os.path.join(data_root, 'val2017')
    val_ann_file = os.path.join(data_root, 'annotations/instances_val2017.json')
    
    print(f"Loading training data from {train_img_dir}")
    
    train_loader = create_coco_dataloader(train_img_dir, 
                                          train_ann_file,
                                          batch_size=args.batch_size, 
                                          img_size=args.img_size,
                                          workers=args.workers)
    
    print(f"Loading validation data from {val_img_dir}")
    val_loader = create_coco_dataloader(val_img_dir,
                                        val_ann_file,
                                        batch_size=args.batch_size,
                                        img_size=args.img_size,
                                        workers=2)  # Fewer workers for val
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_loss = 0
        debug_printed = False
        
        for batch_idx, batch in enumerate(pbar):
            imgs = batch['img'].to(device)
            batch['cls'] = batch['cls'].to(device)
            batch['bboxes'] = batch['bboxes'].to(device)
            
            
            # Forward
            preds = model(imgs)
            
            # Loss
            loss, loss_items = compute_loss(preds, batch)
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Loss is NaN or Inf at batch {batch_idx}")
                continue
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'box': f"{loss_items[0].item():.4f}", 'cls': f"{loss_items[1].item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        
        print(f"\nEpoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Validation
        print(f"Running validation...")
        mAP = validate_model(model, val_loader, device)
        print(f"Epoch {epoch+1} - mAP@0.5: {mAP:.4f}\n")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            ckpt_path = f"runs/train/coco/weights/epoch_{epoch+1}.pt"
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'mAP': mAP,
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
