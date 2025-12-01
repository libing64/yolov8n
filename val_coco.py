import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from yolov8.model import YOLOv8n
from yolov8.coco_dataset import create_coco_dataloader
from yolov8.utils import non_max_suppression, xywh2xyxy, bbox_iou

def compute_ap(recall, precision):
    # Same as val.py
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    method = 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def validate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = YOLOv8n(nc=80).to(device)
    if args.weights:
        print(f"Loading weights from {args.weights}")
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.eval()
    
    # Dataloader
    data_root = '/home/libing/dataset/coco2017'
    val_img_dir = os.path.join(data_root, 'val2017')
    val_ann_file = os.path.join(data_root, 'annotations/instances_val2017.json')
    
    print(f"Loading data from {val_img_dir} and {val_ann_file}")
    
    val_loader = create_coco_dataloader(val_img_dir, 
                                        val_ann_file, 
                                        batch_size=args.batch_size, 
                                        img_size=args.img_size,
                                        workers=args.workers)
    
    # Metrics
    stats = [] 
    
    iou_thres = 0.5
    conf_thres = 0.001
    
    debug_count = 0
    total_preds = 0
    total_gts = 0
    
    pbar = tqdm(val_loader, desc="Validating")
    for batch in pbar:
        imgs = batch['img'].to(device)
        batch['cls'] = batch['cls'].to(device)
        batch['bboxes'] = batch['bboxes'].to(device)
        
        # Reconstruct targets
        targets = []
        h, w = imgs.shape[2:]
        for i in range(imgs.shape[0]):
            mask = batch['batch_idx'] == i
            if mask.sum() > 0:
                t_cls = batch['cls'][i].view(-1)
                t_bbox = batch['bboxes'][i].view(-1, 4)
                
                valid = t_cls > -1
                t_cls = t_cls[valid]
                t_bbox = t_bbox[valid]
                
                t_bbox = t_bbox * torch.tensor([w, h, w, h], device=t_bbox.device)
                t_bbox = xywh2xyxy(t_bbox)
                
                targets.append({'cls': t_cls, 'bboxes': t_bbox})
            else:
                targets.append({'cls': torch.tensor([], device=device), 'bboxes': torch.tensor([], device=device)})
        
        with torch.no_grad():
            preds = model(imgs)
            preds = preds[0] if isinstance(preds, tuple) else preds
            preds = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=0.6)
        
        # Debug first batch
        if debug_count == 0:
            print(f"\nDebug validation:")
            print(f"  Model output shape: {preds[0].shape if len(preds[0]) > 0 else 'no detections'}")
            print(f"  Num detections in batch: {[len(p) for p in preds]}")
            if len(preds[0]) > 0:
                print(f"  Sample detection: {preds[0][0]}")
            debug_count += 1
            
        for i, pred in enumerate(preds):
            total_preds += len(pred)
            total_gts += len(targets[i]['cls'])
            t_cls = targets[i]['cls']
            t_bbox = targets[i]['bboxes']
            
            if len(pred) == 0:
                if len(t_cls):
                    stats.append((torch.zeros(0, 1, dtype=torch.bool), torch.Tensor(), torch.Tensor(), t_cls))
                continue
            
            if len(t_cls):
                iou = bbox_iou(pred[:, :4], t_bbox, xywh=False)
                matches = []
                if iou.max() > iou_thres:
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
                
                matches = torch.from_numpy(matches)
                
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
                
        print(f"mAP@0.5: {np.mean(ap):.4f}")
        print(f"Total predictions: {total_preds}, Total ground truths: {total_gts}")
    else:
        print(f"No matches found. Total predictions: {total_preds}, Total ground truths: {total_gts}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None, help='path to weights')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    
    validate(args)

