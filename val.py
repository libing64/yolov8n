import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import json
from yolov8.model import YOLOv8n
from yolov8.dataset import create_dataloader
from yolov8.utils import non_max_suppression, xywh2xyxy, bbox_iou

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap

def validate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = YOLOv8n(nc=args.nc).to(device)
    if args.weights:
        ckpt = torch.load(args.weights, map_location=device)
        model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.eval()
    
    # Dataloader
    val_loader = create_dataloader(os.path.join(args.data, 'images'), 
                                   os.path.join(args.data, 'labels'), 
                                   batch_size=args.batch_size, 
                                   img_size=args.img_size)
    
    # Metrics
    stats = [] # [(correct, conf, cls_id, cls_gt)]
    
    iou_thres = 0.5
    conf_thres = 0.001
    
    pbar = tqdm(val_loader, desc="Validating")
    for batch in pbar:
        imgs = batch['img'].to(device)
        targets = []
        # Reconstruct targets for this batch (absolute xyxy)
        h, w = imgs.shape[2:]
        for i in range(imgs.shape[0]):
            mask = batch['batch_idx'] == i
            if mask.sum() > 0:
                # cls, xywh
                t_cls = batch['cls'][i].view(-1)
                t_bbox = batch['bboxes'][i].view(-1, 4)
                
                # Filter padding
                valid = t_cls > -1
                t_cls = t_cls[valid]
                t_bbox = t_bbox[valid]
                
                # Normalize to absolute
                t_bbox = t_bbox * torch.tensor([w, h, w, h], device=device)
                t_bbox = xywh2xyxy(t_bbox)
                
                targets.append({'cls': t_cls, 'bboxes': t_bbox})
            else:
                targets.append({'cls': torch.tensor([], device=device), 'bboxes': torch.tensor([], device=device)})
        
        # Forward
        with torch.no_grad():
            preds = model(imgs)
            # preds is (y, x) tuple
            preds = preds[0] if isinstance(preds, tuple) else preds
            
            # NMS
            preds = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=0.6)
            
        # Match
        for i, pred in enumerate(preds):
            # pred: (n, 6) -> xyxy, conf, cls
            t_cls = targets[i]['cls']
            t_bbox = targets[i]['bboxes']
            
            if len(pred) == 0:
                if len(t_cls):
                    stats.append((torch.zeros(0, 1, dtype=torch.bool), torch.Tensor(), torch.Tensor(), t_cls))
                continue
            
            if len(t_cls):
                # IoU
                iou = bbox_iou(pred[:, :4], t_bbox, xywh=False) # (n_pred, n_gt)
                
                # Match
                matches = []
                if iou.max() > iou_thres:
                    # Simple matching: greedy
                    # For correct mAP need to check all thresholds
                    # Here just check 0.5
                    
                    # matches: (pred_idx, gt_idx)
                    # We need to assign each GT to at most one Pred, and each Pred to at most one GT
                    # But actually we check if pred has any match
                    
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
                correct = matches[:, 1].long() if matches.shape[0] else [] # GT indices matched
                
                # Per prediction correct/incorrect
                # We need to know for each detection if it is TP or FP
                
                detected = []
                tps = torch.zeros(len(pred), dtype=torch.bool)
                
                if matches.shape[0]:
                    # matches is [pred_idx, gt_idx, iou]
                    for m in matches:
                        pred_idx = int(m[0])
                        gt_idx = int(m[1])
                        if gt_idx not in detected:
                            if pred[pred_idx, 5] == t_cls[gt_idx]:
                                tps[pred_idx] = True
                                detected.append(gt_idx)
                                
                stats.append((tps, pred[:, 4], pred[:, 5], t_cls))
                
            else:
                stats.append((torch.zeros(len(pred), dtype=torch.bool), pred[:, 4], pred[:, 5], torch.Tensor()))
                
    # Compute mAP
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        tp, conf, pred_cls, target_cls = stats
        # Sort by confidence
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
        
        # Unique classes
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
                # Accumulate FPs and TPs
                fpc = (1 - tp[i]).cumsum()
                tpc = (tp[i]).cumsum()
                
                recall = tpc / (n_gt + 1e-16)
                precision = tpc / (tpc + fpc)
                
                ap.append(compute_ap(recall, precision))
                
        print(f"mAP@0.5: {np.mean(ap):.4f}")
    else:
        print("No matches found.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='path to dataset')
    parser.add_argument('--weights', type=str, default=None, help='path to weights')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--nc', type=int, default=80)
    args = parser.parse_args()
    
    validate(args)

