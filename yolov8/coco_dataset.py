import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class COCODataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size=640, transform=None):
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
        
        print(f"Loading annotations from {ann_file}...")
        with open(ann_file, 'r') as f:
            self.coco = json.load(f)
            
        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
            
        self.img_ids = list(self.images.keys())
        
        # COCO to YOLO class mapping (continuous 0-79)
        # COCO IDs are not continuous (1-90, some missing)
        categories = sorted(self.coco['categories'], key=lambda x: x['id'])
        self.cat_id_to_cls = {cat['id']: i for i, cat in enumerate(categories)}
        self.cls_to_cat_id = {i: cat['id'] for i, cat in enumerate(categories)}
        
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.images[img_id]
        file_name = img_info['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            # Handle missing images or corruptions gracefully? 
            # For now, raise error or return dummy
            raise Exception(f"Could not load image {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h0, w0 = img.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1: 
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
            
        # Labels
        anns = self.annotations.get(img_id, [])
        labels = []
        
        for ann in anns:
            if 'bbox' not in ann:
                continue
            cat_id = ann['category_id']
            if cat_id not in self.cat_id_to_cls:
                continue
            cls = self.cat_id_to_cls[cat_id]
            
            x, y, w, h = ann['bbox'] # COCO is top-left x,y, w, h
            
            # Center x, y
            cx = x + w / 2
            cy = y + h / 2
            
            labels.append([cls, cx, cy, w, h])
            
        labels = np.array(labels, dtype=np.float32) if len(labels) else np.zeros((0, 5), dtype=np.float32)
        
        # Letterbox padding
        h, w = img.shape[:2]
        dw, dh = self.img_size - w, self.img_size - h
        dw, dh = dw / 2, dh / 2
        
        # cv2.copyMakeBorder expects integer parameters
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Check final size
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
             img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            
        if len(labels):
            # Absolute xywh (original) -> Absolute xywh (resized)
            labels[:, 1] = labels[:, 1] * r + left # cx
            labels[:, 2] = labels[:, 2] * r + top # cy
            labels[:, 3] = labels[:, 3] * r # w
            labels[:, 4] = labels[:, 4] * r # h
            
            # Normalize
            labels[:, 1:] /= self.img_size
            
        # Transpose image to CHW
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        
        return torch.from_numpy(img).float() / 255.0, torch.from_numpy(labels)

    @staticmethod
    def collate_fn(batch):
        # Re-use the one from dataset.py or duplicate here
        # Let's duplicate to be self-contained or import
        # Importing is better but for now let's copy to avoid circular deps if any
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs, 0)
        
        # Pad labels
        max_boxes = max([l.shape[0] for l in labels])
        padded_labels = []
        padded_cls = []
        padded_bboxes = []
        
        for l in labels:
            n = l.shape[0]
            if n > 0:
                cls = l[:, 0:1]
                bboxes = l[:, 1:5]
            else:
                cls = torch.zeros((0, 1))
                bboxes = torch.zeros((0, 4))
                
            pad_n = max_boxes - n
            if pad_n > 0:
                cls = torch.cat([cls, torch.zeros((pad_n, 1)) - 1], 0)
                bboxes = torch.cat([bboxes, torch.zeros((pad_n, 4))], 0)
            
            padded_cls.append(cls)
            padded_bboxes.append(bboxes)
            
        padded_cls = torch.stack(padded_cls, 0)
        padded_bboxes = torch.stack(padded_bboxes, 0)
        
        return {
            'img': imgs,
            'cls': padded_cls,
            'bboxes': padded_bboxes,
            'batch_idx': torch.arange(len(batch))
        }

def create_coco_dataloader(img_dir, ann_file, batch_size=16, img_size=640, workers=4):
    dataset = COCODataset(img_dir, ann_file, img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=COCODataset.collate_fn)
    return loader

