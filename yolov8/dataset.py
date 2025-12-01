import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])
        label_path = os.path.join(self.label_dir, self.img_files[index].rsplit('.', 1)[0] + '.txt')
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise Exception(f"Could not load image {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h0, w0 = img.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1: 
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
            
        # Load labels
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    l = line.strip().split()
                    if len(l) == 5:
                        # class, x, y, w, h
                        cls = int(l[0])
                        x, y, w, h = map(float, l[1:])
                        labels.append([cls, x, y, w, h])
        
        labels = np.array(labels, dtype=np.float32) if len(labels) else np.zeros((0, 5), dtype=np.float32)
        
        # Letterbox padding
        h, w = img.shape[:2]
        dw, dh = self.img_size - w, self.img_size - h
        dw, dh = dw / 2, dh / 2
        
        if dw > 0 or dh > 0:
            img = cv2.copyMakeBorder(img, int(dh), int(dh), int(dw), int(dw), cv2.BORDER_CONSTANT, value=(114, 114, 114))
            
        # Adjust labels to padded image
        # Labels are normalized in original image? Usually yes.
        # If normalized, we need to convert to absolute, adjust for resize and padding, then normalize again or keep absolute.
        # YOLOv8 Assigner in my implementation expects absolute xyxy? 
        # In loss.py: gt_bboxes = gt_bboxes * torch.tensor([w, h, w, h], ...)
        # So it expects normalized xywh.
        
        # We need to adjust normalized labels because padding changed the image effective size.
        # Effective image size is (img_size, img_size) now.
        # Original content is in (w, h) centered.
        
        if len(labels):
            # Normalized xywh -> Absolute xywh on resized image
            labels[:, 1] = labels[:, 1] * w0 * r + dw # x
            labels[:, 2] = labels[:, 2] * h0 * r + dh # y
            labels[:, 3] = labels[:, 3] * w0 * r # w
            labels[:, 4] = labels[:, 4] * h0 * r # h
            
            # Absolute -> Normalized on (img_size, img_size)
            labels[:, 1:] /= self.img_size
            
        # Transpose image to CHW
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        
        return torch.from_numpy(img).float() / 255.0, torch.from_numpy(labels)

    @staticmethod
    def collate_fn(batch):
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs, 0)
        
        # Pad labels
        max_boxes = max([l.shape[0] for l in labels])
        padded_labels = []
        padded_cls = []
        padded_bboxes = []
        
        for l in labels:
            n = l.shape[0]
            # l is (n, 5) -> cls, x, y, w, h
            if n > 0:
                cls = l[:, 0:1]
                bboxes = l[:, 1:5]
            else:
                cls = torch.zeros((0, 1))
                bboxes = torch.zeros((0, 4))
                
            pad_n = max_boxes - n
            if pad_n > 0:
                cls = torch.cat([cls, torch.zeros((pad_n, 1)) - 1], 0) # -1 for padding
                bboxes = torch.cat([bboxes, torch.zeros((pad_n, 4))], 0)
            
            padded_cls.append(cls)
            padded_bboxes.append(bboxes)
            
        padded_cls = torch.stack(padded_cls, 0) # (B, max_boxes, 1)
        padded_bboxes = torch.stack(padded_bboxes, 0) # (B, max_boxes, 4)
        
        return {
            'img': imgs,
            'cls': padded_cls,
            'bboxes': padded_bboxes,
            'batch_idx': torch.arange(len(batch)) # (B,)
        }

def create_dataloader(img_dir, label_dir, batch_size=16, img_size=640, workers=4):
    dataset = YOLODataset(img_dir, label_dir, img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=YOLODataset.collate_fn)
    return loader

