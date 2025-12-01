import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import bbox_iou, xywh2xyxy, dist2bbox, make_anchors

class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # IoU loss
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = target - tl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(target.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(target.shape) * wr).mean(-1, keepdim=True)

def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)

class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Args:
            pd_scores (Tensor): (b, num_anchors, num_classes)
            pd_bboxes (Tensor): (b, num_anchors, 4)
            anc_points (Tensor): (num_anchors, 2)
            gt_labels (Tensor): (b, n_max_boxes, 1)
            gt_bboxes (Tensor): (b, n_max_boxes, 4)
            mask_gt (Tensor): (b, n_max_boxes, 1) - mask for valid boxes
        """
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            return torch.full_like(pd_scores[..., 0], self.bg_idx).to(torch.long), \
                   torch.zeros_like(pd_bboxes), \
                   torch.zeros_like(pd_scores), \
                   torch.zeros_like(pd_scores[..., 0]), \
                   torch.zeros_like(pd_scores[..., 0])

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)

        target_gt_idx, fg_mask, mask_pos = self.select_topk_candidates(
            mask_pos, align_metric, overlaps)

        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        # pd_scores: (b, anchors, classes)
        # pd_bboxes: (b, anchors, 4)
        # gt_bboxes: (b, max_boxes, 4)
        
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes) # (b, max_boxes, anchors)
        
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt) 
        
        mask_topk = self.select_topk_candidates(mask_in_gts * mask_gt, align_metric, overlaps)
        
        # mask_pos = mask_topk * mask_in_gts * mask_gt
        
        return mask_in_gts * mask_gt, align_metric, overlaps

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9):
        # xy_centers: (anchors, 2)
        # gt_bboxes: (b, n_boxes, 4)
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        # pd_scores: (b, anchors, classes)
        # gt_labels: (b, n_boxes, 1)
        bs, n_boxes, _ = gt_labels.shape
        ind = torch.zeros([2, bs, n_boxes], dtype=torch.long)  # 2, b, n_boxes
        ind[0] = torch.arange(end=bs).view(-1, 1).repeat(1, n_boxes)  # b indices
        ind[1] = gt_labels.squeeze(-1).long()  # class indices
        
        bbox_scores = pd_scores[ind[0], :, ind[1]]  # (b, n_boxes, anchors)
        
        overlaps = bbox_iou(gt_bboxes.unsqueeze(2), pd_bboxes.unsqueeze(1), xywh=False, CIoU=False).squeeze(3).clamp(0)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        
        if mask_gt is not None:
             align_metric *= mask_gt
             overlaps *= mask_gt
             
        return align_metric, overlaps

    def select_topk_candidates(self, mask_pos, align_metric, overlaps):
        # mask_pos: (b, n_boxes, anchors)
        # align_metric: (b, n_boxes, anchors)
        
        # select topk anchors for each gt
        topk_metrics, topk_idxs = torch.topk(align_metric, self.topk, dim=-1, largest=True)
        
        mask_topk = torch.zeros_like(align_metric, dtype=torch.int8, device=align_metric.device)
        for i in range(self.topk):
             mask_topk.scatter_(2, topk_idxs[:, :, i:i+1], 1)
             
        mask_pos = mask_pos * mask_topk
        
        # filter invalid bboxes (where align_metric is 0 because of mask_gt)
        # done implicitly by mask_pos
        
        # If an anchor is assigned to multiple gts, the one with highest align_metric is selected
        # align_metric: (b, n_boxes, anchors)
        # get max metric per anchor
        
        fg_mask = mask_pos.sum(-2) # (b, anchors)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, self.n_max_boxes, 1])  # (b, n_boxes, anchors)
            max_metric_per_anchor = align_metric.max(-2, keepdim=True)[0]
            mask_pos = torch.where(mask_multi_gts, mask_pos * (align_metric == max_metric_per_anchor), mask_pos)
            fg_mask = mask_pos.sum(-2)
            
        target_gt_idx = mask_pos.argmax(-2) # (b, anchors)
        
        return target_gt_idx, fg_mask, mask_pos

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        # gt_labels: (b, n_boxes, 1)
        # gt_bboxes: (b, n_boxes, 4)
        # target_gt_idx: (b, anchors)
        # fg_mask: (b, anchors)
        
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # flatten index
        
        target_labels = gt_labels.long().flatten()[target_gt_idx] # (b, anchors)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx] # (b, anchors, 4)
        
        target_labels.clamp_(0)
        
        # target scores
        # we need alignment metric again? No, we need one-hot of label * alignment metric (soft label)
        # but here we return just labels and bboxes, scores are computed in loss loop or return here?
        # Usually we return soft labels.
        
        # Wait, the official implementation returns `target_scores` which is (b, anchors, classes)
        # Populated with alignment_metric for the target class.
        
        target_scores = torch.zeros((self.bs, fg_mask.shape[1], self.num_classes), 
                                    dtype=torch.float32, 
                                    device=gt_labels.device) # (b, anchors, classes)
        
        # We need the alignment metric values for the selected pairs
        # But I didn't return them from select_topk_candidates...
        # Let's assume we recompute or change flow.
        # Actually simplest is to compute it in forward
        
        return target_labels, target_bboxes, target_scores # partial return, fixing in forward

    # Redefining forward to include metric collection because splitting was messy
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        self.bs = pd_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)
        
        if self.n_max_boxes == 0:
            return torch.full_like(pd_scores[..., 0], self.bg_idx).to(torch.long), \
                   torch.zeros_like(pd_bboxes), \
                   torch.zeros_like(pd_scores), \
                   torch.zeros_like(pd_scores[..., 0]), \
                   torch.zeros_like(pd_scores[..., 0])

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)

        target_gt_idx, fg_mask, mask_pos = self.select_topk_candidates(
            mask_pos, align_metric, overlaps)
        
        # target_gt_idx: (b, anchors) - index of best gt for each anchor
        # fg_mask: (b, anchors) - whether anchor is foreground (1) or background (0)
        
        # Get targets
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx_flat = target_gt_idx + batch_ind * self.n_max_boxes
        
        target_labels = gt_labels.long().flatten()[target_gt_idx_flat] # (b, anchors)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx_flat] # (b, anchors, 4)
        
        # target scores
        target_scores = torch.zeros((self.bs, pd_scores.shape[1], self.num_classes), 
                                    dtype=torch.float32, 
                                    device=gt_labels.device)
        
        # Get alignment metrics for selected pairs
        # align_metric is (b, n_boxes, anchors)
        # mask_pos is (b, n_boxes, anchors)
        
        align_metric = align_metric * mask_pos
        pos_align_metrics = align_metric.amax(axis=1) # (b, anchors)
        pos_overlaps = (overlaps * mask_pos).amax(axis=1) # (b, anchors)
        
        target_scores.scatter_(2, target_labels.unsqueeze(-1), pos_align_metrics.unsqueeze(-1))
        
        return target_bboxes, target_scores, fg_mask.bool()


class v8DetectionLoss(nn.Module):
    def __init__(self, model, device=None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = {'box': 7.5, 'cls': 0.5, 'dfl': 1.5}
        self.stride = model.stride if hasattr(model, 'stride') else torch.tensor([8., 16., 32.])
        self.nc = model.nc
        self.no = model.detect.no
        self.reg_max = model.detect.reg_max
        self.device = device
        
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max - 1, use_dfl=True) # reg_max - 1 ?? usually reg_max is 16, so 0-15
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def forward(self, preds, batch):
        # preds: list of 3 tensors [b, 144, 80, 80] etc.. 
        # But model output depends on training. 
        # If training=True in Detect, it returns (x[0], x[1], x[2]) concatenated?
        # Detect.forward returns `x` list of [b, 64+nc, h, w] if training
        
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds # preds is (pred, x) or x
        
        # feats is list of [b, c, h, w]
        # we need to concat them to [b, anchors, c]
        
        pred_distri, pred_scores = [], []
        
        # make anchors
        anchor_points, batch_anchor_points, stride_tensor = [], [], []
        
        for i, xi in enumerate(feats):
             b, _, h, w = xi.shape
             # xi: [b, 4*reg_max + nc, h, w]
             xi = xi.view(b, self.no, -1).permute(0, 2, 1).contiguous()
             pred_distri.append(xi[..., :self.reg_max * 4])
             pred_scores.append(xi[..., self.reg_max * 4:])
             
             # create anchors
             if i == 0: # create once? No, dynamic shape
                  pass
        
        # We need strides and anchors logic here or reuse `make_anchors`
        # Let's reuse make_anchors from utils
        # But we need strides corresponding to feats
        
        # Strides are [8, 16, 32] usually
        
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        
        pred_scores = torch.cat(pred_scores, 1)
        pred_distri = torch.cat(pred_distri, 1)
        
        # Targets
        gt_labels = batch['cls'] # (B, max_boxes, 1)
        gt_bboxes = batch['bboxes'] # (B, max_boxes, 4)
        
        # mask_gt: Valid boxes. My dataset pads with -1 for cls, or we can check bboxes sum > 0
        mask_gt = (gt_labels > -1) & (gt_bboxes.sum(-1, keepdim=True) > 0)
        
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=torch.float32) * self.stride[0] # h, w (roughly 640)
        # Actually batch['img'].shape[2:]
        
        # Let's assume input gt_bboxes are xywh normalized.
        # Convert to xyxy absolute
        
        h, w = batch['img'].shape[2:]
        gt_bboxes = gt_bboxes * torch.tensor([w, h, w, h], device=self.device)
        gt_bboxes = xywh2xyxy(gt_bboxes)
        
        # Assign targets
        # Decode predicted bounding boxes for assignment
        pred_bboxes = dist2bbox(pred_distri.view(b, -1, 4, self.reg_max).softmax(-1).matmul(self.proj), anchor_points.unsqueeze(0), xywh=False) * stride_tensor.unsqueeze(0) # xyxy

        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes.detach(),
            anchor_points,
            gt_labels,
            gt_bboxes,
            mask_gt
        )
        
        target_scores_sum = max(target_scores.sum(), 1)
        
        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(pred_scores.dtype)).sum() / target_scores_sum # BCE
        
        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask)
            
        loss[0] *= self.hyp['box']
        loss[1] *= self.hyp['cls']
        loss[2] *= self.hyp['dfl']
        
        return loss.sum() * batch['img'].shape[0], torch.cat((loss[0].unsqueeze(0), loss[1].unsqueeze(0), loss[2].unsqueeze(0))).detach()

