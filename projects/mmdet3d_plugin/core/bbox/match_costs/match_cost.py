import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmdet.core.bbox.iou_calculators import bbox_overlaps

@MATCH_COST.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1., reg_cost=1.0):
        self.weight = weight
        self.reg_cost = reg_cost

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        # 手动放大距离
        bbox_pred_c = bbox_pred.clone()
        bbox_pred_c[..., :2] = bbox_pred_c[..., :2] * self.reg_cost
        gt_bboxes_c = gt_bboxes.clone()
        gt_bboxes_c[..., :2] = gt_bboxes_c[..., :2] * self.reg_cost
        bbox_cost = torch.cdist(bbox_pred_c, gt_bboxes_c, p=1)
        return bbox_cost * self.weight