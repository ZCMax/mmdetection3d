# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmdet.models.losses import smooth_l1_loss

from mmdet3d.registry import MODELS


@MODELS.register_module()
class DenseDepthL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 beta=1.0,
                 reduction='mean',
                 min_depth=0,
                 max_depth=100,
                 loss_weight=1.0):
        super(DenseDepthL1Loss, self).__init__()
        self.beta = beta
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                depth_pred,
                depth_gt,
                masks=None,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        M = (depth_gt < self.min_depth).to(
            torch.float32) + (depth_gt > self.max_depth).to(torch.float32)
        if masks is not None:
            M += (1. - masks).to(torch.float32)
        M = M == 0.
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_dense_depth = self.loss_weight * smooth_l1_loss(
            depth_pred[M],
            depth_gt[M],
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_dense_depth
