# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, InstanceList
from mmengine.model import BaseModule
from torch import Tensor

from mmdet3d.models.layers import Offset
from mmdet3d.structures.det3d_data_sample import SampleList

INF = 1e8


def aligned_bilinear(tensor, factor, offset='none'):
    """Adapted from AdelaiDet:

    https://github.com/aim-uofa/AdelaiDet/blob/master/adet/utils/comm.py
    """
    assert tensor.dim() == 4
    assert factor >= 1
    assert int(factor) == factor

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode='replicate')
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(
        tensor, size=(oh, ow), mode='bilinear', align_corners=True)
    if offset == 'half':
        tensor = F.pad(
            tensor, pad=(factor // 2, 0, factor // 2, 0), mode='replicate')

    return tensor[:, :, :oh - 1, :ow - 1]


@MODELS.register_module()
class DenseDepthHead(BaseModule):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (Sequence[int] or Sequence[Tuple[int, int]]): Strides of points
            in multiple feature levels. Defaults to (4, 8, 16, 32, 64).
        regress_ranges (Sequence[Tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling.
            Defaults to False.
        center_sample_radius (float): Radius of center sampling.
            Defaults to 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets with
            FPN strides. Defaults to False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Defaults to False.
        conv_bias (bool or str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Defaults to "auto".
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_centerness (:obj:`ConfigDict`, or dict): Config of centerness
            loss.
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer.  Defaults to
            ``norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)``.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 in_channels: int,
                 feat_channels: int,
                 num_levels: int,
                 stacked_convs: int,
                 strides: Sequence[int] = (4, 8, 16, 32, 64),
                 mean_depth_per_level: List[float] = [
                     44.921, 20.252, 11.712, 7.166, 8.548
                 ],
                 std_depth_per_level: List[float] = [
                     24.331, 9.833, 6.223, 4.611, 8.275
                 ],
                 scale_depth: bool = True,
                 depth_scale_init_factor: float = 0.3,
                 use_scale: bool = True,
                 scale_depth_by_focal_lengths_factor: float = 500.,
                 feature_locations_offset: str = 'none',
                 dcn_on_last_conv: bool = False,
                 conv_bias: Union[bool, str] = 'auto',
                 loss_dense_depth: ConfigType = dict(type='DenseDepthL1loss'),
                 init_cfg: dict = None,
                 **kwargs) -> None:
        super(DenseDepthHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.num_levels = num_levels
        self.use_scale = use_scale
        self.strides = strides
        self.mean_depth_per_level = mean_depth_per_level
        self.std_depth_per_level = std_depth_per_level
        self.scale_depth = scale_depth
        self.scale_depth_by_focal_lengths_factor = \
            scale_depth_by_focal_lengths_factor
        self.depth_scale_init_factor = depth_scale_init_factor
        self.feature_locations_offset = feature_locations_offset
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.init_cfg = init_cfg
        self.loss_dense_depth = MODELS.build(loss_dense_depth)
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self._init_depth_convs()
        self._init_predictor()
        if self.use_scale:
            self.scales_depth = nn.ModuleList([
                Scale(init_value=sigma * self.depth_scale_init_factor)
                for sigma in self.std_depth_per_level
            ])
            self.offsets_depth = nn.ModuleList(
                [Offset(init_value=b) for b in self.mean_depth_per_level])

    def _init_depth_convs(self) -> None:
        """Initialize layers of the depth head."""
        self.depth_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.depth_convs.append(
                ConvModule(chn, self.feat_channels, 3, stride=1, padding=1))

    def _init_predictor(self) -> None:
        """Initialize predictor layers of the head."""
        self.conv_depth = nn.ModuleList([
            nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
            for _ in range(self.num_levels)
        ])

    def init_weights(self):
        """Initialize model weights."""
        for m in self.depth_convs.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        for m in self.conv_depth.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:  # depth head may not have bias.
                    torch.nn.init.constant_(m.bias, 0)

    def forward(
            self, x: Tuple[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of each level outputs.

            - cls_scores (list[Tensor]): Box scores for each scale level, \
            each is a 4D-tensor, the channel number is \
            num_points * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for each \
            scale level, each is a 4D-tensor, the channel number is \
            num_points * 4.
            - centernesses (list[Tensor]): centerness for each scale level, \
            each is a 4D-tensor, the channel number is num_points * 1.
        """
        assert len(x) == self.num_levels
        dense_depth = []
        for i, features in enumerate(x):
            for depth_layer in self.depth_convs:
                features = depth_layer(features)
            dense_depth_lvl = self.conv_depth[i](features)
            if self.use_scale:
                dense_depth_lvl = self.offsets_depth[i](
                    self.scales_depth[i](dense_depth_lvl))
            dense_depth.append(dense_depth_lvl)

        return dense_depth

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             **kwargs) -> dict:
        """
        Args:
            x (list[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each item
                contains the meta information of each image and corresponding
                annotations.

        Returns:
            tuple or Tensor: When `proposal_cfg` is None, the detector is a \
            normal one-stage detector, The return value is the losses.

            - losses: (dict[str, Tensor]): A dictionary of loss components.

            When the `proposal_cfg` is not None, the head is used as a
            `rpn_head`, the return value is a tuple contains:

            - losses: (dict[str, Tensor]): A dictionary of loss components.
            - results_list (list[:obj:`InstanceData`]): Detection
              results of each image after the post process.
              Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (:obj:`BaseInstance3DBoxes`): Contains a tensor
                  with shape (num_instances, C), the last dimension C of a
                  3D box is (x, y, z, x_size, y_size, z_size, yaw, ...), where
                  C >= 7. C = 7 for kitti and C = 9 for nuscenes with extra 2
                  dims of velocity.
        """

        outs = self(x)
        batch_gt_dense_depth = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_dense_depth.append(data_sample.gt_depth_map.data)

        loss_inputs = outs + (batch_gt_dense_depth, batch_img_metas)
        losses = self.loss_by_feat(*loss_inputs)

        return losses

    def loss_by_feat(
        self,
        dense_depth: List[Tensor],
        batch_gt_dense_depth: List[Tensor],
        batch_img_metas: List[dict],
    ) -> Dict[str, Tensor]:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        cam2imgs = torch.stack([
            dense_depth[0].new_tensor(img_meta['cam2img'])
            for img_meta in batch_img_metas
        ])

        # (B, 1, H, W)
        gt_dense_depth = torch.stack(batch_gt_dense_depth)
        inv_cam2imgs = cam2imgs.inverse()

        # Upsample to the input image shape
        dense_depth = [
            aligned_bilinear(
                x, factor=stride,
                offset=self.feature_locations_offset).squeeze(1)
            for x, stride in zip(dense_depth, self.strides)
        ]

        if self.scale_depth:
            assert inv_cam2imgs is not None
            pixel_size = torch.norm(
                torch.stack([inv_cam2imgs[:, 0, 0], inv_cam2imgs[:, 1, 1]],
                            dim=-1),
                dim=-1)
            scaled_pixel_size = (
                pixel_size * self.scale_depth_by_focal_lengths_factor).reshape(
                    -1, 1, 1)
            dense_depth = [x / scaled_pixel_size for x in dense_depth]

        loss_dict = {}
        for lvl, x in enumerate(dense_depth):
            loss_lvl = self.loss_dense_depth(x, gt_dense_depth.tensor)
            loss_lvl = loss_lvl / (torch.sqrt(2)**lvl)  # Is sqrt(2) good?
            loss_dict.update({f'loss_dense_depth_lvl_{lvl}': loss_lvl})
        return loss_dict

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_panoptic_seg` and `gt_pts_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        predictions = self(x)
        return predictions
