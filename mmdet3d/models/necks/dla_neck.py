import math
import numpy as np
from mmcv.cnn import ConvModule, build_conv_layer
from torch import nn as nn

from mmdet.models.builder import NECKS


def fill_up_weights(up):
    """Simulated bilinear upsampling kernel.

    Args:
        up (Module): ConvTranspose2d module.
    """
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):
    """IDAUp module for upsamling different scale's features to same scale.

    Args:
        out_channel (int): Number of output channels for DeformConv.
        in_channels_list (List[int]): List of input channels of multi-scale
            feature map.
        up_kernel_size_list (List[int]): List of size of the convolving
            kernel of different scales.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        use_dcn (bool): If True, use DCNv2. Default: True.
    """

    def __init__(
        self,
        out_channel,
        in_channels_list,
        up_kernel_size_list,
        norm_cfg=None,
        use_dcn=True,
    ):
        super(IDAUp, self).__init__()
        self.use_dcn = use_dcn
        for i in range(1, len(in_channels_list)):
            in_channel = in_channels_list[i]
            up_kernel_size = int(up_kernel_size_list[i])
            proj = ConvModule(
                in_channel,
                out_channel,
                3,
                padding=1,
                bias=True,
                conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                norm_cfg=norm_cfg)
            node = ConvModule(
                out_channel,
                out_channel,
                3,
                padding=1,
                bias=True,
                conv_cfg=dict(type='DCNv2') if self.use_dcn else None,
                norm_cfg=norm_cfg)
            up = build_conv_layer(
                dict(type='deconv'),
                out_channel,
                out_channel,
                up_kernel_size * 2,
                stride=up_kernel_size,
                padding=up_kernel_size // 2,
                output_padding=0,
                groups=out_channel,
                bias=False)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, start_level, end_level):
        """Forward function.

        Args:
            layers (list[torch.Tensor]): Features from multiple layers.
            start_level (int): Start layer for feature upsampling.
            end_level (int): End layer for feature upsampling.
        """
        for i in range(start_level + 1, end_level):
            upsample = getattr(self, 'up_' + str(i - start_level))
            project = getattr(self, 'proj_' + str(i - start_level))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, 'node_' + str(i - start_level))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    """DLAUp module for multiple layers feature extraction and fusion.

    Args:
        start_level (int): The start layer.
        channels (List[int]): List of input channels of multi-scale
            feature map.
        scales(List[int]): List of scale of different layers' feature.
        in_channels (NoneType): List of input channels of different scales.
            Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
    """

    def __init__(self,
                 start_level,
                 channels,
                 scales,
                 in_channels=None,
                 norm_cfg=None):
        super(DLAUp, self).__init__()
        self.start_level = start_level
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self, 'ida_{}'.format(i),
                IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j],
                      norm_cfg))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        """Forward function.

        Args:
            layers (tuple[torch.Tensor]): Features from multi-scale layers.

        Returns:
            tuple[torch.Tensor]: Up-sampled features of different layers.
        """
        outs = [layers[-1]]
        for i in range(len(layers) - self.start_level - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            outs.insert(0, layers[-1])
        return outs


@NECKS.register_module()
class DLANeck(nn.Module):
    """DLA Neck.

    Args:
        in_channels (list[int]): List of input channels of multi-scale
            feature map.
        start_level (int): The scale level where upsampling starts. Default: 2.
        end_level (int): The scale level where upsampling ends. Default: 5.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
    """

    def __init__(self,
                 in_channels=[16, 32, 64, 128, 256, 512],
                 start_level=2,
                 end_level=5,
                 norm_cfg=None):
        super(DLANeck, self).__init__()
        self.start_level = start_level
        self.end_level = end_level
        scales = [2**i for i in range(len(in_channels[self.start_level:]))]
        self.dla_up = DLAUp(
            start_level=self.start_level,
            channels=in_channels[self.start_level:],
            scales=scales,
            norm_cfg=norm_cfg)
        self.ida_up = IDAUp(
            in_channels[self.start_level],
            in_channels[self.start_level:self.end_level],
            [2**i for i in range(self.end_level - self.start_level)], norm_cfg)

    def forward(self, x):
        x = list(x)
        x = self.dla_up(x)
        outs = []
        for i in range(self.end_level - self.start_level):
            outs.append(x[i].clone())
        self.ida_up(outs, 0, len(outs))
        return outs[-1]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # In order to be consistent with the source code,
                # reset the ConvTranspose2d initialization parameters
                m.reset_parameters()
                # Simulated bilinear upsampling kernel
                fill_up_weights(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # In order to be consistent with the source code,
                # reset the Conv2d initialization parameters
                m.reset_parameters()
