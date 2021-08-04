from __future__ import absolute_import, division, print_function

import torch
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import build_norm_layer
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger

# BN_MOMENTUM = 0.1


def dla_build_norm_layer(cfg, out_channels):  
    cfg_ = cfg.copy()
    if cfg_['type'] == 'GN':
        if out_channels % 32 == 0:
            return build_norm_layer(cfg_, out_channels)
        else:
            cfg_['num_groups'] = cfg_['num_groups'] // 2
            return build_norm_layer(cfg_, out_channels)
    else:
        return build_norm_layer(cfg_, out_channels)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, norm_cfg, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation)
        # self.norm1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.norm1 = dla_build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation)
        self.norm2 = dla_build_norm_layer(norm_cfg, planes)[1]
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):

    def __init__(self, in_channels, out_channels, norm_cfg, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2)
        self.norm = dla_build_norm_layer(norm_cfg, out_channels)[1]
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.norm(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):

    def __init__(self,
                 levels,
                 block,
                 in_channels,
                 out_channels,
                 norm_cfg,
                 stride=1,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 dilation=1,
                 root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(
                in_channels, out_channels, norm_cfg, stride, dilation=dilation)
            self.tree2 = block(
                out_channels, out_channels, norm_cfg, 1, dilation=dilation)
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                norm_cfg,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual)
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                norm_cfg,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, norm_cfg, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False),
                dla_build_norm_layer(norm_cfg, out_channels)[1])

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


@BACKBONES.register_module()
class DLANet(nn.Module):
    arch_settings = {
        34: (BasicBlock, (1, 1, 1, 2, 2, 1), (16, 32, 64, 128, 256, 512)),
    }

    def __init__(self, depth, norm_cfg, residual_root=False, type=None):
        super(DLANet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalida depth {depth} for DLA')
        block, levels, channels = self.arch_settings[depth]
        self.channels = channels
        # self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(
                3, channels[0], kernel_size=7, stride=1, padding=3,
                bias=False), dla_build_norm_layer(norm_cfg, channels[0])[1],
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(channels[0], channels[0],
                                            levels[0], norm_cfg)
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], norm_cfg, stride=2)
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            norm_cfg,
            2,
            level_root=False,
            root_residual=residual_root)
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            norm_cfg,
            2,
            level_root=True,
            root_residual=residual_root)
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            norm_cfg,
            2,
            level_root=True,
            root_residual=residual_root)
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            norm_cfg,
            2,
            level_root=True,
            root_residual=residual_root)

    def _make_conv_level(self, inplanes, planes, convs, norm_cfg, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    padding=dilation,
                    bias=False,
                    dilation=dilation),
                dla_build_norm_layer(norm_cfg, planes)[1],
                nn.ReLU(inplace=True)
            ])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')