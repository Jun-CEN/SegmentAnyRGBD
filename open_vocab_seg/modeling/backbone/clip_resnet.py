# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

from collections import OrderedDict
import torch
import torch.nn as nn
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, 3, padding=1 * dilation, bias=False, dilation=dilation
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, width=64, strides=[2, 1, 2, 2, 2], multi_grid=[1, 1, 1]):
        super().__init__()

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(strides[0]) if strides[0] > 1 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0], stride=strides[1])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=strides[2])
        self.layer3 = self._make_layer(width * 4, layers[2], stride=strides[3])
        self.layer4 = self._make_layer(
            width * 8, layers[3], stride=strides[4], dilations=multi_grid
        )
        self.num_features = [width * 4, width * 8, width * 16, width * 32]

    def _make_layer(self, planes, blocks, stride=1, dilations=None):
        if dilations is None:
            dilations = [1] * blocks
        layers = [Bottleneck(self._inplanes, planes, stride, dilation=dilations[0])]
        self._inplanes = planes * Bottleneck.expansion

        for i in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, dilation=dilations[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [
                (self.conv1, self.bn1),
                (self.conv2, self.bn2),
                (self.conv3, self.bn3),
            ]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        output = {}
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)  # 1/4,1/4
        x = self.layer1(x)
        output["res2"] = x
        x = self.layer2(x)  # 1/8,1/8
        output["res3"] = x
        x = self.layer3(x)  # 1/16,1/16
        output["res4"] = x
        x = self.layer4(x)  # 1/32,1/32
        output["res5"] = x
        return output


@BACKBONE_REGISTRY.register()
class D2ModifiedResNet(ModifiedResNet, Backbone):
    def __init__(self, cfg, input_shape):
        depth = cfg.MODEL.RESNETS.DEPTH
        num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels = num_groups * width_per_group
        num_blocks_per_stage = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]
        strides = [2, 1, 2, 2, 2]
        multi_grid = cfg.MODEL.RESNETS.RES5_MULTI_GRID
        if cfg.MODEL.RESNETS.STEM_TYPE == "deeplab":
            strides = [1, 1, 2, 2, 2]
        super().__init__(
            num_blocks_per_stage,
            bottleneck_channels,
            strides=strides,
            multi_grid=multi_grid,
        )
        self._out_features = cfg.MODEL.RESNETS.OUT_FEATURES

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32
