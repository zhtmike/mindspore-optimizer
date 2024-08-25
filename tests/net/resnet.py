from typing import List, Optional, Type, Union

import mindspore.common.initializer as init
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class GlobalAvgPooling(nn.Cell):
    """
    GlobalAvgPooling, same as torch.nn.AdaptiveAvgPool2d when output shape is 1
    """

    def __init__(self, keep_dims: bool = False) -> None:
        super().__init__()
        self.keep_dims = keep_dims

    def construct(self, x):
        x = ops.mean(x, axis=(2, 3), keep_dims=self.keep_dims)
        return x


class BasicBlock(nn.Cell):
    """define the basic block of resnet"""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
        down_sample: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d
        assert groups == 1, "BasicBlock only supports groups=1"
        assert base_width == 64, "BasicBlock only supports base_width=64"

        self.conv1 = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            pad_mode="pad",
        )
        self.bn1 = norm(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, pad_mode="pad"
        )
        self.bn2 = norm(channels)
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    """
    Bottleneck here places the stride for downsampling at 3x3 convolution(self.conv2) as torchvision does,
    while original implementation places the stride at the first 1x1 convolution(self.conv1)
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
        down_sample: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        width = int(channels * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1)
        self.bn1 = norm(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            pad_mode="pad",
            group=groups,
        )
        self.bn2 = norm(width)
        self.conv3 = nn.Conv2d(
            width, channels * self.expansion, kernel_size=1, stride=1
        )
        self.bn3 = norm(channels * self.expansion)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    r"""ResNet model class, based on
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/abs/1512.03385>`_

    Args:
        block: block of resnet.
        layers: number of layers of each stage.
        num_classes: number of classification classes. Default: 1000.
        in_channels: number the channels of the input. Default: 3.
        groups: number of groups for group conv in blocks. Default: 1.
        base_width: base width of pre group hidden channel in blocks. Default: 64.
        norm: normalization layer in blocks. Default: None.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        groups: int = 1,
        base_width: int = 64,
        norm: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d

        self.norm: nn.Cell = norm  # add type hints to make pylint happy
        self.input_channels = 64
        self.groups = groups
        self.base_with = base_width

        self.conv1 = nn.Conv2d(
            in_channels,
            self.input_channels,
            kernel_size=7,
            stride=2,
            pad_mode="pad",
            padding=3,
        )
        self.bn1 = norm(self.input_channels)
        self.relu = nn.ReLU()
        self.feature_info = [dict(chs=self.input_channels, reduction=2, name="relu")]
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.feature_info.append(
            dict(chs=block.expansion * 64, reduction=4, name="layer1")
        )
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.feature_info.append(
            dict(chs=block.expansion * 128, reduction=8, name="layer2")
        )
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.feature_info.append(
            dict(chs=block.expansion * 256, reduction=16, name="layer3")
        )
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature_info.append(
            dict(chs=block.expansion * 512, reduction=32, name="layer4")
        )

        self.pool = GlobalAvgPooling()
        self.num_features = 512 * block.expansion
        self.classifier = nn.Dense(self.num_features, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(
                        init.HeNormal(mode="fan_out", nonlinearity="relu"),
                        cell.weight.shape,
                        cell.weight.dtype,
                    )
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer("zeros", cell.bias.shape, cell.bias.dtype)
                    )
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(
                    init.initializer("ones", cell.gamma.shape, cell.gamma.dtype)
                )
                cell.beta.set_data(
                    init.initializer("zeros", cell.beta.shape, cell.beta.dtype)
                )
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(
                        init.HeUniform(mode="fan_in", nonlinearity="sigmoid"),
                        cell.weight.shape,
                        cell.weight.dtype,
                    )
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer("zeros", cell.bias.shape, cell.bias.dtype)
                    )

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        channels: int,
        block_nums: int,
        stride: int = 1,
    ) -> nn.SequentialCell:
        """build model depending on cfgs"""
        down_sample = None

        if stride != 1 or self.input_channels != channels * block.expansion:
            down_sample = nn.SequentialCell(
                [
                    nn.Conv2d(
                        self.input_channels,
                        channels * block.expansion,
                        kernel_size=1,
                        stride=stride,
                    ),
                    self.norm(channels * block.expansion),
                ]
            )

        layers = []
        layers.append(
            block(
                self.input_channels,
                channels,
                stride=stride,
                down_sample=down_sample,
                groups=self.groups,
                base_width=self.base_with,
                norm=self.norm,
            )
        )
        self.input_channels = channels * block.expansion

        for _ in range(1, block_nums):
            layers.append(
                block(
                    self.input_channels,
                    channels,
                    groups=self.groups,
                    base_width=self.base_with,
                    norm=self.norm,
                )
            )

        return nn.SequentialCell(layers)

    def forward_features(self, x: Tensor) -> Tensor:
        """Network forward feature extraction."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def resnet50(num_classes: int = 1000, in_channels=3, **kwargs):
    model_args = dict(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_channels=in_channels,
        **kwargs
    )
    return ResNet(**model_args)
