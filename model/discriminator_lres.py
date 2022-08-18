# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import misc, persistence
from torch_utils.ops import bias_act, upfirdn2d

# =====================================================================================================================


@persistence.persistent_class
class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = "linear",
        lr_multiplier: float = 1.0,
        weight_std_init: float = 1.0,
        bias_init: float = 0.0,
    ):
        assert activation in bias_act.activation_funcs
        super().__init__()

        self.activation = activation

        weight_init = torch.randn(out_features, in_features) * weight_std_init / lr_multiplier
        self.weight = nn.Parameter(weight_init)
        self.weight_gain = lr_multiplier / math.sqrt(in_features)

        if bias:
            bias = torch.full((out_features,), bias_init / lr_multiplier, dtype=torch.float32)
            self.bias = nn.Parameter(bias)
            self.bias_gain = lr_multiplier
        else:
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.weight_gain
        weight = weight.transpose(0, 1)

        if self.bias is None or self.bias_gain == 1:
            bias = self.bias
        else:
            bias = self.bias * self.bias_gain

        if self.activation == "linear" and bias is not None:
            output = torch.addmm(bias.unsqueeze(0), input, weight)
        else:
            output = input.matmul(weight)
            output = bias_act.bias_act(output, bias, act=self.activation)

        return output


# =====================================================================================================================


@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class Conv1dLayer(nn.Module):
    in_channels: int
    out_channels: Optional[int] = None
    kernel_size: int = 1
    bias: bool = True
    activation: str = "linear"
    lr_multiplier: float = 1.0
    weight_std_init: float = 1.0
    bias_init: float = 0.0
    downsample: bool = False

    def __post_init__(self):
        self.out_channels = self.out_channels or self.in_channels
        assert self.activation in bias_act.activation_funcs
        super().__init__()

        self.padding = self.kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(self.out_channels, self.in_channels, self.kernel_size)
            * self.weight_std_init
            / self.lr_multiplier
        )
        self.weight_gain = self.lr_multiplier / math.sqrt(self.in_channels * self.kernel_size)

        if self.bias:
            bias = torch.full((self.out_channels,), self.bias_init / self.lr_multiplier, dtype=torch.float32)
            self._bias = nn.Parameter(bias)

        if self.downsample:
            self._downsample = TemporalLinearDownsample(scale=2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.weight_gain
        weight = weight.to(input.dtype)

        if self.bias:
            if self.lr_multiplier != 1:
                bias = self._bias * self.lr_multiplier
            else:
                bias = self._bias
            bias = bias.to(input.dtype)
        else:
            bias = None

        output = F.conv1d(input, weight, bias, padding=self.padding)

        if self.downsample:
            output = self._downsample(output)

        output = bias_act.bias_act(output, act=self.activation)
        return output


# =====================================================================================================================


@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class Conv3dLayer(nn.Module):
    in_channels: int
    out_channels: int
    spatial_ksize: int
    temporal_ksize: int
    bias: bool = True
    spatial_down: bool = False
    temporal_down: bool = False
    activation: str = "linear"
    conv_clamp: Optional[int] = None

    def __post_init__(self):
        assert self.activation in bias_act.activation_funcs
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(
                self.out_channels,
                self.in_channels,
                self.temporal_ksize,
                self.spatial_ksize,
                self.spatial_ksize,
            )
        )

        self.weight_gain = 1 / math.sqrt(self.weight.numel() / self.out_channels)
        self.padding = (self.temporal_ksize // 2, self.spatial_ksize // 2, self.spatial_ksize // 2)

        if self.bias:
            self._bias = nn.Parameter(torch.zeros(self.out_channels))

        if self.spatial_down or self.temporal_down:
            self.downsample = Downsample3d(self.spatial_down, self.temporal_down)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.weight_gain
        weight = weight.type(input.dtype)
        output = F.conv3d(input, weight, padding=self.padding)

        if self.spatial_down or self.temporal_down:
            output = self.downsample(output)

        bias = self._bias.type(input.dtype) if self.bias else None
        output = bias_act.bias_act(output, bias, act=self.activation, clamp=self.conv_clamp)
        return output


# =====================================================================================================================


@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class Downsample3d(nn.Module):
    spatial_down: bool = True
    temporal_down: bool = True
    downsample_filter: tuple[float, ...] = (1.0, 3.0, 3.0, 1.0)

    def __post_init__(self):
        super().__init__()
        downsample_filter = torch.as_tensor(self.downsample_filter, dtype=torch.float32) / sum(self.downsample_filter)
        self.register_buffer("_downsample_filter", downsample_filter)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        assert features.dim() == 5

        if self.spatial_down:
            channels = features.size(1)
            features = einops.rearrange(features, "n c t h w -> n (c t) h w")
            features = upfirdn2d.downsample2d(features, self._downsample_filter, down=2)
            features = einops.rearrange(features, "n (c t) h w -> n c t h w", c=channels)

        if self.temporal_down:
            height = features.size(3)
            downsample_filter = einops.rearrange(self._downsample_filter, "k -> k 1")
            features = einops.rearrange(features, "n c t h w -> n c t (h w)")
            features = upfirdn2d.downsample2d(features, downsample_filter, down=[1, 2])
            features = einops.rearrange(features, "n c t (h w) -> n c t h w", h=height)

        return features


@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class LinearResample(nn.Module):
    scale: int = 2
    padding: int = 0
    padding_mode: str = "replicate"

    def __post_init__(self):
        super().__init__()
        assert self.scale > 1 and isinstance(self.scale, int)
        half_filter = torch.linspace(0.5 / self.scale, 1 - 0.5 / self.scale, self.scale)
        filter = torch.cat((half_filter, half_filter.flip(0)))
        filter /= filter.sum()
        self.register_buffer("filter", filter)


@persistence.persistent_class
class TemporalLinearDownsample(LinearResample):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dim = input.dim()
        assert input_dim in (3, 5)

        if input_dim == 5:
            height = input.size(3)
            input = einops.rearrange(input, "n c t h w -> n c t (h w)")
        else:
            input = einops.rearrange(input, "n c t -> n c t 1")

        padding = self.padding * self.scale
        if self.padding > 0:
            input = F.pad(input, (0, 0, padding, padding), mode=self.padding_mode)

        filter = einops.rearrange(self.filter, "k -> k 1")
        output = upfirdn2d.downsample2d(input, filter, down=(1, self.scale), padding=(0, -padding))

        if input_dim == 5:
            output = einops.rearrange(output, "n c t (h w) -> n c t h w", h=height)
        else:
            output = einops.rearrange(output, "n c t 1 -> n c t")

        return output


# =====================================================================================================================


@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class DiscriminatorBlock(nn.Module):
    in_channels: int
    out_channels: int
    vid_channels: int = 0
    spatial_ksize: int = 3
    temporal_ksize: int = 5
    spatial_ksize_1: Optional[int] = None
    temporal_ksize_1: Optional[int] = None
    spatial_down: bool = True
    temporal_down: bool = True
    conv_clamp: Optional[int] = 256
    use_fp16: bool = False

    def __post_init__(self):
        super().__init__()

        if self.vid_channels > 0:
            self.conv_vid = Conv3dLayer(
                in_channels=self.vid_channels,
                out_channels=self.in_channels,
                spatial_ksize=1,
                temporal_ksize=1,
                activation="lrelu",
                conv_clamp=self.conv_clamp,
            )

        self.conv_0 = Conv3dLayer(
            self.in_channels,
            self.in_channels,
            spatial_ksize=self.spatial_ksize,
            temporal_ksize=self.temporal_ksize,
            activation="lrelu",
            conv_clamp=self.conv_clamp,
        )

        self.conv_1 = Conv3dLayer(
            self.in_channels,
            self.out_channels,
            spatial_ksize=self.spatial_ksize_1 or self.spatial_ksize,
            temporal_ksize=self.temporal_ksize_1 or self.temporal_ksize,
            spatial_down=self.spatial_down,
            temporal_down=self.temporal_down,
            activation="lrelu",
            conv_clamp=self.conv_clamp,
        )

        self.conv_skip = Conv3dLayer(
            self.in_channels,
            self.out_channels,
            spatial_ksize=1,
            temporal_ksize=1,
            bias=False,
            spatial_down=self.spatial_down,
            temporal_down=self.temporal_down,
            conv_clamp=self.conv_clamp,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 5
        dtype = torch.float16 if self.use_fp16 else torch.float32
        input = input.type(dtype)

        if self.vid_channels > 0:
            input = self.conv_vid(input)

        hidden = self.conv_0(input)
        skip = self.conv_skip(input)
        hidden = self.conv_1(hidden)
        output = (hidden + skip) * math.sqrt(0.5)
        return output


# =====================================================================================================================


@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class DiscriminatorEpilogue(nn.Module):
    in_res: int = 4
    in_seq_length: int = 16
    in_channels: int = 512
    channels: int = 1024
    temporal_ksize: int = 3
    num_conv1d_layers: int = 4
    num_linear_layers: int = 2
    conv_clamp: Optional[int] = 256
    num_downsamples: int = 0

    def __post_init__(self, **_kwargs):
        super().__init__()
        assert self.num_downsamples <= self.num_conv1d_layers and self.in_seq_length % (2**self.num_downsamples) == 0

        self.conv1d_layer_names = []
        for index in range(self.num_conv1d_layers):
            layer_name = f"conv1d_{index}"
            self.conv1d_layer_names.append(layer_name)

            if index == 0:
                in_channels = (self.in_res**2) * self.in_channels
                ksize = 1
            else:
                in_channels = self.channels
                ksize = self.temporal_ksize

            layer = Conv1dLayer(
                in_channels,
                self.channels,
                kernel_size=ksize,
                activation="lrelu",
                downsample=index < self.num_downsamples,
            )
            setattr(self, layer_name, layer)

        self.linear_layer_names = []
        for index in range(self.num_linear_layers):
            layer_name = f"linear_{index}"
            self.linear_layer_names.append(layer_name)

            if index == 0:
                in_channels = self.in_seq_length * self.channels // (2**self.num_downsamples)
            else:
                in_channels = self.channels

            if index == self.num_linear_layers - 1:
                out_channels = 1
                activation = "linear"
            else:
                out_channels = self.channels
                activation = "lrelu"

            layer = FullyConnectedLayer(in_channels, out_channels, activation=activation)
            setattr(self, layer_name, layer)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        misc.assert_shape(input, (None, self.in_channels, self.in_seq_length, self.in_res, self.in_res))

        features = input.type(torch.float32)

        features = einops.rearrange(features, "n c t h w -> n (c h w) t")
        for layer_name in self.conv1d_layer_names:
            layer = getattr(self, layer_name)
            features = layer(features)

        features = einops.rearrange(features, "n c t -> n (c t)")
        for layer_name in self.linear_layer_names:
            layer = getattr(self, layer_name)
            features = layer(features)

        return features


# =====================================================================================================================


@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class VideoDiscriminator(nn.Module):
    seq_length: int
    max_edge: int
    channels: int = 3

    channels_base: int = 2048
    channels_max: int = 512
    spatial_ksize: int = 3
    temporal_ksize: int = 5
    spatial_ksize_1: Optional[int] = None
    temporal_ksize_1: Optional[int] = None
    conv_clamp: Optional[int] = 256
    num_fp16_res: int = 0

    epilogue_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        super().__init__()

        # fmt: off
        kwargs = dict(spatial_ksize=self.spatial_ksize, temporal_ksize=self.temporal_ksize, spatial_ksize_1=self.spatial_ksize_1, temporal_ksize_1=self.temporal_ksize_1, conv_clamp=self.conv_clamp)
        self.blocks = nn.ModuleList([
            DiscriminatorBlock(32, 64, self.channels, spatial_ksize=self.spatial_ksize, temporal_ksize=1, temporal_down=False, spatial_down=self.max_edge > 32, use_fp16=self.num_fp16_res > 0, conv_clamp=self.conv_clamp),
            DiscriminatorBlock(64, 128, use_fp16=self.num_fp16_res > 1, temporal_down=self.seq_length >= 4, **kwargs),
            DiscriminatorBlock(128, 256, use_fp16=self.num_fp16_res > 2, temporal_down=self.seq_length >= 8,  **kwargs),
            DiscriminatorBlock(256, 512, use_fp16=self.num_fp16_res > 3, temporal_down=self.seq_length >= 16, **kwargs),
        ])
        # fmt: on

        self.spatail_scale = 1
        self.temporal_scale = 1
        for block in self.blocks:
            if block.spatial_down:
                self.spatail_scale *= 2
            if block.temporal_down:
                self.temporal_scale *= 2

        self.epilogue = DiscriminatorEpilogue(
            self.max_edge // self.spatail_scale,
            self.seq_length // self.temporal_scale,
            self.blocks[-1].out_channels,
            **self.epilogue_kwargs,
        )

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        misc.assert_shape(videos, (None, self.channels, self.seq_length, None, None))
        assert videos.size(3) == self.max_edge or videos.size(4) == self.max_edge

        px = (self.max_edge - videos.size(4)) // 2
        py = (self.max_edge - videos.size(3)) // 2
        features = F.pad(videos, (px, px, py, py))

        for block in self.blocks:
            features = block(features)

        logits = self.epilogue(features)
        return logits
