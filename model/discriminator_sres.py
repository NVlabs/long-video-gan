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
from typing import Optional

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import misc, persistence
from torch_utils.ops import bias_act, conv2d_resample, upfirdn2d

# =====================================================================================================================


def upsample2d_wrapper(x, f, up=2, **kwargs) -> torch.Tensor:
    upx, upy = upfirdn2d._parse_scaling(up)
    output_size = x.numel() * upx * upy
    int_max = 2**31 - 1

    if output_size < int_max:
        return upfirdn2d.upsample2d(x, f, up=up, **kwargs)

    split_dim = np.argmax(x.shape[:2])
    num_chunks = math.ceil(output_size / (int_max - 1))
    assert num_chunks <= x.size(split_dim), "Tensor is too large"

    chunks = []
    for chunk in x.chunk(num_chunks, dim=split_dim):
        chunk = upfirdn2d.upsample2d(chunk, f, up=up, **kwargs)
        chunks.append(chunk)
    output = torch.cat(chunks, dim=split_dim)
    return output


# =====================================================================================================================


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
class SpatialBilinearUpsample(LinearResample):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 5
        channels = input.size(1)
        input = einops.rearrange(input, "n c t h w -> n (c t) h w")

        if self.padding > 0:
            input = F.pad(input, (self.padding, self.padding, self.padding, self.padding), mode=self.padding_mode)

        output = upsample2d_wrapper(input, self.filter, up=self.scale, padding=-self.padding * self.scale)
        output = einops.rearrange(output, "n (c t) h w -> n c t h w", c=channels)
        return output


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
        dropout_p: float = 0.0,
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

        self.dropout_p = dropout_p
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.dropout_p > 0:
            input = self.dropout(input)

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
class Conv2dLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        kernel_size,  # Width and height of the convolution kernel.
        bias=True,  # Apply additive bias before the activation function?
        activation="linear",  # Activation function: 'relu', 'lrelu', etc.
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
        resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
        channels_last=False,  # Expect the input to have memory_format=channels_last?
        trainable=True,  # Update the weights of this layer during training?
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / math.sqrt(in_channels * (kernel_size**2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer("weight", weight)
            if bias is not None:
                self.register_buffer("bias", bias)
            else:
                self.bias = None

        self.dropout_p = dropout_p
        if self.dropout_p > 0:
            self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x, gain=1):
        if self.dropout_p > 0:
            x = self.dropout(x)

        w = self.weight * self.weight_gain

        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = self.up == 1  # slightly faster
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=w.to(x.dtype),
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
            flip_weight=flip_weight,
        )

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)

        return x


# =====================================================================================================================


@persistence.persistent_class
class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels, 0 = first block.
        tmp_channels,  # Number of intermediate channels.
        out_channels,  # Number of output channels.
        resolution,  # Resolution of this block.
        img_channels,  # Number of input color channels.
        first_layer_idx,  # Index of the first layer.
        architecture="resnet2",  # Architecture: 'orig', 'skip', 'resnet'.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16=False,  # Use FP16 for this block?
        fp16_channels_last=False,  # Use channels-last memory format with FP16?
        freeze_layers=0,  # Freeze-D: Number of layers to freeze.
    ):
        assert in_channels in [0, tmp_channels]
        assert architecture in ["orig", "skip", "resnet", "resnet2"]
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = use_fp16 and fp16_channels_last
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.negate_half = False

        self.num_layers = 0

        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = layer_idx >= freeze_layers
                self.num_layers += 1
                yield trainable

        trainable_iter = trainable_gen()

        if in_channels == 0 or architecture == "skip":
            self.fromrgb = Conv2dLayer(
                img_channels,
                tmp_channels,
                kernel_size=1,
                activation=activation,
                trainable=next(trainable_iter),
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
            )

        self.conv0 = Conv2dLayer(
            tmp_channels,
            tmp_channels,
            kernel_size=3,
            activation=activation,
            trainable=next(trainable_iter),
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        self.conv1 = Conv2dLayer(
            tmp_channels,
            out_channels,
            kernel_size=3,
            activation=activation,
            down=2,
            trainable=next(trainable_iter),
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
        )

        if architecture == "resnet":
            self.skip = Conv2dLayer(
                tmp_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                down=2,
                trainable=next(trainable_iter),
                resample_filter=resample_filter,
                channels_last=self.channels_last,
            )

    def forward(self, x, img, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            # misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0 or self.architecture == "skip":
            # misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img)
            x = x + y if x is not None else y
            img = upfirdn2d.downsample2d(img, self.resample_filter) if self.architecture == "skip" else None

        # Main layers.
        if self.architecture == "resnet":
            y = self.skip(x)
            x = self.conv0(x)
            x = self.conv1(x)

            if self.negate_half:
                x = x - 2 * self.negate_mask.to(x.dtype) * x

            x = (x + y) * np.sqrt(0.5)

        elif self.architecture == "resnet2":

            y = upfirdn2d.downsample2d(x, self.resample_filter)
            y = torch.cat((y, y), dim=1)[:, : self.out_channels]

            if self.negate_half:
                y = y - 2 * self.negate_mask.to(y.dtype) * y

            x = self.conv0(x)
            x = self.conv1(x)

            if self.negate_half:
                x = x - 2 * self.negate_mask2.to(x.dtype) * x

            x = (x + y) * np.sqrt(0.5)

        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x, img


@persistence.persistent_class
class MinibatchStdLayer(nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings():  # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(
            G, -1, F, c, H, W
        )  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


@persistence.persistent_class
class DiscriminatorEpilogue(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        height,  # Resolution of this block.
        width,
        mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels=1,  # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation="lrelu",  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
        output_dim=1,
        pool_mode: str = "fully_connected",
    ):
        assert pool_mode in ("fully_connected", "average")
        super().__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.pool_mode = pool_mode

        self.mbstd = (
            MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels)
            if mbstd_num_channels > 0
            else None
        )
        self.conv = Conv2dLayer(
            in_channels + mbstd_num_channels,
            in_channels,
            kernel_size=3,
            activation=activation,
            conv_clamp=conv_clamp,
        )

        self.fc = FullyConnectedLayer(in_channels * height * width, in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, output_dim)

    def forward(self, x: torch.Tensor, conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:

        # misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])  # [NCHW] # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)

        x = self.conv(x)

        if self.pool_mode == "fully_connected":
            x = self.fc(x.flatten(1))
        elif self.pool_mode == "average":
            x = x.mean(dim=(2, 3))

        x = self.out(x)

        # Conditioning.
        if conditioning is not None:
            assert conditioning.dim() == 2
            assert conditioning.size(0) == x.size(0)
            assert conditioning.size(1) == x.size(1)

            x = (x * conditioning).sum(dim=1, keepdim=True) * (1 / np.sqrt(conditioning.size(1)))

        assert x.dtype == dtype
        return x


@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class VideoDiscriminator(nn.Module):
    channels: int = 3
    seq_length: int = 8
    lr_height: int = 32
    lr_width: int = 32
    hr_height: int = 256
    hr_width: int = 256
    channels_base: int = 16384
    channels_max: int = 512
    num_fp16_res: int = 4
    conv_clamp: Optional[int] = 256
    minibatch_std_group_size: int = 4
    minibatch_std_num_channels: int = 0
    architecture: str = "resnet"
    pool_mode: str = "fully_connected"

    def __post_init__(self):
        super().__init__()

        resolution = max(self.hr_height, self.hr_width)
        self.resolution_log2 = int(np.log2(resolution))

        self.block_resolutions = [2**i for i in range(self.resolution_log2, 2, -1)]
        channels_dict = {res: min(self.channels_base // res, self.channels_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.resolution_log2 + 1 - self.num_fp16_res), 8)

        img_channels = 2 * self.channels * self.seq_length
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = res >= fp16_resolution

            block = DiscriminatorBlock(
                in_channels,
                tmp_channels,
                out_channels,
                resolution=res,
                img_channels=img_channels,
                first_layer_idx=cur_layer_idx,
                use_fp16=use_fp16,
                conv_clamp=self.conv_clamp,
                architecture=self.architecture,
            )

            setattr(self, f"b{res}", block)
            cur_layer_idx += block.num_layers

        self.b4 = DiscriminatorEpilogue(
            channels_dict[4],
            height=4,
            width=4,
            mbstd_group_size=self.minibatch_std_group_size,
            mbstd_num_channels=self.minibatch_std_num_channels,
            output_dim=1,
            conv_clamp=self.conv_clamp,
            pool_mode=self.pool_mode,
        )

        self.upsample = SpatialBilinearUpsample(resolution // max(self.lr_height, self.lr_width))

    def forward(self, lr_video: torch.Tensor, hr_video: torch.Tensor) -> torch.Tensor:

        if lr_video.size(3) == self.lr_height and lr_video.size(4) == self.lr_width:
            lr_video = self.upsample(lr_video)
        else:
            assert lr_video.size(3) == self.hr_height and lr_video.size(4) == self.hr_width

        videos = torch.cat((lr_video, hr_video), dim=1)
        p = (videos.size(4) - videos.size(3)) // 2
        videos = F.pad(videos, (0, 0, p, p))
        videos = einops.rearrange(videos, "n c t h w -> n (c t) h w")

        features = None
        for res in self.block_resolutions:
            block = getattr(self, f"b{res}")
            features, videos = block(features, videos)

        logits = self.b4(features)
        return logits
