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
from collections.abc import Iterator
from typing import Any, Optional, Union

import einops
import numpy as np
import scipy.signal
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_utils.distributed as dist_utils
from torch_utils import misc, persistence
from torch_utils.ops import bias_act, upfirdn2d

# =====================================================================================================================


def bias_act_wrapper(x, b=None, dim=1, **kwargs) -> torch.Tensor:
    int_max = 2**31 - 1

    if x.numel() < int_max:
        return bias_act.bias_act(x, b=b, dim=dim, **kwargs)

    split_dim = np.argmax(x.size())
    num_chunks = math.ceil(x.numel() / (int_max - 1))
    assert num_chunks <= x.size(split_dim), "Tensor is too large"

    if split_dim == dim and b is not None:
        b_chunks = b.chunk(num_chunks, dim=0)
    else:
        b_chunks = [b] * num_chunks

    chunks = []
    for x_chunk, b_chunk in zip(x.chunk(num_chunks, dim=split_dim), b_chunks):
        chunk = bias_act.bias_act(x_chunk, b=b_chunk, dim=dim, **kwargs)
        chunks.append(chunk)
    output = torch.cat(chunks, dim=split_dim)
    return output


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


def normalize_2nd_moment(tensor: torch.Tensor, dim: Union[int, tuple[int, ...]] = 1, eps: float = 1e-8) -> torch.Tensor:
    return tensor * tensor.square().mean(dim=dim, keepdim=True).add(eps).rsqrt()


# ======================================================================================================================


def temporal_modulated_conv3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    style: torch.Tensor,
    input_gain: Optional[torch.Tensor] = None,
    padding: tuple[int, int, int] = (0, 0, 0),
    demodulate: bool = True,
) -> torch.Tensor:

    assert input.dim() == 5
    batch_size, in_channels = input.size(0), input.size(1)
    misc.assert_shape(weight, (None, in_channels, None, None, None))
    misc.assert_shape(style, (batch_size, in_channels, None))

    # Pre-normalize inputs.
    if demodulate:
        weight = weight / weight.abs().amax(dim=(1, 2, 3, 4), keepdim=True)
        style = style / style.abs().amax(dim=(1, 2), keepdim=True)

    num_inputs = np.prod(weight.shape[1:])
    weight = weight / math.sqrt(num_inputs)

    # Compute demodulation.
    if demodulate:
        demodulation = torch.einsum("oizyx,nit->not", weight.square(), style.square())
        demodulation = demodulation.add(1e-8).rsqrt()
        demodulation = einops.rearrange(demodulation, "n co t -> n co t 1 1")

    # Apply input scaling.
    if input_gain is not None:
        assert input_gain.dim() == 0
        input = input * input_gain

    # Modulate activations before conv3d.
    style = einops.rearrange(style, "n ci t -> n ci t 1 1")
    input = input * style.type(input.dtype)
    output = F.conv3d(input, weight.type(input.dtype), padding=padding)

    # Demodulate activations after conv3d.
    if demodulate:
        output = output * demodulation.type(output.dtype)

    return output


# ======================================================================================================================


def center_crop(
    input: torch.Tensor, width: Optional[int] = None, height: Optional[int] = None, seq_length: Optional[int] = None
) -> torch.Tensor:

    input_dim = input.dim()
    assert input_dim in (3, 5)

    if width is not None:
        assert input_dim == 5
        x0 = (input.size(4) - width) // 2
        x1 = x0 + width
        input = input[:, :, :, :, x0:x1]

    if height is not None:
        assert input_dim == 5
        y0 = (input.size(3) - height) // 2
        y1 = y0 + height
        input = input[:, :, :, y0:y1]

    if seq_length is not None:
        t0 = (input.size(2) - seq_length) // 2
        t1 = t0 + seq_length
        input = input[:, :, t0:t1]

    return input


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


@persistence.persistent_class
class TemporalLinearUpsample(LinearResample):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_dim = input.dim()
        assert input_dim in (3, 5)

        if input_dim == 5:
            height = input.size(3)
            input = einops.rearrange(input, "n c t h w -> n c t (h w)")
        else:
            input = einops.rearrange(input, "n c t -> n c t 1")

        if self.padding > 0:
            input = F.pad(input, (0, 0, self.padding, self.padding), mode=self.padding_mode)

        filter = einops.rearrange(self.filter, "k -> k 1")
        output = upsample2d_wrapper(input, filter, up=(1, self.scale), padding=(0, -self.padding * self.scale))

        if input_dim == 5:
            output = einops.rearrange(output, "n c t (h w) -> n c t h w", h=height)
        else:
            output = einops.rearrange(output, "n c t 1 -> n c t")

        return output


@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class KaiserResample(nn.Module):
    scale: int = 2
    padding: int = 0
    padding_mode: str = "replicate"
    filter_size: int = 6
    cutoff: float = 1.0
    width: float = 6.0
    sampling_rate: float = 4.0

    def __post_init__(self):
        super().__init__()
        assert self.scale > 1 and isinstance(self.scale, int)
        num_taps = self.scale * self.filter_size
        fs = self.scale * self.sampling_rate
        filter = scipy.signal.firwin(numtaps=num_taps, cutoff=self.cutoff, width=self.width, fs=fs)
        filter = torch.tensor(filter, dtype=torch.float32)
        self.register_buffer("filter", filter)


@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class TemporalKaiserDownsample(KaiserResample):
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
class MagnitudeEMA(nn.Module):
    def __init__(self, dist_sync: bool = True):
        super().__init__()
        self.dist_sync = dist_sync
        self.register_buffer("magnitude_ema", torch.ones(()))

    def forward(self, input: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        # Retrieves and optionally updates reciprocal of moving average of input magnitude.
        if beta != 1:
            input_sq = input.detach().to(torch.float32).square()
            magnitude = input_sq.mean()
            if self.dist_sync and dist_utils.get_world_size() > 1:
                dist.all_reduce(magnitude, dist.ReduceOp.SUM)
                magnitude = magnitude / dist_utils.get_world_size()
            self.magnitude_ema.lerp_(magnitude, 1.0 - beta)

        gain = self.magnitude_ema.rsqrt()
        return gain


# ======================================================================================================================


@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class BlurredNoise(nn.Module):
    channels: int = 1024
    min_sampling_rate: float = 250
    max_sampling_rate: float = 10000
    blur_widths: int = 128
    cutoff: float = 2.0
    width: float = 12.0
    sampling_rate_base: float = 2.0
    normalize_per_filter: float = 1.0

    def __post_init__(self):
        assert self.channels % self.blur_widths == 0
        super().__init__()

        self.noise_channels = self.channels // self.blur_widths
        self.kernel_size = int(np.ceil(self.max_sampling_rate / 2))
        blur_filters = torch.zeros(self.blur_widths, self.kernel_size)

        if self.sampling_rate_base > 1:
            log_min_sampling_rate = math.log(self.min_sampling_rate, self.sampling_rate_base)
            log_max_sampling_rate = math.log(self.max_sampling_rate, self.sampling_rate_base)
            log_sampling_rates = np.linspace(log_min_sampling_rate, log_max_sampling_rate, self.blur_widths)
            sampling_rates = self.sampling_rate_base**log_sampling_rates
            sampling_rates = np.clip(sampling_rates, self.min_sampling_rate, self.max_sampling_rate)
        else:
            sampling_rates = np.linspace(self.min_sampling_rate, self.max_sampling_rate, self.blur_widths)

        for i, sampling_rate in enumerate(sampling_rates):
            num_taps = int(np.ceil(sampling_rate / 2))
            blur_filter = scipy.signal.firwin(numtaps=num_taps, cutoff=self.cutoff, width=self.width, fs=sampling_rate)
            blur_filter = torch.as_tensor(blur_filter, dtype=torch.float32)
            blur_filters[i, -num_taps:] = blur_filter

        if self.normalize_per_filter > 0:
            output_scale = (blur_filters**2).sum(dim=1).rsqrt()
            output_scale = einops.rearrange(output_scale, "c -> 1 c 1")
            self.register_buffer("output_scale", output_scale)

        blur_filters = einops.rearrange(blur_filters, "c k -> c 1 k")
        self.register_buffer("blur_filters", blur_filters)

    def forward(
        self,
        batch_size: int,
        seq_length: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:

        input_seq_length = seq_length + self.kernel_size + -1
        noise = torch.randn(
            batch_size, self.noise_channels, input_seq_length, device=self.blur_filters.device, generator=generator
        )
        features = self.blur(noise)
        return features

    def blur(self, noise: torch.Tensor) -> torch.Tensor:
        misc.assert_shape(noise, (None, self.noise_channels, None))
        noise = einops.repeat(noise, "n c t -> (n c) b t", b=self.blur_widths)
        features = F.conv1d(noise, self.blur_filters, groups=self.blur_widths)

        if self.normalize_per_filter > 0:
            features = features * (1 + self.normalize_per_filter * (self.output_scale - 1))

        features = einops.rearrange(features, "(n c) b t -> n (c b) t", c=self.noise_channels)
        return features


# =====================================================================================================================


@persistence.persistent_class
class FullyConnectedLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = "linear",
        lrate_mul: float = 1.0,
        weight_std_init: float = 1.0,
        bias_init: float = 0.0,
    ):
        assert activation in bias_act.activation_funcs
        super().__init__()

        self.activation = activation

        weight_init = torch.randn(out_features, in_features) * weight_std_init / lrate_mul
        self.weight = nn.Parameter(weight_init)
        self.weight_gain = lrate_mul / math.sqrt(in_features)

        if bias:
            bias = torch.full((out_features,), bias_init / lrate_mul, dtype=torch.float32)
            self.bias = nn.Parameter(bias)
            self.bias_gain = lrate_mul
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
class LatentMappingNetwork(nn.Module):
    temporal_emb_dim: int = 1024
    latent_w_dim: int = 1024
    num_layers: int = 2
    activation: str = "lrelu"
    lrate_mul: float = 0.01
    normalize_input: bool = True

    def __post_init__(self):
        super().__init__()
        self.layer_names = []

        for index in range(self.num_layers):
            input_dim = self.temporal_emb_dim if index == 0 else self.latent_w_dim
            layer = FullyConnectedLayer(
                input_dim, self.latent_w_dim, activation=self.activation, lrate_mul=self.lrate_mul
            )
            layer_name = f"layer_{index}"
            self.layer_names.append(layer_name)
            setattr(self, layer_name, layer)

    def forward(self, temporal_emb: torch.Tensor) -> torch.Tensor:
        misc.assert_shape(temporal_emb, (None, self.temporal_emb_dim, None))

        if self.normalize_input:
            temporal_emb = normalize_2nd_moment(temporal_emb)

        seq_length = temporal_emb.size(2)
        features = einops.rearrange(temporal_emb, "n c t -> (n t) c")

        for layer_name in self.layer_names:
            layer = getattr(self, layer_name)
            features = layer(features)

        latent_w_dim = einops.rearrange(features, "(n t) c -> n c t", t=seq_length)
        return latent_w_dim


# ======================================================================================================================


@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class Synthesis3dResBlock(nn.Module):
    latent_dim: int
    in_channels: int

    # Assumes output sizes same as input if None.
    out_channels: Optional[int] = None
    out_width: Optional[int] = None
    out_height: Optional[int] = None

    temporal_ksize: int = 1
    spatial_ksize: int = 1

    temporal_up: bool = False
    spatial_up: bool = False

    activation: str = "lrelu"
    activation_clamp: Optional[int] = 256.0
    magnitude_ema: bool = True
    demodulate: bool = True
    use_float16: bool = False

    def __post_init__(self):
        super().__init__()
        self.out_channels = self.out_channels or self.in_channels
        assert self.activation in bias_act.activation_funcs

        self.affine_0 = FullyConnectedLayer(self.latent_dim, self.in_channels, bias_init=1.0)
        self.affine_1 = FullyConnectedLayer(self.latent_dim, self.in_channels, bias_init=1.0)

        self.weight_0 = nn.Parameter(
            torch.randn(self.in_channels, self.in_channels, self.temporal_ksize, self.spatial_ksize, self.spatial_ksize)
        )

        self.weight_1 = nn.Parameter(
            torch.randn(
                self.out_channels, self.in_channels, self.temporal_ksize, self.spatial_ksize, self.spatial_ksize
            )
        )

        self.weight_skip = nn.Parameter(torch.randn(self.out_channels, self.in_channels, 1, 1, 1))
        self.weight_skip_gain = 1 / math.sqrt(self.in_channels)

        self.bias_0 = nn.Parameter(torch.zeros(self.in_channels))
        self.bias_1 = nn.Parameter(torch.zeros(self.out_channels))

        self.padding = (self.temporal_ksize // 2, self.spatial_ksize // 2, self.spatial_ksize // 2)

        if self.magnitude_ema:
            self.input_magnitude_ema_0 = MagnitudeEMA()
            self.input_magnitude_ema_1 = MagnitudeEMA()

        if self.temporal_up:
            self.temporal_upsample = TemporalLinearUpsample()

        if self.spatial_up:
            self.spatial_upsample = SpatialBilinearUpsample()

    def forward(
        self,
        input: torch.Tensor,
        latent: torch.Tensor,
        magnitude_ema_beta: float = 1.0,
        out_seq_length: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:

        misc.assert_shape(input, (None, self.in_channels, None, None, None))
        batch_size, in_seq_length = input.size(0), input.size(2)
        misc.assert_shape(latent, (batch_size, self.latent_dim, in_seq_length))

        latent = einops.rearrange(latent, "n c t -> (n t) c")
        style_0 = self.affine_0(latent)
        style_0 = einops.rearrange(style_0, "(n t) c -> n c t", t=in_seq_length)

        dtype = dtype if dtype is not None else (torch.float16 if self.use_float16 and input.is_cuda else torch.float32)
        input = input.type(dtype)

        if self.magnitude_ema:
            input_gain_0 = self.input_magnitude_ema_0(input, magnitude_ema_beta)
            input = input * input_gain_0

        hidden = temporal_modulated_conv3d(input, self.weight_0, style_0, padding=self.padding, demodulate=True)
        bias_0 = self.bias_0.type(hidden.dtype)
        hidden = bias_act_wrapper(hidden, bias_0, act=self.activation, clamp=self.activation_clamp)

        style_1 = self.affine_1(latent)
        style_1 = einops.rearrange(style_1, "(n t) c -> n c t", t=in_seq_length)
        input_gain_1 = self.input_magnitude_ema_1(hidden, magnitude_ema_beta) if self.magnitude_ema else None
        hidden = temporal_modulated_conv3d(hidden, self.weight_1, style_1, input_gain_1, self.padding, demodulate=True)

        weight_skip = (self.weight_skip * self.weight_skip_gain).type(input.dtype)
        input = F.conv3d(input, weight_skip)
        hidden = (input + hidden) * math.sqrt(0.5)

        if self.temporal_up:
            hidden = self.temporal_upsample(hidden)
        hidden = center_crop(hidden, seq_length=out_seq_length)
        if self.spatial_up:
            hidden = self.spatial_upsample(hidden)
        hidden = center_crop(hidden, width=self.out_width, height=self.out_height)

        bias_1 = self.bias_1.type(hidden.dtype)
        output = bias_act_wrapper(hidden, bias_1, act=self.activation, clamp=self.activation_clamp)

        misc.assert_shape(output, (None, self.out_channels, None, self.out_height, self.out_width))
        return output


# ======================================================================================================================


@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class ToRGB(nn.Module):
    latent_dim: int
    in_channels: int
    activation_clamp: Optional[int] = 256.0
    magnitude_ema: bool = True
    use_float16: bool = False

    def __post_init__(self):
        super().__init__()

        self.affine = FullyConnectedLayer(self.latent_dim, self.in_channels, bias_init=1.0)

        self.weight = nn.Parameter(torch.randn(3, self.in_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(3))

        if self.magnitude_ema:
            self.input_magnitude_ema = MagnitudeEMA()

    def forward(
        self,
        input: torch.Tensor,
        latent: torch.Tensor,
        magnitude_ema_beta: float = 1.0,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:

        misc.assert_shape(input, (None, self.in_channels, None, None, None))
        batch_size, in_seq_length = input.size(0), input.size(2)
        misc.assert_shape(latent, (batch_size, self.latent_dim, in_seq_length))

        latent = einops.rearrange(latent, "n c t -> (n t) c")
        style = self.affine(latent)
        style = einops.rearrange(style, "(n t) c -> n c t", t=in_seq_length)

        dtype = dtype if dtype is not None else (torch.float16 if self.use_float16 and input.is_cuda else torch.float32)
        input = input.type(dtype)
        input_gain = self.input_magnitude_ema(input, magnitude_ema_beta) if self.magnitude_ema else None
        output = temporal_modulated_conv3d(input, self.weight, style, input_gain, demodulate=False)

        bias = self.bias.type(output.dtype)
        output = bias_act_wrapper(output, bias, act="linear", clamp=self.activation_clamp)
        return output


# ======================================================================================================================


@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class VideoGenerator(nn.Module):
    out_height: int = 36
    out_width: int = 64
    temporal_emb_dim: int = 1024
    latent_w_dim: int = 1024
    temporal_ksize: int = 3
    spatial_ksize: int = 3
    temporal_padding: int = 8
    spatial_padding: int = 0
    output_scale: float = 0.25
    num_fp16_layers: int = 0

    embedding_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    mapping_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        super().__init__()

        long_edge = max(self.out_height, self.out_width)
        scales = tuple(max(1, long_edge // (2 ** (2 + i))) for i in range(5))

        # fmt: off
        heights = [math.ceil(self.out_height / scale) + 2 * self.spatial_padding for scale in scales]
        widths = [math.ceil(self.out_width / scale) + 2 * self.spatial_padding for scale in scales]
        t_kwargs = dict(spatial_ksize=self.spatial_ksize, temporal_ksize=self.temporal_ksize)
        s_kwargs = dict(spatial_ksize=self.spatial_ksize)
        self.temporal_layers = nn.ModuleList([
            Synthesis3dResBlock(self.latent_w_dim, in_channels=512, out_height=heights[0], out_width=widths[0], temporal_up=True, **t_kwargs),
            Synthesis3dResBlock(self.latent_w_dim, in_channels=512, out_height=heights[1], out_width=widths[1], temporal_up=True, spatial_up=True, **t_kwargs),
            Synthesis3dResBlock(self.latent_w_dim, in_channels=512, temporal_up=True, **t_kwargs),
            Synthesis3dResBlock(self.latent_w_dim, in_channels=512, out_channels=512, out_height=heights[2], out_width=widths[2], temporal_up=True, spatial_up=True, **t_kwargs),
            Synthesis3dResBlock(self.latent_w_dim, in_channels=512, out_channels=256, temporal_up=True, **t_kwargs),
            Synthesis3dResBlock(self.latent_w_dim, in_channels=256, **t_kwargs),
        ])
        self.spatial_layers = nn.ModuleList([
            Synthesis3dResBlock(self.latent_w_dim, in_channels=256, out_channels=128, out_height=heights[3], out_width=widths[3], spatial_up=True, **s_kwargs),
            Synthesis3dResBlock(self.latent_w_dim, in_channels=128, **s_kwargs),
            Synthesis3dResBlock(self.latent_w_dim, in_channels=128, out_channels=64, out_height=heights[4], out_width=widths[4], spatial_up=heights[4] != heights[3], **s_kwargs),
            Synthesis3dResBlock(self.latent_w_dim, in_channels=64, out_height=self.out_height, out_width=self.out_width, **s_kwargs),
        ])
        self.to_rgb = ToRGB(self.latent_w_dim, in_channels=self.spatial_layers[-1].out_channels)
        # fmt: on

        self.num_layers = len(self.temporal_layers) + len(self.spatial_layers) + 1

        layers = [self.to_rgb] + list(reversed(self.spatial_layers)) + list(reversed(self.temporal_layers))
        for layer in layers[: self.num_fp16_layers]:
            layer.use_float16 = True

        self.total_spatial_scale = 1
        self.total_temporal_scale = 1
        for layer in self.temporal_layers:
            if layer.spatial_up:
                self.total_spatial_scale *= 2
            if layer.temporal_up:
                self.total_temporal_scale *= 2
        for layer in self.spatial_layers:
            if layer.spatial_up:
                self.total_spatial_scale *= 2

        self.spatial_input = nn.Parameter(torch.randn(1, self.temporal_layers[0].in_channels, 1, heights[0], widths[0]))
        self.temporal_emb = BlurredNoise(self.temporal_emb_dim, **self.embedding_kwargs)
        self.latent_mapping = LatentMappingNetwork(self.temporal_emb_dim, self.latent_w_dim, **self.mapping_kwargs)
        self.temporal_downsample_latent = TemporalKaiserDownsample()
        self.w_to_temp_input = FullyConnectedLayer(self.latent_w_dim, self.temporal_layers[0].in_channels)

    def synthesize_video(
        self,
        temporal_input: torch.Tensor,
        latent_ws: list[torch.Tensor],
        seq_length: int,
        magnitude_ema_beta: float = 1.0,
        dtype: Optional[torch.dtype] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        in_seq_length, seq_lengths = self.compute_seq_lengths(seq_length)
        misc.assert_shape(temporal_input, (None, self.temporal_layers[0].in_channels, in_seq_length))

        temporal_input = einops.repeat(temporal_input, "n c t -> n c t 1 1")
        input = (temporal_input + self.spatial_input) * math.sqrt(0.5)
        features = input
        w_index = 0

        if return_features:
            features_list = []

        for layer, layer_seq_length in zip(self.temporal_layers, seq_lengths):
            features = layer(features, latent_ws[w_index], magnitude_ema_beta, layer_seq_length, dtype=dtype)
            if return_features:
                features_list.append(features)
            w_index += 1

        for layer in self.spatial_layers:
            features = layer(features, latent_ws[w_index], magnitude_ema_beta, dtype=dtype)
            if return_features:
                features_list.append(features)
            w_index += 1

        lr_video = self.to_rgb(features, latent_ws[w_index], magnitude_ema_beta, dtype=dtype)
        lr_video = lr_video.type(torch.float32) * self.output_scale

        if return_features:
            features_list.append(lr_video)
            return features_list

        return lr_video

    def forward(
        self,
        batch_size: int,
        seq_length: int,
        magnitude_ema_beta: float = 1.0,
        generator_emb: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        temporal_emb = self.sample_temporal_emb(batch_size, seq_length, generator_emb)
        latent_ws = self.compute_latent_ws(temporal_emb, seq_length)

        in_seq_length = self.compute_seq_lengths(seq_length)[0]

        temporal_input = einops.rearrange(
            self.w_to_temp_input(einops.rearrange(latent_ws.pop(0), "n c t -> (n t) c")),
            "(n t) c -> n c t",
            t=in_seq_length,
        )

        return self.synthesize_video(temporal_input, latent_ws, seq_length, magnitude_ema_beta, dtype)

    def sample_video_segments(
        self,
        batch_size: int,
        seq_length: int,
        segment_length: int = 8,
        generator_emb: Optional[torch.Generator] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Iterator[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]:
        # This method generates videos in chunks (of size segment_length), enabling generation of very long videos.

        temporal_emb = self.sample_temporal_emb(batch_size, seq_length, generator_emb)
        latent_ws = self.compute_latent_ws(temporal_emb, seq_length)

        in_seq_length, seq_lengths = self.compute_seq_lengths(seq_length)

        temporal_input = einops.rearrange(
            self.w_to_temp_input(einops.rearrange(latent_ws.pop(0), "n c t -> (n t) c")),
            "(n t) c -> n c t",
            t=in_seq_length,
        )

        temporal_input = einops.repeat(temporal_input, "n c t -> n c t 1 1")
        input = (temporal_input + self.spatial_input) * math.sqrt(0.5)
        features = input
        w_index = 0

        for layer, layer_seq_length in zip(self.temporal_layers, seq_lengths):
            features = layer(features, latent_ws[w_index], out_seq_length=layer_seq_length, dtype=dtype)
            w_index += 1

        for layer in self.spatial_layers:
            features = layer(features, latent_ws[w_index], dtype=dtype)
            w_index += 1

        lr_video = self.to_rgb(features, latent_ws[w_index], dtype=dtype)
        lr_video = lr_video.type(torch.float32) * self.output_scale

        for lr_video_segment in lr_video.split(segment_length, dim=2):
            yield lr_video_segment

    def compute_seq_lengths(self, seq_length: int) -> tuple[int, list[int]]:
        seq_lengths = [seq_length]
        temporal_scale = 1

        for layer in reversed(self.temporal_layers):
            if layer.temporal_up:
                temporal_scale *= 2
            layer_seq_length = math.ceil(seq_length / temporal_scale) + 2 * self.temporal_padding
            seq_lengths.append(layer_seq_length)

        input_seq_length = seq_lengths.pop()
        seq_lengths.reverse()
        return input_seq_length, seq_lengths

    def sample_temporal_input(
        self,
        batch_size: int,
        seq_length: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        input_seq_length = self.compute_seq_lengths(seq_length)[0]
        temporal_input = torch.randn(
            batch_size,
            self.temporal_layers[0].in_channels,
            input_seq_length,
            generator=generator,
            device=device,
        )
        return temporal_input

    def sample_temporal_emb(
        self, batch_size: int, seq_length: int, generator_emb: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        input_seq_length = self.compute_seq_lengths(seq_length)[0]
        temporal_emb_seq_length = input_seq_length * self.total_temporal_scale
        temporal_emb = self.temporal_emb(batch_size, temporal_emb_seq_length, generator_emb)
        return temporal_emb

    def compute_latent_ws(self, temporal_emb: torch.Tensor, seq_length: int) -> torch.Tensor:
        misc.assert_shape(temporal_emb, (None, self.temporal_emb_dim, None))

        latent_w = self.latent_mapping(temporal_emb)
        input_seq_length, seq_lengths = self.compute_seq_lengths(seq_length)

        # First add ws for spatial layers with no temporal connectivity.
        num_spatial_layers = len(self.spatial_layers) + 1
        latent_w_layer = center_crop(latent_w, seq_length=seq_lengths.pop())
        latent_ws = [latent_w_layer.clone() for _ in range(num_spatial_layers)]

        # Next prepend ws for low resolution temporal layers.
        seq_lengths.reverse()
        seq_lengths.append(input_seq_length)
        for layer, layer_seq_length in zip(reversed(self.temporal_layers), seq_lengths):
            if layer.temporal_up:
                latent_w = self.temporal_downsample_latent(latent_w)
            latent_w_layer = center_crop(latent_w, seq_length=layer_seq_length)
            latent_ws.insert(0, latent_w_layer)
        latent_ws.insert(0, latent_ws[0].clone())
        return latent_ws
