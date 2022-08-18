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
from typing import Optional, Union

import einops
import numpy as np
import scipy.optimize
import scipy.signal
import torch
from torch_utils import distributed, misc, persistence
from torch_utils.ops import bias_act, conv2d_gradfix, filtered_lrelu, upfirdn2d

# fmt: off
#----------------------------------------------------------------------------

@misc.profiled_function
def modulated_conv2d(
    x,                  # Input tensor: [batch_size, in_channels, in_height, in_width]
    w,                  # Weight tensor: [out_channels, in_channels, kernel_height, kernel_width]
    s,                  # Style tensor: [batch_size, in_channels]
    demodulate  = True, # Apply weight demodulation?
    padding     = 0,    # Padding: int or [padH, padW]
    input_gain  = None, # Optional scale factors for the input channels: [], [in_channels], or [batch_size, in_channels]
):
    with misc.suppress_tracer_warnings(): # this value will be treated as a constant
        batch_size = int(x.shape[0])
    out_channels, in_channels, kh, kw = w.shape
    misc.assert_shape(w, [out_channels, in_channels, kh, kw]) # [OIkk]
    misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    misc.assert_shape(s, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs.
    if demodulate:
        w = w * w.square().mean([1,2,3], keepdim=True).rsqrt()
        s = s * s.square().mean().rsqrt()

    # Modulate weights.
    w = w.unsqueeze(0) # [NOIkk]
    w = w * s.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Demodulate weights.
    if demodulate:
        dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt() # [NO]
        w = w * dcoefs.unsqueeze(2).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Apply input scaling.
    if input_gain is not None:
        input_gain = input_gain.expand(batch_size, in_channels) # [NI]
        w = w * input_gain.unsqueeze(1).unsqueeze(3).unsqueeze(4) # [NOIkk]

    # Execute as one fused op using grouped convolution.
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_gradfix.conv2d(input=x, weight=w.to(x.dtype), padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        bias            = True,     # Apply additive bias before the activation function?
        lr_multiplier   = 1,        # Learning rate multiplier.
        weight_init     = 1,        # Initial standard deviation of the weight tensor.
        bias_init       = 0,        # Initial value of the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no labels.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output.
        num_layers      = 2,        # Number of mapping layers.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.998,    # Decay for tracking the moving average of W during training.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        # Construct layers.
        self.embed = FullyConnectedLayer(self.c_dim, self.w_dim) if self.c_dim > 0 else None
        features = [self.z_dim + (self.w_dim if self.c_dim > 0 else 0)] + [self.w_dim] * self.num_layers
        for idx, in_features, out_features in zip(range(num_layers), features[:-1], features[1:]):
            layer = FullyConnectedLayer(in_features, out_features, activation='lrelu', lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)
        self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        misc.assert_shape(z, [None, self.z_dim])
        if truncation_cutoff is None:
            truncation_cutoff = self.num_ws

        # Embed, normalize, and concatenate inputs.
        x = z.to(torch.float32)
        x = x * (x.square().mean(1, keepdim=True) + 1e-8).rsqrt()
        if self.c_dim > 0:
            misc.assert_shape(c, [None, self.c_dim])
            y = self.embed(c.to(torch.float32))
            y = y * (y.square().mean(1, keepdim=True) + 1e-8).rsqrt()
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Execute layers.
        for idx in range(self.num_layers):
            x = getattr(self, f'fc{idx}')(x)

        # Update moving average of W.
        if update_emas:
            x_mean = x.detach().mean(dim=0)
            if distributed.get_world_size() > 1:
                torch.distributed.all_reduce(x_mean)
                x_mean = x_mean / distributed.get_world_size()
            self.w_avg.copy_(x_mean.lerp(self.w_avg, self.w_avg_beta))

        # Broadcast and apply truncation.
        x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
        if truncation_psi != 1:
            x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisInput(torch.nn.Module):
    def __init__(self,
        w_dim,          # Intermediate latent (W) dimensionality.
        channels,       # Number of output channels.
        size,           # Output spatial size: int or [width, height].
        sampling_rate,  # Output sampling rate.
        bandwidth,      # Output bandwidth.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.channels = channels
        self.size = np.broadcast_to(np.asarray(size), [2])
        self.sampling_rate = sampling_rate
        self.bandwidth = bandwidth

        # Draw random frequencies from uniform 2D disc.
        freqs = torch.randn([self.channels, 2])
        radii = freqs.square().sum(dim=1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= bandwidth
        phases = torch.rand([self.channels]) - 0.5
        
        theta = torch.eye(2, 3)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grids = torch.nn.functional.affine_grid(theta.unsqueeze(0), [1, 1, self.size[1], self.size[0]], align_corners=False)

        # Compute Fourier features.
        features = torch.einsum("cd,nhwd->nchw", freqs, grids)
        features = features + einops.rearrange(phases, "c -> 1 c 1 1")
        features = torch.sin(features * (np.pi * 2))

        # Setup parameters and buffers.
        self.weight = torch.nn.Parameter(torch.randn([self.channels, self.channels]))
        self.register_buffer("features", features)
        
    def forward(self, batch_size: int):
        # Apply trainable mapping.
        weight = self.weight / np.sqrt(self.channels)
        features = torch.einsum("nchw,kc->nkhw", self.features, weight)
        features = einops.repeat(features, "1 c h w -> n c h w", n=batch_size)
        return features

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, channels={self.channels:d}, size={list(self.size)},',
            f'sampling_rate={self.sampling_rate:g}, bandwidth={self.bandwidth:g}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisLayer(torch.nn.Module):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        is_torgb,                       # Is this the final ToRGB layer?
        is_critically_sampled,          # Does this layer use critical sampling?
        use_fp16,                       # Does this layer use FP16?

        # Input & output specifications.
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
        in_cutoff,                      # Input cutoff frequency (f_c).
        out_cutoff,                     # Output cutoff frequency (f_c).
        in_half_width,                  # Input transition band half-width (f_h).
        out_half_width,                 # Output Transition band half-width (f_h).

        # Hyperparameters.
        conv_kernel         = 3,        # Convolution kernel size. Ignored for final the ToRGB layer.
        filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
        use_radial_filters  = False,    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
        conv_clamp          = 256,      # Clamp the output to [-X, +X], None = disable clamping.
        magnitude_ema_beta  = 0.999,    # Decay rate for the moving average of input magnitudes.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.is_torgb = is_torgb
        self.is_critically_sampled = is_critically_sampled
        self.use_fp16 = use_fp16
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) * (1 if is_torgb else lrelu_upsampling)
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        self.conv_kernel = 1 if is_torgb else conv_kernel
        self.conv_clamp = conv_clamp
        self.magnitude_ema_beta = magnitude_ema_beta

        # Setup parameters and buffers.
        self.affine = FullyConnectedLayer(self.w_dim, self.in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([self.out_channels, self.in_channels, self.conv_kernel, self.conv_kernel]))
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        self.register_buffer('magnitude_ema', torch.ones([]))

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        assert self.in_sampling_rate * self.up_factor == self.tmp_sampling_rate
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1 and not self.is_torgb else 1
        self.register_buffer('up_filter', self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        assert self.out_sampling_rate * self.down_factor == self.tmp_sampling_rate
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 and not self.is_torgb else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer('down_filter', self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial))

        # Compute padding.
        pad_total = (self.out_size - 1) * self.down_factor + 1 # Desired output size before downsampling.
        pad_total -= (self.in_size + self.conv_kernel - 1) * self.up_factor # Input size after upsampling.
        pad_total += self.up_taps + self.down_taps - 2 # Size reduction caused by the filters.
        pad_lo = (pad_total + self.up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]

    def forward(self, x, w, force_fp32=False, update_emas=False):
        misc.assert_shape(x, [None, self.in_channels, int(self.in_size[1]), int(self.in_size[0])])
        misc.assert_shape(w, [x.shape[0], self.w_dim])

        # Track input magnitude.
        if update_emas:
            with torch.autograd.profiler.record_function('update_magnitude_ema'):
                magnitude_cur = x.detach().to(torch.float32).square().mean()
                if distributed.get_world_size() > 1:
                    torch.distributed.all_reduce(magnitude_cur)
                    magnitude_cur = magnitude_cur / distributed.get_world_size()
                self.magnitude_ema.copy_(magnitude_cur.lerp(self.magnitude_ema, self.magnitude_ema_beta))
        input_gain = self.magnitude_ema.rsqrt()

        # Execute affine layer.
        styles = self.affine(w)
        if self.is_torgb:
            weight_gain = 1 / np.sqrt(self.in_channels * (self.conv_kernel ** 2))
            styles = styles * weight_gain

        # Execute modulated conv2d.
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        x = modulated_conv2d(x=x.to(dtype), w=self.weight, s=styles,
            padding=self.conv_kernel-1, demodulate=(not self.is_torgb), input_gain=input_gain)

        # Execute bias, filtered leaky ReLU, and clamping.
        gain = 1 if self.is_torgb else np.sqrt(2)
        slope = 1 if self.is_torgb else 0.2
        x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
            up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=self.conv_clamp)

        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
        assert x.dtype == dtype
        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        assert numtaps >= 1

        # Identity filter.
        if numtaps == 1:
            return None

        # Separable Kaiser low-pass filter.
        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)

        # Radially symmetric jinc-based filter.
        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return torch.as_tensor(f, dtype=torch.float32)

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
            f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
            f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
            f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
            f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
            f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                          # Intermediate latent (W) dimensionality.
        img_width,                      # Output image width.
        img_height,                     # Output image height.
        img_channels,                   # Number of color channels.
        cond_channels,                  # Number of conditioning channels per layer.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_layers          = 14,       # Total number of layers, excluding Fourier features and ToRGB.
        num_critical        = 2,        # Number of critically sampled layers at the end.
        first_cutoff        = 2,        # Cutoff frequency of the first layer (f_{c,0}).
        first_stopband      = 2**2.1,   # Minimum stopband of the first layer (f_{t,0}).
        last_stopband_rel   = 2**0.3,   # Minimum stopband of the last layer, expressed relative to the cutoff.
        margin_size         = 10,       # Number of additional pixels outside the image.
        fourfeats           = False,   
        output_scale        = 0.25,     # Scale factor for the output image.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        **layer_kwargs,                 # Arguments for SynthesisLayer.
    ):
        super().__init__()
        self.w_dim = w_dim
        self.num_ws = num_layers + 1
        self.img_width = img_width
        self.img_height = img_height
        self.img_resolution = max(img_width, img_height)
        self.img_channels = img_channels
        self.cond_channels = cond_channels
        self.num_layers = num_layers
        self.num_critical = num_critical
        self.margin_size = margin_size
        self.output_scale = output_scale
        self.num_fp16_res = num_fp16_res
        self.fourfeats = fourfeats

        # Geometric progression of layer cutoffs and min. stopbands.
        last_cutoff = self.img_resolution / 2 # f_{c,N}
        last_stopband = last_cutoff * last_stopband_rel # f_{t,N}
        exponents = np.minimum(np.arange(self.num_layers + 1) / (self.num_layers - self.num_critical), 1)
        cutoffs = first_cutoff * (last_cutoff / first_cutoff) ** exponents # f_c[i]
        stopbands = first_stopband * (last_stopband / first_stopband) ** exponents # f_t[i]

        # Compute remaining layer parameters.
        sampling_rates = np.exp2(np.ceil(np.log2(np.minimum(stopbands * 2, self.img_resolution)))) # s[i]
        half_widths = np.maximum(stopbands, sampling_rates / 2) - cutoffs # f_h[i]
        # sizes = sampling_rates + self.margin_size * 2
        # sizes[-2:] = self.img_resolution
        sizes_x = np.ceil(sampling_rates * min(1, self.img_width / self.img_height)) + self.margin_size * 2
        sizes_y = np.ceil(sampling_rates * min(1, self.img_height / self.img_width)) + self.margin_size * 2
        sizes_x[-2:] = self.img_width
        sizes_y[-2:] = self.img_height
        channels = np.rint(np.minimum((channel_base / 2) / cutoffs, channel_max))
        channels[-1] = self.img_channels

        # Construct layers.
        if self.fourfeats:
            self.input = SynthesisInput(
                w_dim=self.w_dim, channels=int(channels[0]), size=(int(sizes_x[0]), int(sizes_y[0])),
                sampling_rate=sampling_rates[0], bandwidth=cutoffs[0])
        self.layer_names = []
        for idx in range(self.num_layers + 1):
            prev = max(idx - 1, 0)
            is_torgb = (idx == self.num_layers)
            is_critically_sampled = (idx >= self.num_layers - self.num_critical)
            use_fp16 = (sampling_rates[idx] * (2 ** self.num_fp16_res) > self.img_resolution)

            in_channels = cond_channels
            if idx > 0 or self.fourfeats:
                in_channels += int(channels[prev])

            layer = SynthesisLayer(
                w_dim=self.w_dim, is_torgb=is_torgb, is_critically_sampled=is_critically_sampled, use_fp16=use_fp16,
                in_channels=in_channels, out_channels= int(channels[idx]),
                in_size=(int(sizes_x[prev]), int(sizes_y[prev])), out_size=(int(sizes_x[idx]), int(sizes_y[idx])),
                in_sampling_rate=int(sampling_rates[prev]), out_sampling_rate=int(sampling_rates[idx]),
                in_cutoff=cutoffs[prev], out_cutoff=cutoffs[idx],
                in_half_width=half_widths[prev], out_half_width=half_widths[idx],
                **layer_kwargs)
            name = f'L{idx}_{layer.out_size[0]}_{layer.out_size[1]}_{layer.out_channels}'
            setattr(self, name, layer)
            self.layer_names.append(name)

    def forward(self, ws, conds, **layer_kwargs):
        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        ws = ws.to(torch.float32).unbind(dim=1)
    
        x = self.input(ws[0].size(0)) if self.fourfeats else None
        for name, w, cond in zip(self.layer_names, ws, conds):
            x = cond if x is None else torch.cat((x, cond), dim=1)
            x = getattr(self, name)(x, w, **layer_kwargs)
        if self.output_scale != 1:
            x = x * self.output_scale

        misc.assert_shape(x, [None, self.img_channels, self.img_height, self.img_width])
        x = x.to(torch.float32)
        return x

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_layers={self.num_layers:d}, num_critical={self.num_critical:d},',
            f'margin_size={self.margin_size:d}, num_fp16_res={self.num_fp16_res:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class KaiserResample(torch.nn.Module):
    scale: int
    filter_size: int = 6
    cutoff: float = 1.0
    width: float = 6.0
    sampling_rate: float = 4.0
    pad: bool = True

    def __post_init__(self):
        super().__init__()
        assert self.scale > 1 and isinstance(self.scale, int)
        num_taps = self.scale * self.filter_size
        fs = self.scale * self.sampling_rate
        filter = scipy.signal.firwin(numtaps=num_taps, cutoff=self.cutoff, width=self.width, fs=fs)
        filter = torch.tensor(filter, dtype=torch.float32)
        self.register_buffer("filter", filter)

@persistence.persistent_class
class KaiserDownsample(KaiserResample):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 4

        p = int(self.pad) * self.scale
        if self.pad:
            input = torch.nn.functional.pad(input, (p, p, p, p), mode="replicate")

        output = upfirdn2d.downsample2d(input, self.filter, down=self.scale, padding=-p)
        return output

@persistence.persistent_class
class KaiserUpsample(KaiserResample):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 4
        
        p = int(self.pad)
        if self.pad:
            input = torch.nn.functional.pad(input, (p, p, p, p), mode="replicate")

        output = upfirdn2d.upsample2d(input, self.filter, up=self.scale, padding=-p*self.scale)
        return output

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_width,                  # Output image width.
        img_height,                 # Output image height.
        img_channels,               # Number of color channels.
        cond_width,                 # Input conditioning image width.
        cond_height,                # Input conditioning image height.
        cond_context,               # Input conditioning temporal context.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        margin_size         = 10,   # Number of additional pixels outside the image.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_width = img_width
        self.img_height = img_height
        self.img_channels = img_channels
        self.cond_width = cond_width
        self.cond_height = cond_height
        self.cond_context = cond_context
        self.cond_channels = self.img_channels * (2 * self.cond_context + 1)
        self.margin_size = margin_size

        self.synthesis = SynthesisNetwork(
            w_dim=w_dim, img_width=img_width, img_height=img_height, img_channels=img_channels,
            cond_channels=self.cond_channels, margin_size=margin_size, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=0, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

        self.resamples = torch.nn.ModuleList()
        for name in self.synthesis.layer_names:
            cond_scale = getattr(self.synthesis, name).in_sampling_rate / max(self.cond_width, self.cond_height)
            if cond_scale < 1:
                resample = KaiserDownsample(scale=math.ceil(1 / cond_scale))
            elif cond_scale > 1:
                resample = KaiserUpsample(scale=math.ceil(cond_scale))
            else:
                resample = torch.nn.Identity()
            self.resamples.append(resample)

    def forward(self, z, cond, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        misc.assert_shape(cond, [z.size(0), self.img_channels, None, self.cond_height, self.cond_width])
        out_seq_length = cond.size(2) - 2 * self.cond_context
        assert out_seq_length > 0
        conds = self.prep_cond(cond)
        z = einops.repeat(z, "n c -> (n t) c", t=out_seq_length)
        ws = self.mapping(z, c=None, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, conds, update_emas=update_emas, **synthesis_kwargs)
        vid = einops.rearrange(img, "(n t) c h w -> n c t h w", t=out_seq_length)
        return vid
    
    def prep_cond(self, cond):
        misc.assert_shape(cond, [None, self.img_channels, None, self.cond_height, self.cond_width])
        px0 = (max(self.cond_width, self.cond_height) - cond.size(4)) // 2 + self.margin_size
        px1 = (max(self.cond_width, self.cond_height) - cond.size(4) + 1) // 2 + self.margin_size
        py0 = (max(self.cond_width, self.cond_height) - cond.size(3)) // 2 + self.margin_size
        py1 = (max(self.cond_width, self.cond_height) - cond.size(3) + 1) // 2 + self.margin_size
        cond = torch.nn.functional.pad(cond, (px0, px1, py0, py1, 0, 0), mode="replicate")

        cond = cond.unfold(dimension=2, size=(1 + 2 * self.cond_context), step=1)
        cond = einops.rearrange(cond, "n c t h w s -> (n t) (c s) h w")

        conds = []
        for name, resample in zip(self.synthesis.layer_names, self.resamples):
            layer_cond = resample(cond)
            in_width, in_height = getattr(self.synthesis, name).in_size
            
            x0 = max(0, (layer_cond.size(3) - in_width) // 2)
            y0 = max(0, (layer_cond.size(2) - in_height) // 2)
            x1 = x0 + in_width
            y1 = y0 + in_height
            layer_cond = layer_cond[:, :, y0:y1, x0:x1]
            
            px0 = (in_width - layer_cond.size(3)) // 2
            px1 = (in_width - layer_cond.size(3) + 1) // 2
            py0 = (in_height - layer_cond.size(2)) // 2
            py1 = (in_height - layer_cond.size(2) + 1) // 2
            layer_cond = torch.nn.functional.pad(layer_cond, (px0, px1, py0, py1), mode="replicate")
            conds.append(layer_cond)

        return conds

# ----------------------------------------------------------------------------
# fmt: on


@persistence.persistent_class
@dataclasses.dataclass(eq=False, repr=False)
class VideoGenerator(torch.nn.Module):
    hr_height: int = 256
    hr_width: int = 256
    lr_height: int = 32
    lr_width: int = 32
    temporal_context: int = 4
    latent_z_dim: int = 512
    latent_w_dim: int = 512
    margin_size: int = 10
    fourfeats: bool = False
    num_fp16_res: int = 4

    def __post_init__(self):
        super().__init__()
        self.SG3 = Generator(
            z_dim=self.latent_z_dim,
            w_dim=self.latent_w_dim,
            img_width=self.hr_width,
            img_height=self.hr_height,
            img_channels=3,
            cond_width=self.lr_width,
            cond_height=self.lr_height,
            cond_context=self.temporal_context,
            margin_size=self.margin_size,
            fourfeats=self.fourfeats,
            num_fp16_res=self.num_fp16_res,
        )

    def sample_latent_z(self, batch_size: int, generator_z: Optional[torch.Generator] = None) -> torch.Tensor:
        device = next(self.parameters()).device
        latent_z = torch.randn(batch_size, self.latent_z_dim, generator=generator_z, device=device)
        return latent_z

    def forward(
        self, lr_video: torch.Tensor, generator_z: Optional[torch.Generator] = None, magnitude_ema_beta: float = 1.0
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        batch_size = lr_video.size(0)
        out_seq_length = lr_video.size(2) - 2 * self.temporal_context
        assert out_seq_length > 0
        latent_z = self.sample_latent_z(batch_size, generator_z)
        update_emas = magnitude_ema_beta < 1
        return self.SG3(latent_z, lr_video, update_emas=update_emas)

    def sample_video_segments(
        self, lr_video: torch.Tensor, segment_length: int = 8, generator_z: Optional[torch.Generator] = None
    ) -> Iterator[torch.Tensor]:

        # This method generates videos in chunks (of size segment_length), enabling generation of very long videos.

        batch_size = lr_video.size(0)
        out_seq_length = lr_video.size(2) - 2 * self.temporal_context
        assert out_seq_length > 0
        assert (lr_video.size(2) - 2 * self.temporal_context) % segment_length == 0

        latent_z = self.sample_latent_z(batch_size, generator_z)

        lr_video_segments = lr_video.unfold(
            dimension=2, size=segment_length + 2 * self.temporal_context, step=segment_length
        )
        lr_video_segments = einops.rearrange(lr_video_segments, "n c t h w s -> n c s h w t")
        for lr_video_segment in lr_video_segments.unbind(dim=-1):
            hr_video_segment = self.SG3(latent_z, lr_video_segment)
            yield hr_video_segment
