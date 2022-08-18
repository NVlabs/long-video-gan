# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import dnnlib
import einops
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_utils.distributed as dist_utils
import utils
from torch.optim import Adam
from torch_utils import misc, training_stats

from .ada_augment import AugmentPipe

# =====================================================================================================================


@dataclass
class SuperResVideoGAN:
    seq_length: int
    temporal_context: int
    lr_height: int
    lr_width: int
    hr_height: int
    hr_width: int

    channels: int = 3

    G_lrate: float = 0.003
    G_beta2: float = 0.99
    G_warmup_steps: int = 0
    G_ema_beta: float = 0.99985
    G_ema_warmup_steps: int = 25000
    G_magnitude_ema_beta: float = 0.999
    G_grad_accum: int = 1
    G_kwargs: dict[str, Any] = field(default_factory=dict)

    D_lrate: float = 0.002
    D_beta2: float = 0.99
    D_warmup_steps: int = 0
    D_grad_accum: int = 1
    D_kwargs: dict[str, Any] = field(default_factory=dict)

    r1_gamma: Optional[float] = 1.0
    lr_cond_prob: float = 0.1

    augment_p_init: float = 0.0
    augment_p_max: float = 0.5
    augment_p_update_rate: float = 0.000125
    augment_real_sign_target: Optional[float] = 0.6
    augment_kwargs: dict[str, Any] = field(default_factory=dict)

    in_augment_p: float = 0.5
    in_augment_strength: float = 8.0

    # ==================================================================================================================

    def __post_init__(self):
        self.context_seq_length = self.seq_length + 2 * self.temporal_context

        # Constructs generator and exponential moving average of generator.
        G_kwargs = dict(
            hr_height=self.hr_height,
            hr_width=self.hr_width,
            lr_height=self.lr_height,
            lr_width=self.lr_width,
            temporal_context=self.temporal_context,
        )
        self.G = dnnlib.util.construct_class_by_name(**G_kwargs, **self.G_kwargs)
        self.G_ema = dnnlib.util.construct_class_by_name(**G_kwargs, **self.G_kwargs)
        self.G.cuda().requires_grad_(False).train()
        self.G_ema.cuda().requires_grad_(False).eval()

        self.D = dnnlib.util.construct_class_by_name(
            channels=self.channels,
            seq_length=self.seq_length,
            lr_height=self.lr_height,
            lr_width=self.lr_width,
            hr_height=self.hr_height,
            hr_width=self.hr_width,
            **self.D_kwargs,
        )
        self.D.cuda().requires_grad_(False).train()

        # Synchronizes all weights at start.
        for network in (self.G, self.D):
            for tensor in misc.params_and_buffers(network):
                dist.broadcast(tensor, src=0)

        for tensor, tensor_ema in zip(misc.params_and_buffers(self.G), misc.params_and_buffers(self.G_ema)):
            tensor_ema.copy_(tensor)

        # Constructs opts.
        self.G_opt = Adam(self.G.parameters(), lr=self.G_lrate, betas=(0, self.G_beta2))
        self.D_opt = Adam(self.D.parameters(), lr=self.D_lrate, betas=(0, self.D_beta2))

        self.augment = None
        self.real_sign_collector = None
        self.in_augment = None

        if self.augment_p_init > 0 or self.augment_real_sign_target is not None:
            self.augment = AugmentPipe(**self.augment_kwargs)
            self.augment.cuda().requires_grad_(False).train()
            self.augment.p.fill_(self.augment_p_init)

        if self.augment_real_sign_target is not None:
            self.real_sign_collector = training_stats.Collector(regex=f"loss/D_sign_real")

        if self.in_augment_strength > 0 and self.in_augment_p > 0:
            self.in_augment = AugmentPipe(
                scale=1,
                scale_std=0.01 * self.in_augment_strength,
                rotate=1,
                rotate_max=0.002 * self.in_augment_strength,
                aniso=1,
                aniso_std=0.01 * self.in_augment_strength,
                xfrac=1,
                xfrac_std=0.002 * self.in_augment_strength,
                noise=1,
                noise_std=0.01 * self.in_augment_strength,
            )
            self.in_augment.cuda().requires_grad_(False).train()
            self.in_augment.p.fill_(self.in_augment_p)

    # ==================================================================================================================

    def update_lrates(self, step: int):
        G_lrate = self.G_lrate * min((step + 1) / (self.G_warmup_steps + 1), 1.0)
        D_lrate = self.D_lrate * min((step + 1) / (self.D_warmup_steps + 1), 1.0)
        self.G_opt.param_groups[0]["lr"] = G_lrate
        self.D_opt.param_groups[0]["lr"] = D_lrate
        training_stats.report0(f"progress/G_lrate", G_lrate)
        training_stats.report0(f"progress/D_lrate", D_lrate)

    # ==================================================================================================================

    def update_G(self, lr_video: torch.Tensor):
        misc.assert_shape(lr_video, (None, self.channels, self.context_seq_length, self.lr_height, self.lr_width))
        assert lr_video.size(0) % self.G_grad_accum == 0

        lr_video = lr_video if self.in_augment is None else self.in_augment(lr_video)

        self.G.requires_grad_(True)

        for lr_video_chunk in lr_video.chunk(self.G_grad_accum):
            hr_video_chunk = self.G(lr_video_chunk)
            lr_video_chunk = self.crop_to_seq_length(lr_video_chunk)
            logits = self.run_D(lr_video_chunk, hr_video_chunk)
            loss = F.softplus(-logits)
            loss.mean().backward()

            training_stats.report(f"loss/G_score", logits)
            training_stats.report(f"loss/G_sign", logits.sign())
            training_stats.report(f"loss/G_loss", loss)

        self.G.requires_grad_(False)

        gain = 1 / self.G_grad_accum
        utils.sync_grads(self.G, gain=gain)
        self.G_opt.step()
        self.G_opt.zero_grad(set_to_none=True)

    # ==================================================================================================================

    def update_D(self, fake_lr_video: torch.Tensor, real_lr_video: torch.Tensor, real_hr_video: torch.Tensor):
        misc.assert_shape(fake_lr_video, (None, self.channels, self.context_seq_length, self.lr_height, self.lr_width))
        misc.assert_shape(real_lr_video, (None, self.channels, self.context_seq_length, self.lr_height, self.lr_width))
        misc.assert_shape(real_hr_video, (None, self.channels, self.seq_length, self.hr_height, self.hr_width))
        assert fake_lr_video.size(0) == real_lr_video.size(0) == real_hr_video.size(0)
        assert fake_lr_video.size(0) % self.D_grad_accum == 0

        if self.in_augment is not None:
            fake_lr_video = self.in_augment(fake_lr_video)
            real_lr_video = self.in_augment(real_lr_video)

        fake_hr_video = self.G(fake_lr_video, magnitude_ema_beta=self.G_magnitude_ema_beta)
        fake_lr_video = self.crop_to_seq_length(fake_lr_video)
        real_lr_video = self.crop_to_seq_length(real_lr_video)

        self.D.requires_grad_(True)

        for fake_lr_video_chunk, fake_hr_video_chunk, real_lr_video_chunk, real_hr_video_chunk in zip(
            fake_lr_video.chunk(self.D_grad_accum),
            fake_hr_video.chunk(self.D_grad_accum),
            real_lr_video.chunk(self.D_grad_accum),
            real_hr_video.chunk(self.D_grad_accum),
        ):
            fake_logits = self.run_D(fake_lr_video_chunk, fake_hr_video_chunk)
            fake_loss = F.softplus(fake_logits)
            fake_loss.mean().backward()

            real_logits = self.run_D(real_lr_video_chunk, real_hr_video_chunk)
            real_loss = F.softplus(-real_logits)
            real_loss.mean().backward()

            training_stats.report(f"loss/D_score_fake", fake_logits)
            training_stats.report(f"loss/D_score_real", real_logits)
            training_stats.report(f"loss/D_sign_fake", fake_logits.sign())
            training_stats.report(f"loss/D_sign_real", real_logits.sign())
            training_stats.report(f"loss/D_loss", fake_loss + real_loss)

        self.D.requires_grad_(False)

        gain = 1 / self.D_grad_accum
        utils.sync_grads(self.D, gain=gain)
        self.D_opt.step()
        self.D_opt.zero_grad(set_to_none=True)

    # ==================================================================================================================

    def update_r1(self, lr_video: torch.Tensor, hr_video: torch.Tensor, gain: float = 1.0):
        misc.assert_shape(lr_video, (None, self.channels, self.seq_length, self.lr_height, self.lr_width))
        misc.assert_shape(hr_video, (None, self.channels, self.seq_length, self.hr_height, self.hr_width))
        assert lr_video.size(0) == hr_video.size(0)
        assert lr_video.size(0) % self.D_grad_accum == 0

        if self.in_augment is not None:
            lr_video = self.in_augment(lr_video)

        self.D.requires_grad_(True)

        for lr_video_chunk, hr_video_chunk in zip(lr_video.chunk(self.D_grad_accum), hr_video.chunk(self.D_grad_accum)):
            hr_video_chunk = hr_video_chunk.detach().requires_grad_(True)
            logits = self.run_D(lr_video_chunk, hr_video_chunk)
            r1_grads = torch.autograd.grad(outputs=[logits.sum()], inputs=[hr_video_chunk], create_graph=True)
            r1_penalty = r1_grads[0].square().sum(dim=(1, 2, 3, 4))

            r1_loss = r1_penalty * (self.r1_gamma / 2)
            r1_loss.mean().backward()

            training_stats.report(f"loss/r1_penalty", r1_penalty)
            training_stats.report(f"loss/r1_loss", r1_loss)

        self.D.requires_grad_(False)

        gain /= self.D_grad_accum
        utils.sync_grads(self.D, gain=gain)
        self.D_opt.step()
        self.D_opt.zero_grad(set_to_none=True)

    # ==================================================================================================================

    @torch.no_grad()
    def update_ada(self, gain: float = 1.0):
        if self.real_sign_collector is not None:
            self.real_sign_collector.update()

            update_sign = self.real_sign_collector[f"loss/D_sign_real"] - self.augment_real_sign_target
            update = math.copysign(self.augment_p_update_rate, update_sign) * gain
            self.augment.p.add_(update).clamp_(0, self.augment_p_max)

        if self.augment is not None:
            training_stats.report0(f"progress/augment_p", self.augment.p)

    # ==================================================================================================================

    @torch.no_grad()
    def update_G_ema(self, step: int):
        ema_reciprocal_halflife = math.log(self.G_ema_beta, 0.5) * (self.G_ema_warmup_steps + 1) / (step + 1)
        ema_beta = min(0.5**ema_reciprocal_halflife, self.G_ema_beta)
        training_stats.report0("progress/G_ema_beta", ema_beta)
        for tensor_ema, tensor in zip(misc.params_and_buffers(self.G_ema), misc.params_and_buffers(self.G)):
            tensor_ema.lerp_(tensor, 1.0 - ema_beta)

    # ==================================================================================================================

    def crop_to_seq_length(self, video: torch.Tensor) -> torch.Tensor:
        misc.assert_shape(video, (None, self.channels, None, None, None))
        t0 = (video.size(2) - self.seq_length) // 2
        t1 = t0 + self.seq_length
        video = video[:, :, t0:t1]
        return video

    # ==================================================================================================================

    def ckpt(self):
        train_ckpt = dict(
            G_opt=self.G_opt.state_dict(),
            D_opt=self.D_opt.state_dict(),
            real_sign_collector=copy.deepcopy(self.real_sign_collector),
        )
        for name, module in (
            ("G_ema", self.G_ema),
            ("G", self.G),
            ("D", self.D),
            ("augment", self.augment),
            ("in_augment", self.in_augment),
        ):
            if module is None:
                train_ckpt[name] = None
            else:
                module = copy.deepcopy(module).eval().requires_grad_(False)
                if dist_utils.get_world_size() > 1:
                    misc.check_ddp_consistency(module)
                train_ckpt[name] = module

        G_ema_ckpt = train_ckpt.pop("G_ema")
        return G_ema_ckpt, train_ckpt

    # ==================================================================================================================

    def run_D(self, lr_video: torch.Tensor, hr_video: torch.Tensor) -> torch.Tensor:
        misc.assert_shape(lr_video, (None, self.channels, self.seq_length, self.lr_height, self.lr_width))
        misc.assert_shape(hr_video, (None, self.channels, self.seq_length, self.hr_height, self.hr_width))

        lr_video = self.D.upsample(lr_video)
        hr_video = torch.cat((lr_video, hr_video), dim=2)

        if self.augment is not None:
            hr_video = self.augment(hr_video)

        lr_video, hr_video = hr_video.chunk(2, dim=2)

        if self.lr_cond_prob < 1:
            mask = torch.rand(lr_video.size(0), 1, 1, 1, 1, device=lr_video.device) < self.lr_cond_prob
            lr_video = lr_video * mask.type(lr_video.dtype)

        logits = self.D(lr_video, hr_video)
        return logits
