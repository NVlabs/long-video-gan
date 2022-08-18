# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import contextlib
import functools
import math
import os
import pickle
import re
import time
from collections.abc import Iterable, Iterator
from typing import Optional, Union

import einops
import imageio
import PIL.Image
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import dnnlib
import torch_utils.distributed as dist_utils
from torch_utils import distributed

# =====================================================================================================================


def get_next_run_dir(outdir: str, desc: Optional[str] = None) -> str:
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    name = f"{cur_run_id:05d}" if desc is None else f"{cur_run_id:05d}-{desc}"
    run_dir = os.path.join(outdir, name)
    assert not os.path.exists(run_dir)
    return run_dir


# =====================================================================================================================


def load_G(path: str):
    with dnnlib.util.open_url(path) as fp:
        G = pickle.load(fp)
    return G.requires_grad_(False).eval().cuda()


# =====================================================================================================================


def rank0_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if dist_utils.get_rank() == 0:
            return func(*args, **kwargs)

    return wrapper


@rank0_only
def print0(*args, **kwargs):
    print(*args, **kwargs)


@contextlib.contextmanager
def context_timer0(message: str):
    start_time = time.time()
    print0(message, end="... ")
    try:
        yield
    finally:
        duration = time.time() - start_time
        print0(f"{duration:.2f}s")


# =====================================================================================================================


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.contiguous()
    tensor_list = [torch.zeros_like(tensor) for _ in range(dist_utils.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    tensor = torch.cat(tensor_list)
    return tensor


def all_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor, dist.ReduceOp.SUM)
    tensor = tensor / dist_utils.get_world_size()
    return tensor


def sharded_all_mean(tensor: torch.Tensor, shard_size: int = 2**23) -> torch.Tensor:
    assert tensor.dim() == 1
    shards = tensor.tensor_split(math.ceil(tensor.numel() / shard_size))
    for shard in shards:
        torch.distributed.all_reduce(shard)
    tensor = torch.cat(shards) / dist_utils.get_world_size()
    return tensor


# =====================================================================================================================


def sync_grads(network: nn.Module, gain: Optional[torch.Tensor] = None):
    params = [param for param in network.parameters() if param.grad is not None]
    flat_grads = torch.cat([param.grad.flatten() for param in params])
    flat_grads = sharded_all_mean(flat_grads)
    flat_grads = flat_grads if gain is None else flat_grads * gain
    torch.nan_to_num(flat_grads, nan=0, posinf=1e5, neginf=-1e5, out=flat_grads)
    grads = flat_grads.split([param.numel() for param in params])
    for param, grad in zip(params, grads):
        param.grad = grad.reshape(param.size())


# =====================================================================================================================


def random_seed(max_seed: int = 2**31 - 1) -> int:
    seed = torch.randint(max_seed + 1, (), device="cuda")
    if distributed.get_world_size() > 1:
        dist.broadcast(seed, src=0)
    return seed.item()


# =====================================================================================================================


def multiple_nearest_sqrt(number: int) -> int:
    for i in range(int(math.sqrt(number)), 0, -1):
        if number % i == 0:
            return i


def write_video_grid(
    segments: Union[torch.Tensor, Iterable[torch.Tensor]],
    path: Optional[os.PathLike] = None,
    fps: int = 30,
    max_samples: Optional[int] = None,
    num_rows: Optional[int] = None,
    to_uint8: bool = True,
    gather: bool = False,
):
    if isinstance(segments, torch.Tensor):
        segments = [segments]

    if dist_utils.get_rank() == 0:
        assert path is not None
        video_writer = imageio.get_writer(path, mode="I", fps=fps, codec="libx264", bitrate="16M")

    for segment in segments:
        segment = (segment * 127.5 + 128).clamp(0, 255).to(torch.uint8) if to_uint8 else segment
        segment = all_gather(segment) if gather else segment

        if dist_utils.get_rank() == 0:
            segment = segment[:max_samples] if max_samples else segment
            num_rows = num_rows or multiple_nearest_sqrt(segment.size(0))

            for frame in segment.unbind(dim=2):
                frame_grid = einops.rearrange(frame, "(nw nh) c h w -> (nh h) (nw w) c", nh=num_rows)

                # Ensures each edge is a multiple of 16, resizing if needed.
                scale_y = 16 // math.gcd(frame_grid.size(0), 16)
                scale_x = 16 // math.gcd(frame_grid.size(1), 16)
                scale = scale_y * scale_x // math.gcd(scale_y, scale_x)
                if scale > 1:
                    frame_grid = einops.rearrange(frame_grid, "h w c -> 1 c h w")
                    frame_grid = F.interpolate(frame_grid, scale_factor=scale, mode="nearest")
                    frame_grid = einops.rearrange(frame_grid, "1 c h w -> h w c")

                frame_grid = frame_grid.cpu().numpy()
                video_writer.append_data(frame_grid)

    if dist_utils.get_rank() == 0:
        video_writer.close()


# =====================================================================================================================


def save_image_grid(
    image: torch.Tensor,
    path: Optional[os.PathLike] = None,
    max_samples: Optional[int] = None,
    num_rows: Optional[int] = None,
    to_uint8: bool = True,
    gather: bool = False,
):
    if dist_utils.get_rank() == 0:
        assert path is not None

    image = (image * 127.5 + 128).clamp(0, 255).to(torch.uint8) if to_uint8 else image
    image = all_gather(image) if gather else image

    if dist_utils.get_rank() == 0:
        image = image[:max_samples] if max_samples else image
        num_rows = num_rows or multiple_nearest_sqrt(image.size(0))
        image_grid = einops.rearrange(image, "(nw nh) c h w -> (nh h) (nw w) c", nh=num_rows)
        PIL.Image.fromarray(image_grid.cpu().numpy()).save(path)


# =====================================================================================================================


def get_infinite_data_iter(dataset: Dataset, seed: Optional[int] = None, **loader_kwargs) -> Iterator:
    seed = random_seed() if seed is None else seed
    generator = torch.Generator().manual_seed(seed)
    sampler = DistributedSampler(dataset, seed=seed) if distributed.get_world_size() > 1 else None
    loader = DataLoader(dataset, shuffle=(sampler is None), sampler=sampler, generator=generator, **loader_kwargs)

    epoch = 0
    while True:
        if distributed.get_world_size() > 1:
            sampler.set_epoch(epoch)
        for sample in loader:
            yield sample
        epoch += 1


# =====================================================================================================================
