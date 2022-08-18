# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from zipfile import ZIP_STORED, ZipFile

import joblib
import tqdm
from filelock import FileLock
from PIL import Image

# =====================================================================================================================


def time_str_to_sec(time_str: str) -> int:
    parts = [int(part) for part in reversed(time_str.split(":"))]
    multipliers = [1, 60, 3600]

    time_int = 0
    for part, multiplier in zip(parts, multipliers):
        time_int += part * multiplier

    return time_int


# =====================================================================================================================


def center_crop_and_resize(frame: Image, height: int, width: int) -> Image:
    # Measures by what factor height and width are larger/smaller than desired.
    height_scale = frame.height / height
    width_scale = frame.width / width

    # Center crops whichever dimension has a greater scale factor.
    if height_scale > width_scale:
        crop_height = height * width_scale
        y0 = (frame.height - crop_height) // 2
        y1 = y0 + crop_height
        frame = frame.crop((0, y0, frame.width, y1))

    elif width_scale > height_scale:
        crop_width = width * height_scale
        x0 = (frame.width - crop_width) // 2
        x1 = x0 + crop_width
        frame = frame.crop((x0, 0, x1, frame.height))

    # Resizes to desired height and width.
    frame = frame.resize((width, height), Image.LANCZOS)
    return frame


# =====================================================================================================================


def resize_long_edge(frame: Image, long_edge: int) -> Image:
    scale = long_edge / max(frame.size)
    height = round(frame.height * scale)
    width = round(frame.width * scale)
    frame = frame.resize((width, height), Image.LANCZOS)
    return frame


# =====================================================================================================================


class FrameWriteBuffer:
    def __init__(self, zipfile_path: Path, buffer_size: int = 100, **save_kwargs):
        self.zipfile_path = zipfile_path
        self.buffer_size = buffer_size
        self.save_kwargs = save_kwargs

        self.zipfile_lock = FileLock(f"{zipfile_path}.lock")
        self.frame_paths_buffer = []
        self.frame_bytes_buffer = []
        Image.init()

    def add(self, frame_path: str, frame: Image):
        self.frame_paths_buffer.append(frame_path)

        frame_bytes_io = BytesIO()
        frame.save(frame_bytes_io, format=Image.EXTENSION[Path(frame_path).suffix], **self.save_kwargs)
        frame_bytes = frame_bytes_io.getbuffer()
        self.frame_bytes_buffer.append(frame_bytes)

        if len(self.frame_paths_buffer) == self.buffer_size:
            self.flush()

    def flush(self):
        with self.zipfile_lock, ZipFile(file=self.zipfile_path, mode="a", compression=ZIP_STORED) as zf:
            for frame_path, frame_bytes in zip(self.frame_paths_buffer, self.frame_bytes_buffer):
                zf.writestr(frame_path, frame_bytes)

        self.frame_paths_buffer.clear()
        self.frame_bytes_buffer.clear()


# =====================================================================================================================


class ParallelProgressBar(joblib.Parallel):
    def set_tqdm_kwargs(self, **kwargs):
        self._tqdm_kwargs = kwargs

    def __call__(self, function: Callable[[int, tuple[Any, ...]], Any], args_list: list[tuple[Any, ...]]):
        tqdm_kwargs = getattr(self, "_tqdm_kwargs", dict())
        tqdm_kwargs["total"] = tqdm_kwargs.get("total", len(args_list))

        with tqdm.tqdm(**tqdm_kwargs) as self._progress_bar:
            return super().__call__(joblib.delayed(function)(*args) for args in args_list)

    def print_progress(self):
        self._progress_bar.n = self.n_completed_tasks
        self._progress_bar.refresh()
