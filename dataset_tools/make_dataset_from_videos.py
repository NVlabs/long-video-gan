# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional
from zipfile import ZIP_STORED, ZipFile

import av
import numpy as np
import tqdm

from dataset_tools.utils import FrameWriteBuffer, ParallelProgressBar, center_crop_and_resize, resize_long_edge

# =====================================================================================================================


def save_video_clip(
    zipfile_path: Path,
    video_path: Path,
    video_relative_dir: Path,
    height: Optional[int],
    width: Optional[int],
    long_edge: Optional[int],
    trim_start: float,
    trim_end: float,
) -> tuple[str, list[str]]:

    try:
        container = av.open(str(video_path))
        assert len(container.streams.video) > 0
    except:
        print(f"Failed to open video file: {video_path}")
        return str(video_relative_dir), []

    video = container.streams.video[0]

    end_time = float(video.duration * video.time_base) - trim_end
    offset = int(trim_start / video.time_base)

    if offset != 0:
        try:
            container.seek(offset, stream=video)
        except:
            print(f"Failed to seek in video: {video_path}")
            return str(video_relative_dir), []

    frame_index = 0
    frame_names = []
    frame_write_buffer = FrameWriteBuffer(zipfile_path, quality=100, subsample=0)
    frame_iterator = container.decode(video)

    while True:
        try:
            frame = next(frame_iterator)
        except:
            break

        if frame.time < trim_start:
            continue

        if frame.time > end_time:
            break

        # Converts video frame to PIL image.
        frame = frame.to_image()
        if height is not None and width is not None:
            frame = center_crop_and_resize(frame, height, width)
        elif long_edge is not None:
            frame = resize_long_edge(frame, long_edge)

        # Adds frame to the frame write buffer.
        frame_name = f"frame_{frame_index:06d}.jpg"
        frame_names.append(frame_name)
        frame_path = str(video_relative_dir.joinpath(frame_name))
        frame_write_buffer.add(frame_path, frame)

        frame_index += 1

    frame_write_buffer.flush()
    return str(video_relative_dir), frame_names


# =====================================================================================================================


def list_video_paths(
    input_dir: str, extensions: tuple[str, ...] = (".avi", ".mkv", ".mov", ".mp4", ".wmv")
) -> list[tuple[Path, Path]]:

    if not Path(input_dir).is_dir():
        raise ValueError(f"Directory not found: {input_dir}")

    paths = []
    directories = [input_dir]

    with tqdm.tqdm(desc=f"Listing video paths") as progress_bar:
        while directories:
            directory = directories.pop()
            for entry in os.scandir(directory):

                if entry.is_dir(follow_symlinks=False):
                    directories.append(entry.path)

                elif entry.name.lower().endswith(extensions):
                    paths.append((Path(entry.path), Path(entry.path).relative_to(input_dir).with_suffix("")))
                    progress_bar.update()

    paths.sort()
    return paths


# =====================================================================================================================


def make_dataset(
    input_dir: str,
    output_dir: str,
    height: Optional[int],
    width: Optional[int],
    long_edge: Optional[int],
    trim_start: float,
    trim_end: float,
    partition: int,
    num_partitions: int,
):
    assert (height is None and width is None) or (height is not None and width is not None and long_edge is None)

    output_dir = Path(output_dir)
    if height is not None and width is not None:
        output_dir = output_dir.joinpath(f"{height:04d}x{width:04d}")

    output_dir.mkdir(parents=True, exist_ok=True)
    zipfile_path = output_dir.joinpath(f"partition_{partition:04d}.zip")
    video_paths = list_video_paths(input_dir)

    print(f"Partition index {partition} ({partition + 1} / {num_partitions})")
    video_paths = np.array_split(video_paths, num_partitions)[partition]

    parallel_args = [
        (zipfile_path, video_path, video_relative_dir, height, width, long_edge, trim_start, trim_end)
        for video_path, video_relative_dir in video_paths
    ]

    with ParallelProgressBar(n_jobs=-1) as parallel:
        parallel.set_tqdm_kwargs(desc="Saving video clips to ZIP file")
        frame_paths = parallel(save_video_clip, parallel_args)

    frame_paths = [(video_relative_dir, frame_names) for video_relative_dir, frame_names in frame_paths if frame_names]
    frame_paths_json = json.dumps(dict(frame_paths), indent=2, sort_keys=True)

    with ZipFile(file=zipfile_path, mode="a", compression=ZIP_STORED) as zf:
        zf.writestr("frame_paths.json", frame_paths_json)

    Path(f"{zipfile_path}.lock").unlink(missing_ok=True)


# =====================================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process frames from videos to dataset partitions.")
    parser.add_argument("input_dir", help="Path to input video directory.")
    parser.add_argument("output_dir", help="Path to output dataset directory.")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--long-edge", type=int)
    parser.add_argument("--trim-start", type=float, default=0.0)
    parser.add_argument("--trim-end", type=float, default=0.0)
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--num-partitions", type=int, default=10)
    args = parser.parse_args()

    make_dataset(
        args.input_dir,
        args.output_dir,
        args.height,
        args.width,
        args.long_edge,
        args.trim_start,
        args.trim_end,
        args.partition,
        args.num_partitions,
    )
