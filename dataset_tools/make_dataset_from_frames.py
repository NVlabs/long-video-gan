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
import time
import uuid
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import tqdm
from PIL import Image

from dataset_tools.utils import FrameWriteBuffer, ParallelProgressBar, center_crop_and_resize

# =====================================================================================================================


def save_video_clip(
    zipfile_path: Path, relative_dir: str, paths: list[str], height: int, width: int
) -> tuple[str, list[str]]:

    frame_names = []
    # frame_write_buffer = FrameWriteBuffer(zipfile_path, quality=100, subsample=0)
    frame_write_buffer = FrameWriteBuffer(zipfile_path, quality=95)

    for path in paths:
        frame = Image.open(path)
        frame = center_crop_and_resize(frame, height, width)

        # Adds frame to the frame write buffer.
        frame_name = Path(path).with_suffix(".jpg").name
        frame_names.append(frame_name)
        frame_path = str(Path(relative_dir).joinpath(frame_name))
        frame_write_buffer.add(frame_path, frame)

    frame_write_buffer.flush()
    return relative_dir, frame_names


# =====================================================================================================================


def list_frame_paths(input_dir: str, wait_for_list_file: bool = False) -> list[tuple[str, list[str]]]:
    if not Path(input_dir).is_dir():
        raise ValueError(f"Directory not found: {input_dir}")

    list_path = Path(input_dir).joinpath("frame_paths.json")
    while True:
        if list_path.is_file():
            with open(list_path, "r") as fp:
                return json.load(fp)
        if not wait_for_list_file:
            break
        time.sleep(10)

    Image.init()
    extensions = tuple(Image.EXTENSION.keys())

    frame_paths = {}
    directories = [input_dir]

    with tqdm.tqdm(desc=f"Listing frame paths") as progress_bar:
        while directories:
            directory = directories.pop()
            for entry in os.scandir(directory):

                if entry.is_dir(follow_symlinks=False):
                    directories.append(entry.path)

                elif entry.name.lower().endswith(extensions):
                    relative_dir = str(Path(entry).parent.relative_to(input_dir))
                    if relative_dir in frame_paths:
                        frame_paths[relative_dir].append(entry.path)
                    else:
                        frame_paths[relative_dir] = [entry.path]
                    progress_bar.update()

    for relative_dir, paths in frame_paths.items():
        frame_paths[relative_dir] = sorted(paths)
    frame_paths = sorted(list(frame_paths.items()))

    temp_path = list_path.with_name(f"temp_{uuid.uuid4().hex}")
    with open(temp_path, "w") as fp:
        json.dump(frame_paths, fp, indent=2)
    os.replace(temp_path, list_path)
    print("Saved list of frame paths.")

    return frame_paths


# =====================================================================================================================


def make_dataset(
    input_dir: str,
    output_dir: str,
    height: int,
    width: int,
    partition: int,
    num_partitions: int,
    wait_for_list_file: bool,
):
    output_dir = Path(output_dir, f"{height:04d}x{width:04d}")
    output_dir.mkdir(parents=True, exist_ok=True)
    zipfile_path = output_dir.joinpath(f"partition_{partition:04d}.zip")
    frame_paths = list_frame_paths(input_dir, wait_for_list_file)

    print(f"Partition index {partition} ({partition + 1} / {num_partitions})")
    frame_paths = np.array_split(np.array(frame_paths, dtype=object), num_partitions)[partition]

    parallel_args = [(zipfile_path, relative_dir, paths, height, width) for relative_dir, paths in frame_paths]

    with ParallelProgressBar(n_jobs=-1) as parallel:
        parallel.set_tqdm_kwargs(desc="Saving video clips to ZIP file")
        frame_paths = parallel(save_video_clip, parallel_args)

    frame_paths_json = json.dumps(dict(frame_paths), indent=2, sort_keys=True)

    with ZipFile(file=zipfile_path, mode="a", compression=ZIP_STORED) as zf:
        zf.writestr("frame_paths.json", frame_paths_json)

    Path(f"{zipfile_path}.lock").unlink(missing_ok=True)


# =====================================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process directories of frames to dataset partitions.")
    parser.add_argument("input_dir", help="Path to input frame directories.")
    parser.add_argument("output_dir", help="Path to output dataset directory.")
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--num-partitions", type=int, default=10)
    parser.add_argument("--wait-for-list-file", default=False, action="store_true")
    args = parser.parse_args()

    make_dataset(
        args.input_dir,
        args.output_dir,
        args.height,
        args.width,
        args.partition,
        args.num_partitions,
        args.wait_for_list_file,
    )
