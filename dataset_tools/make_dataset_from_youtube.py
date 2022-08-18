# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import av
import numpy as np
import yt_dlp

from dataset_tools.utils import FrameWriteBuffer, ParallelProgressBar, center_crop_and_resize, time_str_to_sec

# =====================================================================================================================


def save_video_clip(
    zipfile_path: Path, video_path: Path, clip_interval: str, height: int, width: int
) -> tuple[str, list[str]]:

    clip_name = clip_interval.replace(":", "_").replace(" ", "-")
    clip_path = Path(video_path.stem, clip_name)

    try:
        container = av.open(str(video_path))
        assert len(container.streams.video) > 0
    except:
        print(f"Failed to open video file: {video_path}")
        return str(clip_path), []

    video = container.streams.video[0]

    start_end_time = clip_interval.split(" ")
    assert len(start_end_time) == 2
    start_time = time_str_to_sec(start_end_time[0])
    end_time = time_str_to_sec(start_end_time[1])
    offset = int(start_time / video.time_base)
    container.seek(offset, stream=video)

    frame_index = 0
    frame_names = []
    frame_write_buffer = FrameWriteBuffer(zipfile_path, quality=100, subsample=0)
    frame_iterator = container.decode(video)

    while True:
        try:
            frame = next(frame_iterator)
        except:
            print(f"Failed to read frame from video file: {video_path}")
            break

        if frame.time < start_time:
            continue

        if frame.time > end_time:
            break

        # Converts video frame to PIL image.
        frame = frame.to_image()
        frame = center_crop_and_resize(frame, height, width)

        # Adds frame to the frame write buffer.
        frame_name = f"frame_{frame_index:06d}.jpg"
        frame_names.append(frame_name)
        frame_path = str(clip_path.joinpath(frame_name))
        frame_write_buffer.add(frame_path, frame)

        frame_index += 1

    frame_write_buffer.flush()
    return str(clip_path), frame_names


# =====================================================================================================================


def download_youtube_video(youtube_url: str, video_cache_dir: Path):
    video_path_template = str(video_cache_dir.joinpath("%(id)s.%(ext)s"))
    options = {"outtmpl": video_path_template, "quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(options) as downloader:
        downloader.download([youtube_url])


# =====================================================================================================================


def download_youtube_videos(youtube_ids: list[str], video_cache_dir: Path) -> list[Path]:
    video_cache_dir.mkdir(exist_ok=True)
    parallel_args = [(f"https://www.youtube.com/watch?v={youtube_id}", video_cache_dir) for youtube_id in youtube_ids]
    n_jobs = min(8, len(parallel_args))

    with ParallelProgressBar(n_jobs=n_jobs) as parallel:
        parallel.set_tqdm_kwargs(desc="Downloading YouTube videos")
        parallel(download_youtube_video, parallel_args)

    video_paths = [next(video_cache_dir.glob(f"{glob.escape(youtube_id)}.*")) for youtube_id in youtube_ids]
    return video_paths


# =====================================================================================================================


def make_dataset(
    clips_config_path: str,
    output_dir: str,
    video_cache_dir: str,
    height: int,
    width: int,
    partition: int,
    num_partitions: int,
):
    output_dir = Path(output_dir, f"{height:04d}x{width:04d}")
    output_dir.mkdir(parents=True, exist_ok=True)
    zipfile_path = output_dir.joinpath(f"partition_{partition:04d}.zip")

    video_cache_dir = Path(video_cache_dir)
    video_cache_dir.mkdir(parents=True, exist_ok=True)

    with open(clips_config_path) as open_file:
        clips_config = json.load(open_file)

    youtube_ids = clips_config.keys()
    video_paths = download_youtube_videos(youtube_ids, video_cache_dir)

    parallel_args = []
    for video_path, clip_intervals in zip(video_paths, clips_config.values()):
        for clip_interval in clip_intervals:
            parallel_args.append((zipfile_path, video_path, clip_interval, height, width))

    print(f"Partition index {partition} ({partition + 1} / {num_partitions})")
    parallel_args = np.array_split(parallel_args, num_partitions)[partition]

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
    parser = argparse.ArgumentParser(description="Process frames from YouTube videos to dataset partitions.")
    parser.add_argument("clips_config_path", help="Path to input JSON file specifying YouTube video clips.")
    parser.add_argument("output_dir", help="Path to output dataset directory.")
    parser.add_argument("video_cache_dir", help="Path to directory used for caching YouTube videos.")
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--num-partitions", type=int, default=10)
    args = parser.parse_args()

    make_dataset(
        args.clips_config_path,
        args.output_dir,
        args.video_cache_dir,
        args.height,
        args.width,
        args.partition,
        args.num_partitions,
    )
