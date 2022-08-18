# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import pathlib

import click
import torch

import utils

# =====================================================================================================================


@click.command()
@click.option("--outdir", help="Where to save the output videos", type=str, required=True)
@click.option("--seed", help="Random seed", type=int, required=True)
@click.option("--lres", "lres_path", help="Low-res network pickle path/URL", type=str, required=True)
@click.option("--sres", "sres_path", help="Super-res network pickle path/URL", type=str)
@click.option("--len", "seq_length", help="Video length in frames", type=int, default=301)
@click.option("--save-lres", help="Whether to also save the low res video", type=bool, default=False)
@click.option("--save-index", "-i", "save_frame_indices", help="Frame indices to save as images", default=[], type=int, multiple=True)  # fmt: skip
def generate(
    outdir: str,
    seed: int,
    lres_path: str,
    sres_path: str,
    seq_length: int,
    save_lres: bool,
    save_frame_indices: list[int],
):
    """Generate videos using pretrained model pickles.
    Examples:

    \b
    # Generate high-resolution video using pre-trained horseback riding model.
    python generate.py --outdir=outputs/horseback --seed=49 \\
        --lres=https://nvlabs-fi-cdn.nvidia.com/long-video-gan/pretrained/horseback_lres.pkl \\
        --sres=https://nvlabs-fi-cdn.nvidia.com/long-video-gan/pretrained/horseback_sres.pkl
    
    \b
    # Generate low-resolution video using pre-trained horseback riding model.
    python generate.py --outdir=outputs/horseback --seed=49 --save-lres=True \\
        --lres=https://nvlabs-fi-cdn.nvidia.com/long-video-gan/pretrained/horseback_lres.pkl

    \b
    # Generate low- and high-resolution videos and frame images using pre-trained mountain biking model.
    python generate.py --outdir=outputs/biking --seed=41 --save-lres=True -i 0 -i 15 -i 30 -i 60 -i 150 -i 300 \\
        --lres=https://nvlabs-fi-cdn.nvidia.com/long-video-gan/pretrained/biking_lres.pkl \\
        --sres=https://nvlabs-fi-cdn.nvidia.com/long-video-gan/pretrained/biking_sres.pkl
    """
    lres_G = utils.load_G(lres_path)
    sres_G = None if sres_path is None else utils.load_G(sres_path)

    print("Generating video...")
    segment_length = 16
    lr_seq_length = ((seq_length + segment_length - 1) // segment_length) * segment_length
    lr_seq_length = lr_seq_length if sres_path is None else lr_seq_length + 2 * sres_G.temporal_context
    generator = None if seed is None else torch.Generator("cuda").manual_seed(seed)
    lr_video = lres_G(1, lr_seq_length, generator_emb=generator)

    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    if sres_path is not None:
        # Returns an iterator over segments, which enables efficiently handling long videos.
        segments = sres_G.sample_video_segments(lr_video, segment_length, generator_z=generator)
        video = torch.cat(list(segments), dim=2)[:, :, :seq_length]
        path = pathlib.Path(outdir, f"seed={seed}_len={seq_length}_sres.mp4")
        print(f"Saving high-resolution video: {path}")
        utils.write_video_grid(video, path)

        if len(save_frame_indices) > 0:
            print(f"Saving frame images: {pathlib.Path(outdir, f'seed={seed}_len={seq_length}_frame=*.png')}")
        for i in save_frame_indices:
            utils.save_image_grid(
                video[:, :, i], pathlib.Path(outdir, f"seed={seed}_len={seq_length}_frame={i:04d}.png")
            )

        lr_video = lr_video[:, :, sres_G.temporal_context : sres_G.temporal_context + seq_length]

    if save_lres:
        path = pathlib.Path(outdir, f"seed={seed}_len={seq_length}_lres.mp4")
        print(f"Saving low-resolution video: {path}")
        utils.write_video_grid(lr_video, path)

    print("Enjoy!")


# =====================================================================================================================


if __name__ == "__main__":
    generate()
