# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import pathlib

import click
import matplotlib.pyplot as plt
import torch
import tqdm

import utils
from dataset import VideoDataset

# =====================================================================================================================


class RunningMeanStd:
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.sum_of_squares = 0.0

    def push(self, x: torch.Tensor):
        self.n += 1
        self.sum += x
        self.sum_of_squares += x**2

    def std_mean(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.n > 1
        mean = self.sum / self.n
        std = torch.sqrt((self.sum_of_squares - self.n * mean**2) / (self.n - 1))
        return std, mean


def video_color_intersection(video: torch.Tensor, bins_per_color: int = 20) -> torch.Tensor:
    assert video.dim() == 4  # C T H W
    num_pixels = video.size(2) * video.size(3)
    x = (video / 2 + 0.5) * (bins_per_color - 1)
    x = (x + 0.5).floor()
    x = x.clamp(0, bins_per_color - 1).type(torch.long)
    x = ((x[0] * bins_per_color) + x[1]) * bins_per_color + x[2]
    bins = bins_per_color**3
    x = torch.stack([torch.histc(xi, bins=bins, min=0, max=bins - 1) for xi in x])
    x = torch.minimum(x[1:], x[:1]).sum(dim=1)
    similarity = x / num_pixels
    return similarity


def plot(videos: torch.Tensor, label: str, color: str = "blue"):
    assert videos.dim() == 5  # N C T H W
    stats = RunningMeanStd()
    for video in tqdm.tqdm(videos, desc="Computing color intersection"):
        similarity = video_color_intersection(video.cuda())
        stats.push(similarity)
    std, mean = stats.std_mean()
    xs = torch.arange(mean.size(0) + 1)
    std = torch.cat((torch.tensor([0]), std.cpu()))
    mean = torch.cat((torch.tensor([1]), mean.cpu()))

    plt.plot(xs, mean, label=label, color=color)
    plt.fill_between(xs, mean - std, mean + std, alpha=0.2, color=color, linewidth=0, zorder=-10)
    plt.plot(max(xs), mean[-1], marker="o", markersize=4, color=color, zorder=10)


# =====================================================================================================================


@click.command()
@click.option("--path", help="Path to image filename for saving the plot", type=str, required=True)
@click.option("--dataset", "dataset_dir", help="Path to dataset directory", type=str, required=True)
@click.option("--lres", "lres_path", help="Low-res network pickle path/URL", type=str, required=True)
@click.option("--sres", "sres_path", help="Super-res network pickle path/URL", type=str, required=True)
@click.option("--len", "seq_length", help="Video length in frames", type=int, default=128)
@click.option("--samples", "num_samples", help="Number of video samples", type=int, default=1000)
@click.option("--batch", "batch_size", help="Batch size for generated video samples", type=int, default=10)
def plot_color_similarity(
    path: str,
    dataset_dir: str,
    lres_path: str,
    sres_path: str,
    seq_length: int,
    num_samples: int,
    batch_size: int,
):
    """Plot color similarity over time.
    Example:

    \b
    # Color similarity for pretrained horseback riding model.
    python plot_color_similarity.py --path=outputs/horseback_color_similarity.pdf --dataset=datasets/horseback \\
        --lres=https://nvlabs-fi-cdn.nvidia.com/long-video-gan/pretrained/horseback_lres.pkl \\
        --sres=https://nvlabs-fi-cdn.nvidia.com/long-video-gan/pretrained/horseback_sres.pkl

    """
    lres_G = utils.load_G(lres_path)
    sres_G = utils.load_G(sres_path)

    seq_length += 1
    segment_length = 16
    lr_seq_length = ((seq_length + segment_length - 1) // segment_length) * segment_length
    lr_seq_length = lr_seq_length if sres_path is None else lr_seq_length + 2 * sres_G.temporal_context

    dataset = VideoDataset(dataset_dir, seq_length, sres_G.hr_height, sres_G.hr_width)
    data_iter = utils.get_infinite_data_iter(dataset, batch_size=batch_size, num_workers=2, drop_last=True)

    generated_samples = []
    dataset_samples = []

    for _ in tqdm.trange(math.ceil(num_samples / batch_size), desc="Generating video batches"):
        lr_video = lres_G(batch_size, lr_seq_length)
        segments = sres_G.sample_video_segments(lr_video, segment_length)
        video = torch.cat(list(segments), dim=2)[:, :, :seq_length]
        generated_samples.append(video.cpu())
        dataset_samples.append(next(data_iter)["video"])

    generated_samples = torch.cat(generated_samples, dim=0)[:num_samples]
    dataset_samples = torch.cat(dataset_samples, dim=0)[:num_samples]

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 3), dpi=200)
    plt.xlabel("Frame separation")
    plt.ylabel("Color similarity")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plot(generated_samples, "LongVideoGAN", color="tab:blue")
    plot(dataset_samples, "Dataset", color="tab:orange")

    plt.xlim(0, seq_length + 2)
    plt.ylim(0, 1)
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()

    print(f"Saved plot {path}")


# =====================================================================================================================


if __name__ == "__main__":
    plot_color_similarity()
