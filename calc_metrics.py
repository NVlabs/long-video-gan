# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

import utils
from metrics import metric_main

# =====================================================================================================================


@click.command()
@click.option("--dataset", "dataset_dir", help="Path to dataset directory", type=str, required=True)
@click.option("--lres", "lres_path", help="Low-res network pickle path/URL", type=str, required=True)
@click.option("--sres", "sres_path", help="Super-res network pickle path/URL", type=str, required=True)
@click.option("--metric", "-m", "metrics", help="Metrics to compute", default=["fvd2048_128f", "fvd2048_16f"], type=str, multiple=True)  # fmt: skip
@click.option("--num-runs", help="How many runs of the metric to average over", type=int, default=1)
@click.option("--replace-cache", help="Whether to replace the dataset stats cache", type=bool, default=False)
@click.option("--verbose", help="Whether to log progress", type=bool, default=False)
@click.option("--path", "result_path", help="Path to JSON filename for saving metrics", type=str)
def calc_metrics(
    dataset_dir: str,
    lres_path: str,
    sres_path: str,
    metrics: list[str],
    num_runs: int,
    replace_cache: bool,
    verbose: bool,
    result_path: Optional[str],
):
    """Calculates metrics using pretrained model pickles.
    Examples:

    \b
    # Previous training run.
    python calc_metrics.py --dataset=datasets/horseback -m fvd2048_16f --verbose=True \\
        --lres=runs/lres/00000-horseback-64batch-2accum-1.0gamma/checkpoints/ckpt-00000000-G-ema.pkl \\
        --sres=runs/sres/00000-horseback-32batch-1accum-1.0gamma/checkpoints/ckpt-00000000-G-ema.pkl

    \b
    # Pretrained model.
    python calc_metrics.py --dataset=datasets/horseback -m fvd2048_128f -m fvd2048_16f -m fid50k_full --verbose=True \\
        --lres=https://nvlabs-fi-cdn.nvidia.com/long-video-gan/pretrained/horseback_lres.pkl \\
        --sres=https://nvlabs-fi-cdn.nvidia.com/long-video-gan/pretrained/horseback_sres.pkl
    """
    print(f"Metrics: {', '.join(metrics)}")

    if result_path is not None:
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)

    print("LR:", lres_path)
    print("SR:", sres_path)
    lres_G = utils.load_G(lres_path)
    sres_G = utils.load_G(sres_path)

    dataset_kwargs = dict(dataset_dir=dataset_dir, seq_length=1, height=sres_G.hr_height, width=sres_G.hr_width)

    for metric in metrics:
        result_dict = metric_main.calc_metric(
            metric=metric,
            G=sres_G,
            lr_G=lres_G,
            dataset_kwargs=dataset_kwargs,
            replace_cache=replace_cache,
            verbose=verbose,
            num_runs=num_runs,
        )
        json_line = json.dumps(
            dict(
                result_dict,
                lres_path=lres_path,
                sres_path=sres_path,
            )
        )
        print(json_line)

        if result_path is not None:
            with open(result_path, "at") as fp:
                fp.write(f"{json_line}\n")


# =====================================================================================================================


if __name__ == "__main__":
    calc_metrics()
