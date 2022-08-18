#!/bin/bash

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Script to launch dataset creation on Slurm. Replace SRC_DIR and DST_DIR with source and destination directories.
# Usage: sbatch make_dataset_sbatch.sh

#SBATCH -N 1
#SBATCH -n 10
#SBATCH -c 20
#SBATCH --exclusive

echo "Starting job..."
source anaconda3/bin/activate
conda activate dataset_tools

for i in $(seq 0 10)
do
    args=(
        SRC_DIR
        DST_DIR
        --num-partitions 10
        --partition $i
        --height 144
        --width 256
        # --wait-for-list-file
    )
    # [[ $i != 0 ]] && args+=(--wait-for-list-file)
    srun -N 1 -n 1 --exclusive python dataset_tools/make_dataset_from_videos.py "${args[@]}" &
done

wait
