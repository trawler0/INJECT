#!/bin/bash

clip_model=$1
dinov2_model=$2
epoch_multiplier=$3
n_runs=$4

experiment="ablate_k $clip_model $dinov2_model"
# python main_clip_soup.py imagenet "$clip_model" 8 CLIP_IMAGENET_TEMPLATES --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" --eval-continuously
python main_dinov2_soup.py imagenet "$dinov2_model" 8 --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" --val-frequency 1 --save checkpoints/"$dinov2_model"_16.pth --eval-continuously
