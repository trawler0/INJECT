#!/bin/bash

dinov2_model=$1
epoch_multiplier=$2
n_runs=$3

experiment="clip_soup_imagenet_v2 $dinov2_model"
python main_clip_soup.py imagenet "$dinov2_model" 32 CLIP_IMAGENET_TEMPLATES --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
python main_clip_soup.py imagenet "$dinov2_model" 16 CLIP_IMAGENET_TEMPLATES --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
python main_clip_soup.py imagenet "$dinov2_model" 8 CLIP_IMAGENET_TEMPLATES --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
python main_clip_soup.py imagenet "$dinov2_model" 4 CLIP_IMAGENET_TEMPLATES --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
python main_clip_soup.py imagenet "$dinov2_model" 2 CLIP_IMAGENET_TEMPLATES --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
