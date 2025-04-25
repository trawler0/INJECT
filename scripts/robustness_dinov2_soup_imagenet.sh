#!/bin/bash

dinov2_model=$1
epoch_multiplier=$2
n_runs=$3

experiment="dinov2_soup_imagenet_v2 $dinov2_model"
# python main_dinov2_soup.py imagenet "$dinov2_model" 32 --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" --val-frequency 1 --save checkpoints/"$dinov2_model"_32.pth
# python main_dinov2_soup.py imagenet "$dinov2_model" 16 --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" --val-frequency 1 --save checkpoints/"$dinov2_model"_16.pth
# python main_dinov2_soup.py imagenet "$dinov2_model" 8 --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" --val-frequency 1 --save checkpoints/"$dinov2_model"_8.pth
#python main_dinov2_soup.py imagenet "$dinov2_model" 4 --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" --val-frequency 1 --save checkpoints/"$dinov2_model"_4.pth
python main_dinov2_soup.py imagenet "$dinov2_model" 2 --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" --val-frequency 1 --save checkpoints/"$dinov2_model"_2.pth
