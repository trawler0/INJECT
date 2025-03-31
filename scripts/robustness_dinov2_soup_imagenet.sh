#!/bin/bash

dinov2_model=$1
epoch_multiplier=$2
n_runs=$3

experiment="dinov2_soup_imagenet_dummy $dinov2_model"
python main_dinov2_soup.py imagenet "$dinov2_model" 32 --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
python main_dinov2_soup.py imagenet "$dinov2_model" 16 --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
python main_dinov2_soup.py imagenet "$dinov2_model" 8 --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
python main_dinov2_soup.py imagenet "$dinov2_model" 4 --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
python main_dinov2_soup.py imagenet "$dinov2_model" 2 --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
