#!/bin/bash

dinov2_model=$1
n_shots=$2
epoch_multiplier=$3
n_runs=$4

experiment="dinov2_soup $dinov2_model $n_shots"

python main_dinov2_soup.py fgvc_aircraft "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment"  --n-runs "$n_runs"
python main_dinov2_soup.py eurosat "$dinov2_model" "$n_shots"  --epochs 100 --batch-size 32 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
python main_dinov2_soup.py caltech-101 "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
python main_dinov2_soup.py dtd "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
python main_dinov2_soup.py food-101 "$dinov2_model" "$n_shots"  --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
python main_dinov2_soup.py oxford_flowers "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
python main_dinov2_soup.py oxford_pets "$dinov2_model" "$n_shots"  --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
python main_dinov2_soup.py standford_cars "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
python main_dinov2_soup.py ucf101 "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
#python main_dinov2_soup.py imagenet "$dinov2_model" "$n_shots" --epochs 15 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
