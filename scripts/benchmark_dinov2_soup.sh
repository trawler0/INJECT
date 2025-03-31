#!/bin/bash

dinov2_model=$1
dataset=$2
n_shots=$3
epoch_multiplier=$4
n_runs=$5

experiment="dinov2_soup_v2 $dinov2_model $n_shots"

echo "$dinov2_model"  # Fixed echo for variable

if [ "$dataset" == "fgvc_aircraft" ]; then
    python main_dinov2_soup.py fgvc_aircraft "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment"  --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "eurosat" ]; then
    python main_dinov2_soup.py eurosat "$dinov2_model" "$n_shots"  --epochs 100 --batch-size 32 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "caltech-101" ]; then
    python main_dinov2_soup.py caltech-101 "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "dtd" ]; then
    python main_dinov2_soup.py dtd "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "food-101" ]; then
    python main_dinov2_soup.py food-101 "$dinov2_model" "$n_shots"  --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "oxford_flowers" ]; then
    python main_dinov2_soup.py oxford_flowers "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "oxford_pets" ]; then
    python main_dinov2_soup.py oxford_pets "$dinov2_model" "$n_shots"  --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "stanford_cars" ]; then
    python main_dinov2_soup.py stanford_cars "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "ucf101" ]; then
    python main_dinov2_soup.py ucf101 "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi

# Uncomment if needed for Imagenet:
# python main_dinov2_soup.py imagenet "$dinov2_model" "$n_shots" --epochs 15 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
