#!/bin/bash

dinov2_model=$1
dataset=$2
n_shots=$3
epoch_multiplier=$4
n_runs=$5
optional_args=$6  # <- Accept optional arguments like --no-mask

experiment="dinov2_soup_v3 $dinov2_model $n_shots"

echo "$dinov2_model"

if [ "$dataset" == "fgvc_aircraft" ]; then
    python main_dinov2_soup.py fgvc_aircraft "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment"  --n-runs "$n_runs" $optional_args
    exit
fi
if [ "$dataset" == "eurosat" ]; then
    python main_dinov2_soup.py eurosat "$dinov2_model" "$n_shots"  --epochs 400 --batch-size 32 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" $optional_args
    exit
fi
if [ "$dataset" == "caltech-101" ]; then
    python main_dinov2_soup.py caltech-101 "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" $optional_args
    exit
fi
if [ "$dataset" == "dtd" ]; then
    python main_dinov2_soup.py dtd "$dinov2_model" "$n_shots" --epochs 120 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" $optional_args
    exit
fi
if [ "$dataset" == "food-101" ]; then
    python main_dinov2_soup.py food-101 "$dinov2_model" "$n_shots"  --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" $optional_args
    exit
fi
if [ "$dataset" == "oxford_flowers" ]; then
    python main_dinov2_soup.py oxford_flowers "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" $optional_args
    exit
fi
if [ "$dataset" == "oxford_pets" ]; then
    python main_dinov2_soup.py oxford_pets "$dinov2_model" "$n_shots"  --epochs 150 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" $optional_args
    exit
fi
if [ "$dataset" == "standford_cars" ]; then
    python main_dinov2_soup.py standford_cars "$dinov2_model" "$n_shots" --epochs 40 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" $optional_args
    exit
fi
if [ "$dataset" == "ucf101" ]; then
    python main_dinov2_soup.py ucf101 "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" $optional_args
    exit
fi
if [ "$dataset" == "sun397" ]; then
    python main_dinov2_soup.py sun397 "$dinov2_model" "$n_shots" --epochs 30 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" $optional_args
    exit
fi

# Uncomment if needed for Imagenet:
# python main_dinov2_soup.py imagenet "$dinov2_model" "$n_shots" --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs" $optional_args
