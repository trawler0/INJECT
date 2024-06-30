#!/bin/bash

dinov2_model=$1
n_shots=$2
epoch_multiplier=$3

python main_dinov2.py fgvc_aircraft "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier"
python main_dinov2.py eurosat "$dinov2_model" "$n_shots"  --epochs 100 --batch-size 32 --epoch-multiplier "$epoch_multiplier"
python main_dinov2.py caltech-101 "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier"
python main_dinov2.py dtd "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier"
python main_dinov2.py food-101 "$dinov2_model" "$n_shots"  --epochs 80 --epoch-multiplier "$epoch_multiplier"
python main_dinov2.py oxford_flowers "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier"
python main_dinov2.py oxford_pets "$dinov2_model" "$n_shots"  --epochs 80 --epoch-multiplier "$epoch_multiplier"
python main_dinov2.py standford_cars "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" # no proper tuning
python main_dinov2.py ucf101 "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier"
python main_dinov2.py imagenet "$dinov2_model" "$n_shots" --epochs 15 --epoch-multiplier "$epoch_multiplier"


""