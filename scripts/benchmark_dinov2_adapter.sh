#!/bin/bash

dinov2_model=$1
n_shots=$2
epoch_multiplier=$3

experiment="dinov2_adapter $dinov2_model $n_shots"

#python main_dinov2_adapter.py fgvc_aircraft "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment"
python main_dinov2_adapter.py fgvc_aircraft "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --no-masking

python main_dinov2_adapter.py eurosat "$dinov2_model" "$n_shots"  --epochs 100 --batch-size 32 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment"
python main_dinov2_adapter.py eurosat "$dinov2_model" "$n_shots"  --epochs 100 --batch-size 32 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --no-masking

python main_dinov2_adapter.py caltech-101 "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment"
python main_dinov2_adapter.py caltech-101 "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --no-masking

python main_dinov2_adapter.py dtd "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment"
python main_dinov2_adapter.py dtd "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --no-masking

python main_dinov2_adapter.py food-101 "$dinov2_model" "$n_shots"  --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment"
python main_dinov2_adapter.py food-101 "$dinov2_model" "$n_shots"  --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --no-masking

python main_dinov2_adapter.py oxford_flowers "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment"
python main_dinov2_adapter.py oxford_flowers "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --no-masking

python main_dinov2_adapter.py oxford_pets "$dinov2_model" "$n_shots"  --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment"
python main_dinov2_adapter.py oxford_pets "$dinov2_model" "$n_shots"  --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --no-masking

python main_dinov2_adapter.py standford_cars "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment"
python main_dinov2_adapter.py standford_cars "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --no-masking

python main_dinov2_adapter.py ucf101 "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment"
python main_dinov2_adapter.py ucf101 "$dinov2_model" "$n_shots" --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --no-masking

#python main_dinov2_adapter.py imagenet "$dinov2_model" "$n_shots" --epochs 15 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment"
#python main_dinov2_adapter.py imagenet "$dinov2_model" "$n_shots" --epochs 15 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --no-masking