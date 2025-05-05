#!/bin/bash

model=$1
n_shot=$2
n_augs=$3

for dataset in  fgvc_aircraft ucf101 standford_cars oxford_pets oxford_flowers food-101 eurosat dtd caltech-101 imagenet # sun397

do
    python cache_features.py "$dataset" "$model" --cache-prompts --n-shot "$n_shot" --n-augs "$n_augs"
done