#!/bin/bash

model=$1
n_shot=$2
n_augs=$3

for dataset in caltech-101 dtd eurosat food-101 oxford_flowers standford_cars ucf101 fgvc_aircraft imagenet

do
    python cache_features.py "$dataset" "$model" --cache-prompts --n-shot "$n_shot" --n-augs "$n_augs"
done
""