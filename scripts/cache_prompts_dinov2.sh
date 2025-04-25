#!/bin/bash

model=$1
n_shot=$2
n_augs=$3

for dataset in sun397

do
    python cache_features.py "$dataset" "$model" --cache-prompts --n-shot "$n_shot" --n-augs "$n_augs"
done
""