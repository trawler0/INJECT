#!/bin/bash

model=$1
n_shot=$2

for dataset in caltech-101 food-101 standford_cars

do
    python cache_features.py "$dataset" "$model" --cache-prompts --n-shot "$n_shot" --n-augs 4
done
""