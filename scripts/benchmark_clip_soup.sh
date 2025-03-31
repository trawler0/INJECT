#!/bin/bash

clip_model=$1
dataset=$2
n_shots=$3
epoch_multiplier=$4
n_runs=$5

echo "$clip_model"  # Fixed echo for variable

experiment="clip_soup_v2 $clip_model $n_shots"

if [ "$dataset" == "fgvc_aircraft" ]; then
    python main_clip_soup.py "fgvc_aircraft" "$clip_model" "$n_shots" AUGMENTED_FGVC_TEMPLATES --epochs 100 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "eurosat" ]; then
    python main_clip_soup.py "eurosat" "$clip_model" "$n_shots" AUGMENTED_EUROSAT_TEMPLATES  --epochs 300 --batch-size 32 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "caltech-101" ]; then
    python main_clip_soup.py "caltech-101" "$clip_model" "$n_shots" CLIP_CALTECH_TEMPLATES --epochs 50 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "dtd" ]; then
    python main_clip_soup.py "dtd" "$clip_model" "$n_shots" CLIP_DTD_TEMPLATES --epochs 60 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "food-101" ]; then
    python main_clip_soup.py "food-101" "$clip_model" "$n_shots" AUGMENTED_FOOD101_TEMPLATES  --epochs 50 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "oxford_flowers" ]; then
    python main_clip_soup.py "oxford_flowers" "$clip_model" "$n_shots" AUGMENTED_FLOWERS_TEMPLATES --epochs 100 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "oxford_pets" ]; then
    python main_clip_soup.py "oxford_pets" "$clip_model" "$n_shots" AUGMENTED_OXFORD_PETS_TEMPLATES  --epochs 40 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "stanford_cars" ]; then
    python main_clip_soup.py "stanford_cars" "$clip_model" "$n_shots" AUGMENTED_STANDFORD_CARS_TEMPLATES --epochs 40 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi
if [ "$dataset" == "ucf101" ]; then
    python main_clip_soup.py "ucf101" "$clip_model" "$n_shots" CLIP_UCF101_TEMPLATES --epochs 100 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs"
    exit
fi

# Uncomment if needed for Imagenet:
# python main_clip_soup.py imagenet "$clip_model" "$n_shots" CLIP_IMAGENET_TEMPLATES --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs
