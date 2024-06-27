#!/bin/bash

clip_model=$1
n_shots=$2
epoch_multiplier=$3

python main.py fgvc_aircraft "$clip_model" "$n_shots" AUGMENTED_FGVC_TEMPLATES --epochs 100 --epoch-multiplier "$epoch_multiplier"
python main.py eurosat "$clip_model" "$n_shots" AUGMENTED_EUROSAT_TEMPLATES  --epochs 300 --batch-size 32 --epoch-multiplier "$epoch_multiplier"
python main.py caltech-101 "$clip_model" "$n_shots" CLIP_CALTECH_TEMPLATES --epochs 50 --epoch-multiplier "$epoch_multiplier"
python main.py dtd "$clip_model" "$n_shots" CLIP_DTD_TEMPLATES --epochs 60 --epoch-multiplier "$epoch_multiplier"
python main.py food-101 "$clip_model" "$n_shots" AUGMENTED_FOOD101_TEMPLATES  --epochs 50 --epoch-multiplier "$epoch_multiplier"
python main.py oxford_flowers "$clip_model" "$n_shots" AUGMENTED_FLOWERS_TEMPLATES --epochs 100 --epoch-multiplier "$epoch_multiplier"
python main.py oxford_pets "$clip_model" "$n_shots" AUGMENTED_OXFORD_PETS_TEMPLATES  --epochs 40 --epoch-multiplier "$epoch_multiplier"
python main.py standford_cars "$clip_model" "$n_shots" AUGMENTED_STANDFORD_CARS_TEMPLATES --epochs 40 --epoch-multiplier "$epoch_multiplier" # no proper tuning
python main.py ucf101 "$clip_model" "$n_shots" CLIP_UCF101_TEMPLATES --epochs 100 --epoch-multiplier "$epoch_multiplier"
python main.py imagenet "$clip_model" 16 CLIP_IMAGENET_TEMPLATES --epochs 20 --epoch-multiplier "$epoch_multiplier"


""