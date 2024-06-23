#!/bin/bash

clip_model=$1
n_shots=$2
# python main.py caltech-101 "$clip_model" CLIP_CALTECH_TEMPLATES --epochs 10  # mislabeled
python main.py dtd "$clip_model" "$n_shots" CLIP_DTD_TEMPLATES --epochs 40
python main.py eurosat "$clip_model" "$n_shots" AUGMENTED_EUROSAT_TEMPLATES  --epochs 100 --batch-size 32
python main.py food-101 "$clip_model" "$n_shots" AUGMENTED_FOOD101_TEMPLATES  --epochs 30
# python main.py oxford_flowers "$clip_model" "$n_shots" AUGMENTED_FLOWERS_TEMPLATES  # mislabeled
python main.py oxford_pets "$clip_model" "$n_shots" AUGMENTED_OXFORD_PETS_TEMPLATES  --epochs 35
# python main.py standford_cars "$clip_model" "$n_shots" AUGMENTED_STANDFORD_CARS_TEMPLATES  # mislabeled
python main.py ucf101 "$clip_model" "$n_shots" CLIP_UCF101_TEMPLATES --epochs 35
python main.py fgvc_aircraft "$clip_model" "$n_shots" AUGMENTED_FGVC_TEMPLATES --epochs 35
python main.py imagenet "$clip_model" "$n_shots" CLIP_IMAGENET_TEMPLATES --epochs 15
