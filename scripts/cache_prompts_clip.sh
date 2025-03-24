#!/bin/bash

clip_model=$1
n_shot=$2

python cache_features.py caltech-101 "$clip_model" --cache-prompts --templates CLIP_CALTECH_TEMPLATES --n-shot "$n_shot" --n-augs 1
python cache_features.py dtd "$clip_model" --cache-prompts --templates CLIP_DTD_TEMPLATES --n-shot "$n_shot" --n-augs 1
python cache_features.py eurosat "$clip_model" --cache-prompts --templates CLIP_EUROSAT_TEMPLATES --n-shot "$n_shot" --n-augs 1
python cache_features.py eurosat "$clip_model" --cache-prompts --templates AUGMENTED_EUROSAT_TEMPLATES --n-shot "$n_shot" --n-augs 1
python cache_features.py food-101 "$clip_model" --cache-prompts --templates CLIP_FOOD101_TEMPLATES --n-shot "$n_shot" --n-augs 1
python cache_features.py food-101 "$clip_model" --cache-prompts --templates AUGMENTED_FOOD101_TEMPLATES --n-shot "$n_shot" --n-augs 1
python cache_features.py oxford_flowers "$clip_model" --cache-prompts --templates CLIP_FLOWERS_TEMPLATES --n-shot "$n_shot" --n-augs 1
python cache_features.py oxford_flowers "$clip_model" --cache-prompts --templates AUGMENTED_FLOWERS_TEMPLATES --n-shot "$n_shot" --n-augs 1
python cache_features.py oxford_pets "$clip_model" --cache-prompts --templates CLIP_OXFORD_PETS_TEMPLATES --n-shot "$n_shot" --n-augs 1
python cache_features.py oxford_pets "$clip_model" --cache-prompts --templates AUGMENTED_OXFORD_PETS_TEMPLATES --n-shot "$n_shot" --n-augs 1
python cache_features.py standford_cars "$clip_model" --cache-prompts --templates CLIP_STANDFORD_CARS_TEMPLATES --n-shot "$n_shot" --n-augs 1
python cache_features.py standford_cars "$clip_model" --cache-prompts --templates AUGMENTED_STANDFORD_CARS_TEMPLATES --n-shot "$n_shot" --n-augs 1
python cache_features.py ucf101 "$clip_model" --cache-prompts --templates CLIP_UCF101_TEMPLATES --n-shot "$n_shot" --n-augs 1
python cache_features.py fgvc_aircraft "$clip_model" --cache-prompts --templates CLIP_FGVC_TEMPLATES --n-shot "$n_shot" --n-augs 1
python cache_features.py fgvc_aircraft "$clip_model" --cache-prompts --templates AUGMENTED_FGVC_TEMPLATES --n-shot "$n_shot" --n-augs 1
python cache_features.py imagenet "$clip_model" --cache-prompts --templates CLIP_IMAGENET_TEMPLATES --n-shot "$n_shot" --n-augs 1

""