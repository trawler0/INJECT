#!/bin/bash

clip_model=$1

python cache_features.py caltech-101 "$clip_model" $split --cache-prompts --templates CLIP_CALTECH_TEMPLATES
python cache_features.py dtd "$clip_model" $split --cache-prompts --templates CLIP_DTD_TEMPLATES
python cache_features.py eurosat "$clip_model" $split --cache-prompts --templates CLIP_EUROSAT_TEMPLATES
python cache_features.py eurosat "$clip_model" $split --cache-prompts --templates AUGMENTED_EUROSAT_TEMPLATES
python cache_features.py food-101 "$clip_model" $split --cache-prompts --templates CLIP_FOOD101_TEMPLATES
python cache_features.py food-101 "$clip_model" $split --cache-prompts --templates AUGMENTED_FOOD101_TEMPLATES
python cache_features.py oxford_flowers "$clip_model" $split --cache-prompts --templates CLIP_FLOWERS_TEMPLATES
python cache_features.py oxford_flowers "$clip_model" $split --cache-prompts --templates AUGMENTED_FLOWERS_TEMPLATES
python cache_features.py oxford_pets "$clip_model" $split --cache-prompts --templates CLIP_OXFORD_PETS_TEMPLATES
python cache_features.py oxford_pets "$clip_model" $split --cache-prompts --templates AUGMENTED_OXFORD_PETS_TEMPLATES
python cache_features.py standford_cars "$clip_model" $split --cache-prompts --templates CLIP_STANDFORD_CARS_TEMPLATES
python cache_features.py standford_cars "$clip_model" $split --cache-prompts --templates AUGMENTED_STANDFORD_CARS_TEMPLATES
python cache_features.py ucf101 "$clip_model" $split --cache-prompts --templates CLIP_UCF101_TEMPLATES
python cache_features.py fgvc_aircraft "$clip_model" $split --cache-prompts --templates CLIP_FGVC_TEMPLATES
python cache_features.py fgvc_aircraft "$clip_model" $split --cache-prompts --templates AUGMENTED_FGVC_TEMPLATES
python cache_features.py imagenet "$clip_model" $split --cache-prompts --templates CLIP_IMAGENET_TEMPLATES