#!/bin/bash

clip_model=$1

for split in "val" "test";
do
  python cache_features.py caltech-101 "$clip_model" --split $split --cache-dataset
  python cache_features.py dtd "$clip_model" --split $split --cache-dataset
  python cache_features.py eurosat "$clip_model" --split $split --cache-dataset
  python cache_features.py food-101 "$clip_model" --split $split --cache-dataset
  python cache_features.py oxford_flowers "$clip_model" --split $split --cache-dataset
  python cache_features.py oxford_pets "$clip_model" --split $split --cache-dataset
  python cache_features.py standford_cars "$clip_model" --split $split --cache-dataset
  python cache_features.py ucf101 "$clip_model" --split $split --cache-dataset
  python cache_features.py fgvc_aircraft "$clip_model" --split $split --cache-dataset
done

for split in "v2" "val" "sketch" "imagenet-a" "imagenet-r";
do
  python cache_features.py imagenet "$clip_model" --split $split --cache-dataset
done

""