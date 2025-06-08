#!/bin/bash

clip_model=$1
for dataset in "caltech-101" "dtd" "eurosat" "food-101" "oxford_flowers" "oxford_pets" "standford_cars" "ucf101" "fgvc_aircraft" # "sun397"
do
  for split in "val" "test";
  do
    python cache_features.py $dataset "$clip_model" --split $split --cache-dataset
  done
done

for split in "v2" "val" "sketch" "imagenet-a" "imagenet-r";
do
  python cache_features.py imagenet "$clip_model" --split $split --cache-dataset
done

