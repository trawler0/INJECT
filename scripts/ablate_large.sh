#!/bin/bash

n_shot=$1
epoch_multiplier=$2
n_runs_dinov2=$3
n_runs_clip=$4

experiment="ablate_dino_arch"
python main_dinov2_soup.py imagenet "dinov2_vitg14_reg" "$n_shot" --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs_dinov2" --save checkpoints/"dinov2_vitg14_reg_$n_shot".pth
python main_dinov2_soup.py imagenet "dinov2_vitl14_reg" "$n_shot" --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs_dinov2" --save checkpoints/"dinov2_vitl14_reg_$n_shot".pth
python main_dinov2_soup.py imagenet "dinov2_vitb14_reg" "$n_shot" --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs_dinov2" --save checkpoints/"dinov2_vitb14_reg_$n_shot".pth
python main_dinov2_soup.py imagenet "dinov2_vits14" "$n_shot" --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs_dinov2" --save checkpoints/"dinov2_vits14_$n_shot".pth

python main_clip_soup.py imagenet ViT-L/14@336px "$n_shot" CLIP_IMAGENET_TEMPLATES --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs_clip" --save checkpoints/"vit-l_14@336px_$n_shot".pth
python main_clip_soup.py imagenet ViT-L/14 "$n_shot" CLIP_IMAGENET_TEMPLATES --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs_clip" --save checkpoints/"vit-l_14_$n_shot".pth
python main_clip_soup.py imagenet ViT-B/16 "$n_shot" CLIP_IMAGENET_TEMPLATES --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs_clip" --save checkpoints/"vit-b_16_$n_shot".pth
python main_clip_soup.py imagenet ViT-B/32 "$n_shot" CLIP_IMAGENET_TEMPLATES --epochs 10 --epoch-multiplier "$epoch_multiplier" --experiment "$experiment" --n-runs "$n_runs_clip" --save checkpoints/"vit-b_32_$n_shot".pth
