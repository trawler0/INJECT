#!/bin/bash
dinov2_model=$1
epoch_multiplier=$2

python main_dinov2_soup.py fgvc_aircraft "$dinov2_model" 2 --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment ablate_mask  --n-runs 5
python main_dinov2_soup.py fgvc_aircraft "$dinov2_model" 2 --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment ablate_mask  --n-runs 5 --no-mask

python main_dinov2_soup.py fgvc_aircraft "$dinov2_model" 4 --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment ablate_mask  --n-runs 5
python main_dinov2_soup.py fgvc_aircraft "$dinov2_model" 4 --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment ablate_mask  --n-runs 5 --no-mask

python main_dinov2_soup.py fgvc_aircraft "$dinov2_model" 8 --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment ablate_mask  --n-runs 5
python main_dinov2_soup.py fgvc_aircraft "$dinov2_model" 8 --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment ablate_mask  --n-runs 5 --no-mask

python main_dinov2_soup.py fgvc_aircraft "$dinov2_model" 16 --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment ablate_mask  --n-runs 5
python main_dinov2_soup.py fgvc_aircraft "$dinov2_model" 16 --epochs 80 --epoch-multiplier "$epoch_multiplier" --experiment ablate_mask  --n-runs 5 --no-mask
