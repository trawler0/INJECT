bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "fgvc_aircraft" 2 0.5 10
bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "fgvc_aircraft" 4 0.5 10
bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "fgvc_aircraft" 8 0.5 10
bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "fgvc_aircraft" 16 0.5 10

for dataset in "eurosat" "caltech-101" "dtd" "food-101" "oxford_flowers" "oxford_pets" "stanford_cars" "ucf101"
do
  # Benchmarking CLIP
  bash scripts/benchmark_clip_soup.sh ViT-B/32 "$dataset" 2 0.5 10
  bash scripts/benchmark_clip_soup.sh ViT-B/32 "$dataset" 4 0.5 10
  bash scripts/benchmark_clip_soup.sh ViT-B/32 "$dataset" 8 0.5 10
  bash scripts/benchmark_clip_soup.sh ViT-B/32 "$dataset" 16 0.5 10

  # Benchmarking DinoV2
  bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "$dataset" 2 0.5 10
  bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "$dataset" 4 0.5 10
  bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "$dataset" 8 0.5 10
  bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "$dataset" 16 0.5 10
done

# Robustness tests for CLIP and DinoV2 on ImageNet
bash scripts/robustness_clip_soup_imagenet.sh ViT-B/32 1 10
bash scripts/robustness_dinov2_soup_imagenet.sh dinov2_vits14 1 10

# Running the main Python script for DinoV2
python main_dinov2_soup.py imagenet dinov2_vitl14_reg 16 --epochs 8 --epoch-multiplier 1 --experiment dinov2_heavy --n-runs 10 --val-frequency
