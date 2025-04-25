bash scripts/robustness_dinov2_soup_imagenet.sh dinov2_vits14 1 10
bash scripts/robustness_dinov2_soup_imagenet.sh dinov2_vitb14_reg .8 10
bash scripts/robustness_clip_soup_imagenet.sh ViT-B/16 .8 10
bash scripts/robustness_clip_soup_imagenet.sh ViT-B/32 1 10


for dataset in "sun397" "fgvc_aircraft" "eurosat" "caltech-101" "dtd" "food-101" "oxford_flowers" "oxford_pets" "standford_cars" "ucf101"
do
  # Benchmarking DinoV2
  bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "$dataset" 2 0.5 10
  bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "$dataset" 4 0.5 10
  bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "$dataset" 8 0.5 10
  bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "$dataset" 16 0.5 10


  # Benchmarking CLIP
  bash scripts/benchmark_clip_soup.sh ViT-B/32 "$dataset" 2 0.5 10
  bash scripts/benchmark_clip_soup.sh ViT-B/32 "$dataset" 4 0.5 10
  bash scripts/benchmark_clip_soup.sh ViT-B/32 "$dataset" 8 0.5 10
  bash scripts/benchmark_clip_soup.sh ViT-B/32 "$dataset" 16 0.5 10

done




