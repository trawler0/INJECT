# smaller models
# bash scripts/robustness_dinov2_soup_imagenet.sh dinov2_vits14 1 8
bash scripts/robustness_clip_soup_imagenet.sh ViT-B/32 1 8

# larger models
#bash scripts/robustness_dinov2_soup_imagenet.sh dinov2_vitb14_reg .8 8
#bash scripts/robustness_clip_soup_imagenet.sh ViT-B/16 .8 8

# standard evaluation for adapters
for dataset in  "eurosat" "caltech-101" "dtd" "food-101" "oxford_flowers" "oxford_pets" "standford_cars" "ucf101" # "sun397"
do
  # Benchmarking DinoV2
  #bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "$dataset" 2 1 8 --no-mask
  #bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "$dataset" 4 1 8 --no-mask
  #bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "$dataset" 8 1 8
  #bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 "$dataset" 16 1 8


  # Benchmarking CLIP
  bash scripts/benchmark_clip_soup.sh ViT-B/32 "$dataset" 2 1 8
  bash scripts/benchmark_clip_soup.sh ViT-B/32 "$dataset" 4 1 8
  bash scripts/benchmark_clip_soup.sh ViT-B/32 "$dataset" 8 1 8
  bash scripts/benchmark_clip_soup.sh ViT-B/32 "$dataset" 16 1 8

done




