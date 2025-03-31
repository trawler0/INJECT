bash scripts/benchmark_clip_soup.sh ViT-B/32 2 .5 10
bash scripts/benchmark_clip_soup.sh ViT-B/32 4 .5 10
bash scripts/benchmark_clip_soup.sh ViT-B/32 8 .5 10
bash scripts/benchmark_clip_soup.sh ViT-B/32 16 .5 10

bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 2 .5 10
bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 4 .5 10
bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 8 .5 10
bash scripts/benchmark_dinov2_soup.sh dinov2_vits14 16 .5 10

bash scripts/robustness_dinov2_soup_imagenet.sh dinov2_vits14 1 10
bash scripts/robustness_clip_soup_imagenet.sh ViT-B/32 1 10

python main_dinov2_soup.py imagenet dinov2_vitl14_reg 16 --epochs 8 --epoch-multiplier 1 --experiment dinov2_heavy --n-runs 10 --val-frequency 1
