model="$1"
epoch_multiplier="$2"
n_runs="$3"

python main_clip_soup.py "standford_cars" "$model" 16 AUGMENTED_STANDFORD_CARS_TEMPLATES  --epochs 40 --batch-size 32 --epoch-multiplier "$epoch_multiplier" --experiment "compare" --n-runs "$n_runs" --save-example-feats
