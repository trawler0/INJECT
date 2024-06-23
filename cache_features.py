import clip
import torch
import numpy as np
import argparse
import os
import templates as temps
from data import DATASETS
from tqdm import tqdm

@torch.no_grad()
def cache_prompts(file_name, clip_model, templates, classes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(clip_model)
    model.to(device)
    model.eval()
    embeddings = []
    for cls in tqdm(classes):
        temp_plus_cls = [t.format(cls) for t in templates]
        text = clip.tokenize(temp_plus_cls).to(device)
        text_features = model.encode_text(text)  # L x D
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        embeddings.append(text_features.cpu().numpy())
    embeddings = np.stack(embeddings)
    np.save(file_name, embeddings)

@torch.no_grad()
def cache_dataset(file_name, clip_model, dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model)
    model.to(device)
    model.eval()
    embeddings = []
    labels = []
    dataset.transform = preprocess
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4)
    for batch in tqdm(loader):
        image, y = batch
        image = image.to(device)
        image_features = model.encode_image(image)
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        embeddings.append(image_features.cpu().numpy())
        labels.append(y.numpy())
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    np.savez(file_name, embeddings=embeddings, labels=labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    try:
        default_root = os.getenv("DATA_ROOT")
    except:
        raise ValueError("Please set the environment variable DATA_ROOT to the folder where all datasets are stored")
    CACHED_FEATURES = "cached-features"

    parser.add_argument("dataset_identifier", type=str)
    parser.add_argument("clip_model", type=str)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--cache-dir", default=CACHED_FEATURES, type=str)
    parser.add_argument("--root", default=default_root, type=str)

    parser.add_argument("--cache-prompts", action="store_true", default=False)
    parser.add_argument("--cache-dataset", action="store_true", default=False)

    # must be set if cache-prompts is true
    parser.add_argument("--templates", default=None, type=str)

    args = parser.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    cache_dir = args.cache_dir
    cache_dir = os.path.join(cache_dir, args.clip_model)
    os.makedirs(cache_dir, exist_ok=True)

    if args.cache_prompts:
        assert args.templates is not None
        templates = getattr(temps, args.templates)
        file_name = os.path.join(cache_dir, f"{args.dataset_identifier}-{args.templates}.npy")
        ds = DATASETS.get(args.dataset_identifier)(args.root, "train")
        classes = [cls for _, cls in ds.idx_to_class.items()]
        cache_prompts(file_name, args.clip_model, templates, classes)

    if args.cache_dataset:
        assert args.split is not None
        file_name = os.path.join(cache_dir, f"{args.dataset_identifier}-{args.split}-features.npz")
        ds = DATASETS.get(args.dataset_identifier)(args.root, args.split)
        cache_dataset(file_name, args.clip_model, ds)










