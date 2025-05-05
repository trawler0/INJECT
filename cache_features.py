import clip
import torch
import numpy as np
import argparse
import os
import templates as temps
from data import DATASETS
from tqdm import tqdm
from utils import Backbone
import random

@torch.no_grad()
def get_prompts_clip(clip_model, templates, classes):
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
    return embeddings

@torch.no_grad()
def get_prompts_dinov2(dinov2_model, n_aug, ds):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Backbone(dinov2_model)
    model.to(device)
    model.eval()
    embeddings = {j: [] for j in range(len(ds.classes))}
    idxs = {j: [] for j in range(len(ds.classes))}
    ds.transform = model.preprocess
    for j in tqdm(range(len(ds))):
        _, y = ds[j]
        x = []
        for _ in range(n_aug):
            im = ds[j][0]
            x.append(im)
        x = torch.stack(x)
        x = x.to(device)
        image_features = model(x)
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        embeddings[y].append(image_features.cpu().numpy())
        idx = np.ones(len(image_features)) * j
        idxs[y].append(idx)

    embeddings = [np.concatenate(embeddings[j]) for j in range(len(ds.classes))]
    idxs = [np.concatenate(idxs[j]) for j in range(len(ds.classes))]
    M = max(x.shape[0] for x in embeddings)
    for j in range(len(ds.classes)):
        if embeddings[j].shape[0] < M:
            n = M - embeddings[j].shape[0]
            choice = np.random.randint(0, embeddings[j].shape[0], n)
            embeddings[j] = np.concatenate([embeddings[j], embeddings[j][choice]])
            idxs[j] = np.concatenate([idxs[j], idxs[j][choice]])
    embeddings = np.stack(embeddings)
    idxs = np.stack(idxs)
    return embeddings, idxs

@torch.cuda.amp.autocast()
@torch.inference_mode()
def cache_dataset(file_name, model, dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Backbone(model)
    model.to(device)
    model.eval()

    embeddings = []
    labels = []
    dataset.transform = model.preprocess
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=8)
    for batch in tqdm(loader):
        image, y = batch
        image = image.to(device)

        image_features = model(image)
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        embeddings.append(image_features.cpu().numpy())
        labels.append(y.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    np.savez(file_name, embeddings=embeddings, labels=labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    default_root = os.getenv("DATA_ROOT")
    if default_root is None:
        default_root = os.path.join("/home/marco/data")
    CACHED_FEATURES = "cached-features"

    parser.add_argument("dataset_identifier", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("--n-augs", type=int, default=4)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--cache-dir", default=CACHED_FEATURES, type=str)
    parser.add_argument("--root", default=default_root, type=str)
    parser.add_argument("--n-shot", type=int, default=-1)

    parser.add_argument("--cache-prompts", action="store_true", default=False)
    parser.add_argument("--cache-dataset", action="store_true", default=False)

    # must be set if cache-prompts or cache-distillation-prompts is true
    parser.add_argument("--templates", default=None, type=str)


    args = parser.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    cache_dir = args.cache_dir
    cache_dir = os.path.join(cache_dir, args.model)
    os.makedirs(cache_dir, exist_ok=True)

    if args.cache_prompts:
        if args.model.startswith("dinov2"):
            assert args.n_augs > 0
            n_shot = args.n_shot
            file_name = os.path.join(cache_dir, f"{args.dataset_identifier}-{n_shot}.npz")
            ds = DATASETS.get(args.dataset_identifier)(args.root, "train", n_shot=n_shot)
            emb, idxs = get_prompts_dinov2(args.model, args.n_augs, ds)
            np.savez(file_name, emb=emb, idxs=idxs)
        else:  # clip
            assert args.templates is not None
            templates = getattr(temps, args.templates)
            file_name = os.path.join(cache_dir, f"{args.dataset_identifier}-{args.templates}.npy")
            ds = DATASETS.get(args.dataset_identifier)(args.root, "train", n_shot=args.n_shot)
            classes = [cls for _, cls in ds.idx_to_class.items()]
            embeddings = get_prompts_clip(args.model, templates, classes)
            np.save(file_name, embeddings)

    if args.cache_dataset:
        assert args.split is not None
        file_name = os.path.join(cache_dir, f"{args.dataset_identifier}-{args.split}-features.npz")
        ds = DATASETS.get(args.dataset_identifier)(args.root, args.split, n_shot=args.n_shot)
        cache_dataset(file_name, args.model, ds)











