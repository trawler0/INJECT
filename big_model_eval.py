import torch
import torch.nn as nn
import numpy as np
from model import Adapter, Soup
from data import DATASETS
from templates import IMAGENET_A_IDX, IMAGENET_R_IDX
import clip

model = clip.load("ViT-L/14", device="cpu")[0]  # Load the CLIP model

out = {}
method = "clip"
if method == "clip":
    models = ["ViT-L/14@336px", "ViT-L/14", "ViT-B/16", "ViT-B/32"]
    r = 0.6
else:
    models = ["dinov2_vitg14_reg", "dinov2_vitl14_reg", "dinov2_vitb14_reg", "dinov2_vits14"]
    r = 1
for model in models:
    out[model] = {}
    adapters = torch.load(f"checkpoints/{model.lower().replace('/', '_')}_4.pth")
    if method == "clip":
        p = np.load(f"cached-features/{model}/imagenet-CLIP_IMAGENET_TEMPLATES.npy")
    else:
        prompts = np.load(f"cached-features/{model}/imagenet-4.npz")
        p, idxs = prompts["emb"], prompts["idxs"]

    p = torch.tensor(p).float()
    p = nn.functional.normalize(p, dim=-1)
    p = p.mean(1)
    p = torch.nn.functional.normalize(p, dim=-1)

    IDX = list(range(1000))
    for name, id in [("val", IDX), ("v2", IDX), ("sketch", IDX), ("imagenet-a", IMAGENET_A_IDX), ("imagenet-r", IMAGENET_R_IDX)]:
        print("Evaluating on", name)
        feats = f"cached-features/{model}/imagenet-{name}-features.npz"
        feats = np.load(feats)
        X, y = feats["embeddings"], feats["labels"]
        X = torch.tensor(X).float()
        X = nn.functional.normalize(X, dim=-1)
        y = torch.tensor(y).long()

        a1 = adapters[0](X) * r
        a2 = adapters[1](X) * r
        a3 = adapters[2](X) * r
        a4 = adapters[3](X) * r

        soup = .25 * (a1 + a2 + a3 + a4)

        a1 = torch.nn.functional.normalize(X + a1, dim=-1)
        a2 = torch.nn.functional.normalize(X + a2, dim=-1)
        a3 = torch.nn.functional.normalize(X + a3, dim=-1)
        a4 = torch.nn.functional.normalize(X + a4, dim=-1)
        soup = torch.nn.functional.normalize(X + soup, dim=-1)

        zero = X @ p.T
        a1 = a1 @ p.T
        a2 = a2 @ p.T
        a3 = a3 @ p.T
        a4 = a4 @ p.T
        soup = soup @ p.T

        zero = zero[:, id]
        a1 = a1[:, id]
        a2 = a2[:, id]
        a3 = a3[:, id]
        a4 = a4[:, id]
        soup = soup[:, id]

        acc_zero = (zero.argmax(-1) == y).float().mean().item()
        acc_a1 = (a1.argmax(-1) == y).float().mean().item()
        acc_a2 = (a2.argmax(-1) == y).float().mean().item()
        acc_a3 = (a3.argmax(-1) == y).float().mean().item()
        acc_a4 = (a4.argmax(-1) == y).float().mean().item()
        acc_soup = (soup.argmax(-1) == y).float().mean().item()

        out[model][name] = {
            "prototypical": "%.2f" % (acc_zero * 100),
            "adapter": "%.2f" % (.25 * (acc_a1 + acc_a2 + acc_a3 + acc_a4) * 100),
            "soup": "%.2f" % (acc_soup * 100)
        }
print(out)
