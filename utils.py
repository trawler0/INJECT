import clip
from torch import nn
import torch
import numpy as np
import mlflow
from torchvision import transforms as T

__all__ = [
    "Backbone",
    "CachedDataset",
    "log_metrics",
    "DEFAULT_TRANSFORMS"
]

DEFAULT_IMAGE_SIZE = 448  # for dinov2 models, will automatically be re-resized if clip is used
# these have proven to work well for training, augmentations are very important for these purposes
DEFAULT_TRANSFORMS = T.Compose([
            T.RandomResizedCrop(DEFAULT_IMAGE_SIZE, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.4)
        ])
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class Backbone(nn.Module):
    def __init__(self, name, size=DEFAULT_IMAGE_SIZE):
        super().__init__()
        if name.startswith("dinov2"):
            self.model_type = "dinov2"
            self.model = torch.hub.load('facebookresearch/dinov2', name)
            self.preprocess = T.Compose([
                T.Resize((size, size), interpolation=3),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)  # default imagenet values, these are actually used in DINOv2
            ])
        else:
            self.model_type = "clip"
            self.model, self.preprocess = clip.load(name)
            self.model = self.model.float()

    def forward(self, x):
        if self.model_type == "dinov2":
            x = self.model.forward_features(x)
            patches = x["x_norm_patchtokens"]
            x = x["x_norm_clstoken"]
            return x
        else:
            return self._clip_transformer(x)

    def _clip_transformer(self, x):
        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        patches = x

        x = self.model.visual.ln_post(x[:, 0, :])

        if self.model.visual.proj is not None:
            x = x @ self.model.visual.proj

        return x

# cache the validation datasets to avoid repeated computation of the frozen features
class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, file_name):
        data = np.load(file_name)
        self.embeddings = data["embeddings"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def log_metrics(results, test_flags):
    for i, result in enumerate(results):
        for key, value in result.items():
            mlflow.log_metric(key, value)

if __name__ == "__main__":
    backbone = Backbone("ViT-B/32").cpu()
    print(backbone)
    x = torch.randn(1, 3, 224, 224)
    y = backbone(x)
    print(y)