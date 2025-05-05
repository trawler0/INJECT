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
    "default_transforms"
]

DEFAULT_IMAGE_SIZE = 350  # for dinov2 models, will automatically be re-resized if clip is used
DINOV2_TRAIN_SIZE = 350  # for dinov2 models, this is the size used for training

# these have proven to work well for training, augmentations are very important for these purposes
def default_transforms(strength):
    min_scale = strength * 0.08 + (1 - strength) * 0.4
    return T.Compose([
        T.RandomResizedCrop(DINOV2_TRAIN_SIZE, scale=(min_scale, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        ], p=0.8 * strength),
        T.RandomGrayscale(p=0.2 * strength),
        T.RandomApply([
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.2 * strength),
        T.RandomSolarize(threshold=128.0, p=0.2 * strength),
    ])


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Backbone(nn.Module):
    def __init__(self, name, size=DEFAULT_IMAGE_SIZE, force_amp=True):
        super().__init__()
        if name.startswith("dinov2"):
            self.model_type = "dinov2"
            self.model = torch.hub.load('facebookresearch/dinov2', name)
            self.preprocess = T.Compose([
                T.Resize((size * 8) // 7, interpolation=3),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD)  # default imagenet values, these are actually used in DINOv2
            ])
            self.force_amp = force_amp
        else:
            self.model_type = "clip"
            self.model, self.preprocess = clip.load(name)
            self.model = self.model.float()
            self.force_amp = force_amp

    def forward(self, x):
        if self.force_amp:
            with torch.autocast("cuda"):
                if self.model_type == "dinov2":
                    x = self.model.forward_features(x)
                    x = x["x_norm_clstoken"]
                    return x
                else:
                    return self._clip_transformer(x)
        else:
            if self.model_type == "dinov2":
                x = self.model.forward_features(x)
                x = x["x_norm_clstoken"]
                return x
            else:
                return self._clip_transformer(x)

    def _clip_transformer(self, x):
        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                         device=x.device),
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


def log_metrics(results):
    for i, result in enumerate(results):
        for key, value in result.items():
            mlflow.log_metric(key, value)


if __name__ == "__main__":
    backbone = Backbone("ViT-B/32").cpu()
    print(backbone)
    x = torch.randn(1, 3, 224, 224)
    y = backbone(x)
    print(y)
