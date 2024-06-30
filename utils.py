import clip
from torch import nn
import torch
import numpy as np
import mlflow
from torchvision import transforms as T

class Backbone(nn.Module):
    def __init__(self, name, size=224):
        super().__init__()
        if name.startswith("dinov2"):
            self.model_type = "dinov2"
            self.model = torch.hub.load('facebookresearch/dinov2', name)
            self.preprocess = T.Compose([
                T.Resize((size, size), interpolation=3),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # default imagenet values
            ])
        else:  # clip
            self.model_type = "clip"
            self.model, self.preprocess = clip.load(name)
            self.model = self.model.float()

    def forward(self, x):
        if self.model_type == "dinov2":
            return self.model.forward_features(x)["x_norm_clstoken"]
        else:
            return self.model.encode_image(x)


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
            mlflow.log_metric(f"{test_flags[i]}_{key}", value)


DEFAULT_TRANSFORMS = T.Compose([
            T.RandomResizedCrop(224, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=0.4)
        ])


if __name__ == "__main__":
    model = Backbone("dinov2_vits14")
    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)