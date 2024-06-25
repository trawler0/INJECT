import clip
from torch import nn
import torch
import numpy as np
import mlflow

class CLIPBackbone(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip.load(clip_model)[0].float()

    def forward(self, x):
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


class DistillDataset(torch.utils.data.Dataset):
    def __init__(self, ds, augmentation, preprocess_student, preprocess_teacher):
        self.ds = ds
        self.augmentation = augmentation
        self.preprocess_student = preprocess_student
        self.preprocess_teacher = preprocess_teacher

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, _ = self.ds[idx]
        x = self.augmentation(x)
        x_student = self.preprocess_student(x)
        x_teacher = self.preprocess_teacher(x)

        return x_student, x_teacher