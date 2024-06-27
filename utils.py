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

    def __init__(self, embeddings_student, embeddings_teacher):
        self.embeddings_student = embeddings_student
        self.embeddings_teacher = embeddings_teacher


    def __len__(self):
        return len(self.embeddings_student)

    def __getitem__(self, idx):
        return self.embeddings_student[idx], self.embeddings_teacher[idx]