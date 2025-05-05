import torch
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from copy import deepcopy
from time import sleep
import numpy as np
from typing import *
import math
from templates import IMAGENET_A_IDX, IMAGENET_R_IDX

DEFAULT_LR = 1e-3
DEFAULT_EMA_DECAY = 0.997
DEFAULT_SQUEEZE_RATIO = 0.25
DEFAULT_ALPHA = 1.0
DEFAULT_LABEL_SMOOTHING = 0.02
DEFAULT_LOGIT_SCALE = 10.


# reimplementation
class Adapter(LightningModule):
    def __init__(
        self,
        reduction: int,
        backbone: nn.Module,
        text_features: Union[np.ndarray, torch.Tensor],
        idxs: Union[np.ndarray, torch.Tensor],
        label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
        lr: float = DEFAULT_LR,
        weight_decay: float = 0.01,
        ema_decay: float = DEFAULT_EMA_DECAY,
        logit_scale: float = DEFAULT_LOGIT_SCALE,
        test_flags: List[str] = None,
    ):
        super().__init__()
        self.backbone = backbone
        text_features = torch.tensor(text_features).float()
        self.register_buffer("text_features", text_features)
        self.register_buffer("idxs", torch.tensor(idxs))

        N_class, L, D = text_features.shape

        self.adapter_layer = nn.Sequential(
            nn.Linear(D, D//reduction),
            nn.GELU(),
            nn.Linear(D//reduction, D),
        )

        self.label_smoothing = label_smoothing
        self.lr = lr
        self.weight_decay = weight_decay

        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale)))
        self.test_flags = test_flags

        self.ema = deepcopy(self)
        self.ema_decay = ema_decay

    def embeddings(self, image_features, ratio=1.):
        image_features = image_features.float()
        image_features = F.normalize(image_features, p=2, dim=-1)  # B x D

        image_features = image_features + ratio * self.adapter_layer(image_features)
        image_features = F.normalize(image_features, p=2, dim=-1)  # B x D

        return image_features

    def forward_clip_adapter(self, image_features, ratio=1., idx=None):

        image_features = self.embeddings(image_features, ratio=ratio)

        weights = self.text_features  # N_class x L x D
        weights = F.normalize(weights, p=2, dim=-1)
        if idx is not None:
            mask = (idx.view(-1, 1, 1) != self.idxs.unsqueeze(0)).float()  # B x N_class x L
            mask = mask.unsqueeze(-1)  # B x N_class x L x 1
            weights = (weights.unsqueeze(0) * mask).sum(2)  # B x N_class x D
        else:
            weights = weights.unsqueeze(0).expand(image_features.shape[0], -1, -1, -1)  # B x N_class x L x D
            weights = weights.sum(2)  # B x N_class x D

        weights = F.normalize(weights, p=2, dim=-1)  # B x N_class x D

        logits = torch.einsum("bd,bnd->bn", image_features, weights)  # B x N_class
        logit_scale = torch.exp(self.logit_scale)
        logits = logits * logit_scale

        return logits


    def forward(self, image, ratio=.9):
        image_features = self.backbone(image)
        return self.forward_clip_adapter(image_features, ratio=ratio)

    def training_step(self, batch, batch_idx):
        sleep(0.005)  # if using encoded features, need this to prevent computer from freezing
        torch.cuda.empty_cache()
        (image_features, target), idx = batch
        with torch.no_grad():
            self.backbone.eval()  # it is very important to run CLIP in eval if resnets are used (batch-norm), otherwise it won't work
            if len(image_features.shape) == 4:  # check if features are already encoded
                image_features = self.backbone(image_features)


        logits = self.forward_clip_adapter(image_features, idx=idx)
        loss = F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
        acc = (logits.argmax(1) == target).float().mean() * 100


        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=image_features.shape[0])
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=image_features.shape[0])

        estd = self.ema.state_dict()
        mstd = self.state_dict()
        for k, v in self.ema.state_dict().items():
            estd[k] = self.ema_decay * v + (1 - self.ema_decay) * mstd[k]
        self.ema.load_state_dict(estd)


        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            print()
        image, target = batch
        if len(image.shape) == 4:  # check if features are already encoded
            self.backbone.eval()
            image_features = self.backbone(image)
        else:
            image_features = image
        for ratio in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
            flag = "val" if self.test_flags is None else self.test_flags[dataloader_idx]
            logits = self.forward_clip_adapter(image_features, ratio=ratio)
            if flag == "imagenet-a":
                logits = logits[:, IMAGENET_A_IDX]
            if flag == "imagenet-r":
                logits = logits[:, IMAGENET_R_IDX]
            acc = (logits.argmax(1) == target).float().mean()
            self.log("{}_acc_{}".format(flag, ratio), acc, on_epoch=True, prog_bar=True, on_step=False, batch_size=image.shape[0])

        return acc

    def configure_optimizers(self):
        params = [
            {"params": self.adapter_layer.parameters()},
            {"params": self.logit_scale, "lr": self.lr * 10}
        ]

        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

class Soup(LightningModule):

    def __init__(self, models: List[Adapter], test_flags=None, flag="uniform"):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.test_flags = test_flags
        self.flag = flag

    def forward_clip_adapter(self, image_features, ratio=1.):

        x = []
        for model in self.models:
            x.append(model.embeddings(image_features, ratio=ratio))
        image_features = torch.stack(x, dim=0).mean(0)
        image_features = F.normalize(image_features, p=2, dim=-1)

        weights = self.models[0].text_features  # N_class x L x D
        weights = F.normalize(weights, p=2, dim=-1)
        weights = F.normalize(weights.mean(1), p=2, dim=-1)  # N_class x D

        logits = image_features @ weights.T * 50  # B x N_class

        return logits


    def forward(self, image, ratio=.9):
        image_features = self.backbone(image)
        return self.forward_clip_adapter(image_features, ratio=ratio)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            print()
        image, target = batch
        if len(image.shape) == 4:  # check if features are already encoded
            image_features = self.models[0].backbone(image)
        else:
            image_features = image
        for ratio in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
            flag = "val" if self.test_flags is None else self.test_flags[dataloader_idx]
            logits = self.forward_clip_adapter(image_features, ratio=ratio)
            if flag == "imagenet-a":
                logits = logits[:, IMAGENET_A_IDX]
            if flag == "imagenet-r":
                logits = logits[:, IMAGENET_R_IDX]
            acc = (logits.argmax(1) == target).float().mean()
            self.log(fr"acc_{ratio}_{self.flag}", acc, on_epoch=True, prog_bar=True, on_step=False, batch_size=image.shape[0])

        return acc


class BaselineEvaluator(LightningModule):

    def __init__(self, backbone, feats, labels, tau=.1, K=10, test_flags=None):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("feats", F.normalize(feats, p=2, dim=-1))
        self.register_buffer("labels", labels)
        self.tau = tau
        self.K = K
        self.test_flags = test_flags


    def training_step(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("This model is only for evaluation")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        image, y = batch
        if len(image.shape) == 4:
            self.backbone.eval()
            image_features = self.backbone(image)
            image_features = F.normalize(image_features, p=2, dim=-1)
        else:
            image_features = image

        sim = torch.exp(image_features @ self.feats.T / self.tau)  # B x N
        sim, idx = sim.topk(self.K, dim=-1)  # B x K
        target = self.labels.unsqueeze(0).expand(sim.shape[0], -1)  # B x N
        target = target.gather(1, idx)  # B x K
        target = F.one_hot(target, num_classes=torch.max(self.labels) + 1).float()  # B x K x C
        sim = sim.unsqueeze(-1)

        flag = "val" if self.test_flags is None else self.test_flags[dataloader_idx]

        logits = (sim * target).sum(1)  # B x C
        if flag == "imagenet-a":
            logits = logits[:, IMAGENET_A_IDX]
        if flag == "imagenet-r":
            logits = logits[:, IMAGENET_R_IDX]
        knn_acc = (logits.argmax(1) == y).float().mean()
        self.log("knn_acc", knn_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=image.shape[0])

        target = F.one_hot(self.labels, num_classes=torch.max(self.labels) + 1).float()
        """feats = self.feats.unsqueeze(-1)
        print(feats.shape, target.shape, image_features.shape)
        proto = (target * feats)
        print(proto.shape)
        proto = F.normalize(proto.sum(0), p=2, dim=1)
        print(proto.shape)"""
        feats = F.normalize(self.feats, p=2, dim=-1)
        proto = torch.einsum("nd,nc->dc", feats, target)
        proto = F.normalize(proto, p=2, dim=0)
        logits = image_features @ proto
        if flag == "imagenet-a":
            logits = logits[:, IMAGENET_A_IDX]
        if flag == "imagenet-r":
            logits = logits[:, IMAGENET_R_IDX]
        proto_acc = (logits.argmax(1) == y).float().mean()
        self.log("proto_acc", proto_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=image.shape[0])



