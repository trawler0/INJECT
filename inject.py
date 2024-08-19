import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from copy import deepcopy
from time import sleep
import numpy as np
from typing import *
import math
from templates import IMAGENET_A_IDX, IMAGENET_R_IDX
import random

DEFAULT_LR = 1e-3
DEFAULT_EMA_DECAY = 0.997
DEFAULT_SQUEEZE_RATIO = 0.25
DEFAULT_ALPHA = 1.0
DEFAULT_LABEL_SMOOTHING = 0.02
DEFAULT_LOGIT_SCALE = 10.
DEFAULT_T = 10.  # this ends up in the identity function


class OrthogonalLinear(nn.Module):
    def __init__(self, features: int):
        """
        Uses the exponential map of the Lie algebra of skew-symmetric matrices into the Lie group of orthogonal matrices
        to create an orthogonal linear layer
        :param features: dimension of the embedding space
        """
        super(OrthogonalLinear, self).__init__()
        self.features = features
        self.param = nn.Parameter(torch.rand(features, features))

    def forward(self, input: torch.Tensor, inverse: bool = False):
        # Ensure skew-symmetry during the forward pass, this is a slight over-parameterization
        A = self.param - self.param.T
        # when using inverse, harness that exp(-A) = exp(A)^-1
        if inverse:
            A = -A
        weight_orthogonal = torch.matrix_exp(A)  # this is not an entry-wise exponential, google matrix exponential as reference
        return F.linear(input, weight_orthogonal)


class Rose(nn.Module):

    def __init__(
            self,
            features: int,
            alpha: float = DEFAULT_ALPHA,
            squeeze_ratio: float = DEFAULT_SQUEEZE_RATIO,
            apply_se=True,
            apply_rotation=True
    ):
        """
        Rose module, first rotates the embedding space, then applies squeeze excitation, and finally rotates back
        :param features: dimension of the input features
        :param alpha: interpolation parameter between identity and the learned rotation
        :param squeeze_ratio: ratio of the squeeze excitation MLP hidden layer size to the input size
        :param apply_se: whether to apply squeeze excitation (will act as identity if False)
        :param apply_rotation: whether to apply the rotation (will act as squeeze-excitation if False)
        """
        super(Rose, self).__init__()
        self.ol = OrthogonalLinear(features)

        # can be used to interpolate between identity and the learned layer
        self.register_buffer('alpha', torch.tensor(alpha))

        self.mlp = nn.Sequential(
            nn.Linear(features, int(features * squeeze_ratio)),
            nn.ReLU(),
            nn.Linear(int(features * squeeze_ratio), features),
        )
        self.sigmoid = nn.Sigmoid()

        self.apply_se = apply_se
        self.apply_rotation = apply_rotation

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 2, "Input shape must be (B, D)"
        # rotate to model universal squeeze excitation
        if self.apply_rotation:
            x = self.ol(x)

        # apply squeeze excitation
        if self.apply_se:
            logits = self.mlp(x)
            sigma = self.sigmoid(logits)
            sigma = (1 - self.alpha) + self.alpha * sigma
            x = x * sigma

        # rotate back into original alignment
        if self.apply_rotation:
            x = self.ol(x, inverse=True)
        return x


class INJECT(LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            text_features: Union[np.ndarray, torch.Tensor],
            label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
            lr: float = DEFAULT_LR,
            ema_decay: float = DEFAULT_EMA_DECAY,
            logit_scale: float = DEFAULT_LOGIT_SCALE,
            test_flags: List[str] = None,
            T: float = DEFAULT_T
    ):
        """
        INJECT model, injects new knowledge into the baseline heuristic of comparing inputs with prompts
        :param backbone: a CLIP or DINOv2 model
        :param text_features: precomputed embeddings of the prompts N_class x L x D (L is the number of prompts for each class)
        :param label_smoothing: label smoothing during training
        :param lr: learning rate
        :param ema_decay: exponential moving average decay, can improve robustness
        :param logit_scale: the learnable logit scale initialization
        :param test_flags: used if imagenet is the dataset, to evaluate on imagenet-a and imagenet-r
        :param T: a rescaling parameter for experimental purposes, does not seem to be useful
        """
        super().__init__()
        self.backbone = backbone
        text_features = torch.tensor(text_features).float()
        self.register_buffer("text_features", text_features)

        print(text_features.shape)
        N_class, L, D = text_features.shape

        self.se = Rose(D)

        self.alpha = nn.Parameter(torch.zeros(N_class, L))
        self.label_smoothing = label_smoothing
        self.lr = lr

        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale)))
        self.test_flags = test_flags

        self.ema = deepcopy(self)
        self.ema_decay = ema_decay
        self.T = T

    def _weighing(self, sims):
        if self.training:
            return sims
        else:
            return self.T * (torch.exp(sims / self.T) - 1)  # renormalization

    def forward_inject(self, image_features, ratio=1.):
        weights = self.text_features  # N_class x L x D
        weights = F.normalize(weights, p=2, dim=-1)

        image_features = image_features.float()
        image_features = F.normalize(image_features, p=2, dim=-1)  # B x D

        original_logits = image_features @ F.normalize(weights.mean(1), p=2, dim=-1).T  # B x N_class

        image_features = self.se(image_features)
        image_features = F.normalize(image_features, p=2, dim=-1)  # B x D

        sims = weights @ image_features.T  # N_class x L x B
        sims = sims.permute(2, 0, 1)  # B x N_class x L
        sims = self._weighing(sims)

        t = torch.softmax(self.alpha, dim=-1).unsqueeze(0)  # 1 x N_class x L
        #t = torch.exp(self.alpha).unsqueeze(0)  # 1 x N_class x L
        #t = t / t.sum(dim=-1, keepdim=True)
        sims = sims * t  # B x N_class x L

        logit_scale = torch.exp(self.logit_scale)
        logits = sims.sum(dim=-1)   # B x N_class

        logits = (ratio * logits + (1 - ratio) * original_logits) * logit_scale

        return logits


    def forward(self, image, ratio=.9):
        image_features = self.backbone(image)
        return self.forward_inject(image_features, ratio=ratio)

    def training_step(self, batch, batch_idx):
        sleep(0.005)  # if using encoded features, need this to prevent computer from freezing
        torch.cuda.empty_cache()
        image_features, target = batch
        with torch.no_grad():
            self.backbone.eval()  # it is very important to run CLIP in eval if resnets are used (batch-norm), otherwise it won't work
            if len(image_features.shape) == 4:  # check if features are already encoded
                image_features = self.backbone(image_features)


        logits = self.forward_inject(image_features)
        loss = F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
        acc = (logits.argmax(1) == target).float().mean() * 100


        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)



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
            logits = self.forward_inject(image_features, ratio=ratio)
            if flag == "imagenet-a":
                logits = logits[:, IMAGENET_A_IDX]
            if flag == "imagenet-r":
                logits = logits[:, IMAGENET_R_IDX]
            acc = (logits.argmax(1) == target).float().mean()
            self.log("{}_acc_{}".format(flag, ratio), acc, on_epoch=True, prog_bar=True, on_step=False)

        return acc

    def configure_optimizers(self):
        params = [
            {"params": self.se.parameters()},
            {"params": self.alpha, "lr": self.lr * 10},
            {"params": self.logit_scale, "lr": self.lr * 10}
        ]
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

# for DINOv2 models we need data samples as prompts, so splitting 50/50 into prompts and training samples is natural.
# Performance is improved by switching roles of the two models and ensembling them.

class INJECTEnsemble(LightningModule):


    def __init__(
            self,
            model1: INJECT,
            model2: INJECT,
    ):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.test_flags = model1.test_flags

    def forward(self, image, ratio=.9):
        logits1 = self.model1(image, ratio=ratio)
        logits2 = self.model2(image, ratio=ratio)
        return (logits1 + logits2) / 2

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        image, target = batch
        if len(image.shape) == 4:  # check if features are already encoded
            self.backbone.eval()
            image_features = self.model1.backbone(image)
        else:
            image_features = image
        for ratio in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
            flag = "val" if self.test_flags is None else self.test_flags[dataloader_idx]
            logits1 = self.model1.forward_inject(image_features, ratio=ratio)
            logits2 = self.model2.forward_inject(image_features, ratio=ratio)
            logits = (logits1 + logits2) / 2
            if flag == "imagenet-a":
                logits = logits[:, IMAGENET_A_IDX]
            if flag == "imagenet-r":
                logits = logits[:, IMAGENET_R_IDX]
            acc = (logits.argmax(1) == target).float().mean()
            self.log("{}_acc_{}".format(flag, ratio), acc, on_epoch=True, prog_bar=True, on_step=False)

        return acc


class INJECTSSL(LightningModule):
    def __init__(
            self,
            backbone: nn.Module,
            text_features: Union[np.ndarray, torch.Tensor],
            label_smoothing: float = DEFAULT_LABEL_SMOOTHING,
            lr: float = DEFAULT_LR,
            ema_decay: float = DEFAULT_EMA_DECAY,
            temperature: float = .1,
            test_flags: List[str] = None,
            T: float = DEFAULT_T
    ):
        """
        INJECT model, injects new knowledge into the baseline heuristic of comparing inputs with prompts
        :param backbone: a CLIP or DINOv2 model
        :param text_features: precomputed embeddings of the prompts N_class x L x D (L is the number of prompts for each class)
        :param label_smoothing: label smoothing during training
        :param lr: learning rate
        :param ema_decay: exponential moving average decay, can improve robustness
        :param logit_scale: the learnable logit scale initialization
        :param test_flags: used if imagenet is the dataset, to evaluate on imagenet-a and imagenet-r
        :param T: a rescaling parameter for experimental purposes, does not seem to be useful
        """
        super().__init__()
        self.backbone = backbone
        text_features = torch.tensor(text_features).float()
        self.register_buffer("text_features", text_features)

        print(text_features.shape)
        N_class, L, D = text_features.shape

        self.se = Rose(D)

        self.alpha = nn.Parameter(torch.zeros(N_class, L))
        self.label_smoothing = label_smoothing
        self.lr = lr

        self.temperature = temperature
        self.test_flags = test_flags

        self.ema = deepcopy(self)
        self.ema_decay = ema_decay
        self.T = T

    def _weighing(self, sims):
        if self.training:
            return sims
        else:
            return self.T * (torch.exp(sims / self.T) - 1)  # renormalization

    def forward_inject(self, image_features, ratio=1.):
        weights = self.text_features  # N_class x L x D
        weights = F.normalize(weights, p=2, dim=-1)

        image_features = image_features.float()
        image_features = F.normalize(image_features, p=2, dim=-1)  # B x D

        original_logits = image_features @ F.normalize(weights.mean(1).T, p=2, dim=-1)  # B x N_class

        image_features = self.se(image_features)
        image_features = F.normalize(image_features, p=2, dim=-1)  # B x D

        sims = weights @ image_features.T  # N_class x L x B
        sims = sims.permute(2, 0, 1)  # B x N_class x L
        sims = self._weighing(sims)

        t = torch.exp(self.alpha).unsqueeze(0)  # 1 x N_class x L
        t = t / t.sum(dim=-1, keepdim=True)
        sims = sims * t  # B x N_class x L

        logits = sims.sum(dim=-1)   # B x N_class

        logits = (ratio * logits + (1 - ratio) * original_logits) / self.temperature
        target = torch.argmax(original_logits, dim=-1)

        return logits, target


    def forward(self, image, ratio=.9):
        image_features = self.backbone(image)
        return self.forward_inject(image_features, ratio=ratio)

    def training_step(self, batch, batch_idx):
        sleep(0.005)  # if using encoded features, need this to prevent computer from freezing
        torch.cuda.empty_cache()
        image_features, y = batch
        with torch.no_grad():
            self.backbone.eval()  # it is very important to run CLIP in eval if resnets are used (batch-norm), otherwise it won't work
            if len(image_features.shape) == 4:  # check if features are already encoded
                image_features = self.backbone(image_features)


        logits, target = self.forward_inject(image_features)
        loss = F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)

        acc = (logits.argmax(1) == y).float().mean() * 100

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)


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
            logits, _ = self.forward_inject(image_features, ratio=ratio)
            if flag == "imagenet-a":
                logits = logits[:, IMAGENET_A_IDX]
            if flag == "imagenet-r":
                logits = logits[:, IMAGENET_R_IDX]
            acc = (logits.argmax(1) == target).float().mean()
            self.log("{}_acc_{}".format(flag, ratio), acc, on_epoch=True, prog_bar=True, on_step=False)

        return acc

    def configure_optimizers(self):
        params = [
            {"params": self.se.parameters()},
            {"params": self.alpha, "lr": self.lr * 10},
        ]
        optimizer = torch.optim.AdamW(params, lr=self.lr)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }