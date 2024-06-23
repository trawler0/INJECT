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
DEFAULT_LABEL_SMOOTHING = 0.0
DEFAULT_WEIGHING_STRATEGY = "linear"
DEFAULT_LOGIT_SCALE = 10.

__all__ = ["INJECT"]

class OrthogonalLinear(nn.Module):
    def __init__(self, features: int):
        """
        Uses the exponential map of the Lie algebra of skew-symmetric matrices into the Lie group of orthogonal matrices
        to create an orthogonal linear layer
        :param features: dimension of the embedding space
        """
        super(OrthogonalLinear, self).__init__()
        self.features = features
        # initialize zero which exponentiates to identity
        self.skew_symmetric = nn.Parameter(torch.rand(features, features))

    def forward(self, input: torch.Tensor, inverse: bool = False):
        # Ensure skew-symmetry during the forward pass, this is a slight over-parameterization
        A = self.skew_symmetric - self.skew_symmetric.T
        # when using inverse, harness that exp(-A) = exp(A)^-1
        if inverse:
            A = -A
        weight_orthogonal = torch.matrix_exp(A)
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
            weighing_strategy: str = DEFAULT_WEIGHING_STRATEGY,
            label_smoothing: float =DEFAULT_LABEL_SMOOTHING,
            lr: float = DEFAULT_LR,
            ema_decay: float = DEFAULT_EMA_DECAY,
            logit_scale: float = DEFAULT_LOGIT_SCALE,
            test_flags: List[str] = None
    ):
        """
        INJECT model, minimal CLIP domain adaption model

        :param cache_features: prompt templates/feature used for KNN
        :param label_smoothing: label smoothing
        :param lr: learning rate for AdamW
        :param ema_decay: exponential moving average decay
        """
        super().__init__()
        self.backbone = backbone
        text_features = torch.tensor(text_features).float()
        self.register_buffer("text_features", text_features)

        N_class, L, D = text_features.shape

        self.se = Rose(D)

        self.alpha = nn.Parameter(torch.zeros(N_class, L))
        self.label_smoothing = label_smoothing
        self.lr = lr

        self.weighing_strategy = weighing_strategy
        # the ones listed here work well, but others might work too, the popular (in SSL) rule exp(sim / T) does not work
        # nonlinear strategy might be more powerful if the cache is large
        if weighing_strategy == "logistic":
            self.delta_prime = torch.nn.Parameter(torch.zeros(1))
            self.b = torch.nn.Parameter(torch.zeros(1))
        elif weighing_strategy == "linear":
            pass
        else:
            raise ValueError("Weighing strategy {} not supported".format(weighing_strategy))

        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale)))
        self.test_flags = test_flags

        self.ema = deepcopy(self)
        self.ema_decay = ema_decay

    def _weighing(self, sims):
        if self.weighing_strategy == "logistic":
            delta = torch.exp(self.delta_prime) + 1
            sims = 2 / (1 + torch.exp(-delta * sims + self.b))
            self.log("delta", delta, on_step=False, on_epoch=True, prog_bar=True)
            self.log("b", self.b, on_step=False, on_epoch=True, prog_bar=True)
        elif self.weighing_strategy == "linear":
            pass
        return sims

    def _forward(self, image_features):
        image_features = image_features.float()
        image_features = self.se(image_features)
        image_features = F.normalize(image_features, p=2, dim=-1)  # B x D

        weights = self.text_features  # N_class x L x D
        weights = F.normalize(weights, p=2, dim=-1)

        sims = weights @ image_features.T  # N_class x L x B
        sims = sims.permute(2, 0, 1)  # B x N_class x L
        sims = self._weighing(sims)

        t = torch.exp(self.alpha).unsqueeze(0)  # 1 x N_class x L
        t = t / t.sum(dim=-1, keepdim=True)
        sims = sims * t  # B x N_class x L

        logit_scale = torch.exp(self.logit_scale)
        logits = sims.sum(dim=-1) * logit_scale  # B x N_class

        self.log("logit_scale", logit_scale, on_step=False, on_epoch=True, prog_bar=True)

        return logits


    def forward(self, image):
        image_features = self.backbone(image)
        return self._forward(image_features)

    def training_step(self, batch, batch_idx):
        sleep(0.005)  # if using encoded features, need this to prevent computer from freezing
        torch.cuda.empty_cache()
        image_features, target = batch
        with torch.no_grad():
            self.backbone.eval()  # it is very important to run CLIP in eval if resnets are used (batch-norm), otherwise it won't work
            if len(image_features.shape) == 4:  # check if features are already encoded
                image_features = self.backbone(image_features)


        logits = self._forward(image_features)
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
        torch.cuda.empty_cache()
        image, target = batch
        if len(image.shape) == 4:  # check if features are already encoded
            self.backbone.eval()
            image_features = self.backbone(image)
        else:
            image_features = image
        current_alpha = self.se.alpha.data  # to set back

        for alpha in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
            flag = "val" if self.test_flags is None else self.test_flags[dataloader_idx]
            self.se.alpha.data = torch.tensor(alpha)
            logits = self._forward(image_features)
            if flag == "imagenet-a":
                logits = logits[:, IMAGENET_A_IDX]
            if flag == "imagenet-r":
                logits = logits[:, IMAGENET_R_IDX]
            acc = (logits.argmax(1) == target).float().mean()
            self.log("{}_acc_{}".format(flag, alpha), acc, on_epoch=True, prog_bar=True, on_step=False)

        self.se.alpha.data = current_alpha
        return acc

    def configure_optimizers(self):
        params = [
            {"params": self.se.parameters()},
            {"params": self.alpha, "lr": self.lr * 10},
            {"params": self.logit_scale, "lr": self.lr * 10}
        ]
        if self.weighing_strategy == "logistic":
            params.extend([
                {"params": self.delta_prime, "lr": 5 * self.lr},
                {"params": self.b, "lr": 5 * self.lr}
            ])
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