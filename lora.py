import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import math
from typing import Callable, Optional


__all__ = [
    'DinoV2Attn', 'DinoV2MLP', 'DinoV2LoraAttn', 'LoraMLP', 'lora_dinov2', 'get_lora_params'
]


class DinoV2Attn(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DinoV2MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DinoV2LoraAttn(nn.Module):

    def __init__(self, attn: DinoV2Attn, alpha, rank):
        super().__init__()
        self.attn = attn

        self.alpha = alpha
        self.rank = rank
        self.lora_in_A = nn.Parameter(torch.zeros(rank, attn.qkv.weight.size(1)))
        self.lora_in_B = nn.Parameter(torch.zeros(3 * attn.qkv.weight.size(1), rank))

        self.lora_out_A = nn.Parameter(torch.zeros(rank, attn.qkv.weight.size(1)))
        self.lora_out_B = nn.Parameter(torch.zeros(attn.qkv.weight.size(1), rank))

        nn.init.kaiming_uniform_(self.lora_in_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_out_A, a=math.sqrt(5))

    def in_proj_weight(self):
        lora_weight = (self.lora_in_B @ self.lora_in_A) * self.alpha / self.rank
        return lora_weight

    def out_proj_weight(self):
        lora_weight = (self.lora_out_B @ self.lora_out_A) * self.alpha / self.rank
        return lora_weight

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.attn.qkv(x) + F.linear(x, self.in_proj_weight())
        qkv = qkv.reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.attn.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.attn.proj(x) + F.linear(x, self.out_proj_weight())
        x = self.attn.proj_drop(x)
        return x


class LoraMLP(nn.Module):
    def __init__(
        self,
        mlp: DinoV2MLP,
        alpha,
        rank
    ) -> None:
        super().__init__()
        self.mlp = mlp

        self.alpha = alpha
        self.rank = rank

        self.lora_in_A = nn.Parameter(torch.zeros(rank, mlp.fc1.weight.size(1)))
        self.lora_in_B = nn.Parameter(torch.zeros(mlp.fc1.weight.size(0), rank))

        self.lora_out_A = nn.Parameter(torch.zeros(rank, mlp.fc2.weight.size(1)))
        self.lora_out_B = nn.Parameter(torch.zeros(mlp.fc2.weight.size(0), rank))


        nn.init.kaiming_uniform_(self.lora_in_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_out_A, a=math.sqrt(5))

    def in_proj_weight(self):
        lora_weight = (self.lora_in_B @ self.lora_in_A) * self.alpha / self.rank
        return lora_weight

    def out_proj_weight(self):
        lora_weight = (self.lora_out_B @ self.lora_out_A) * self.alpha / self.rank
        return lora_weight

    def forward(self, x):
        x = self.mlp.fc1(x) + F.linear(x, self.in_proj_weight())
        x = self.mlp.act(x)
        x = self.mlp.drop(x)
        x = self.mlp.fc2(x) + F.linear(x, self.out_proj_weight())
        x = self.mlp.drop(x)
        return x

def lora_dinov2(model, alpha, rank, strategy='attn+mlp'):
    for layer in model.blocks:
        attn = layer.attn
        if strategy == 'attn+mlp':
            layer.attn = DinoV2LoraAttn(attn, alpha, rank)
            layer.mlp = LoraMLP(layer.mlp, alpha, rank)
        elif strategy == 'attn':
            layer.attn = DinoV2LoraAttn(attn, alpha, rank)
        elif strategy == 'mlp':
            layer.mlp = LoraMLP(layer.mlp, alpha, rank)
        elif strategy == 'none':
            pass
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    return model

def get_lora_params(model):
    params = []
    for layer in model.blocks:
        if hasattr(layer, 'attn'):
            if isinstance(layer.attn, DinoV2LoraAttn):
                params.append(layer.attn.lora_in_A)
                params.append(layer.attn.lora_in_B)
                params.append(layer.attn.lora_out_A)
                params.append(layer.attn.lora_out_B)

        if hasattr(layer, 'mlp'):
            if isinstance(layer.mlp, LoraMLP):
                params.append(layer.mlp.lora_in_A)
                params.append(layer.mlp.lora_in_B)
                params.append(layer.mlp.lora_out_A)
                params.append(layer.mlp.lora_out_B)
    print(f"Number of lora parameters: {sum(p.numel() for p in params)}")
    return params

if __name__ == "__main__":

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    print(model.blocks[-1].norm1.weight.size(0))
    params = sum(p.numel() for p in model.parameters()) / 1e6
    model = lora_dinov2(model, 1., 4, strategy='attn+mlp')
    new_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Number of dinov2 parameters: {params}, number of lora parameters: {new_params - params}")
    x = torch.randn(1, 3, 224, 224)
    model(x)

    x = torch.rand(128, 3, 224, 224).cuda()
    model.cuda()
    lora_params = get_lora_params(model)
    optimizer = torch.optim.SGD(lora_params, lr=0.01)
    for _ in range(10):
        optimizer.zero_grad()
        model(x)
        loss = torch.abs(model(x)).sum()
        loss.backward()
        optimizer.step()
        print(loss.item())
