import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def mlp(sizes, act=nn.GELU, layer_norm=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i+1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class SoftmaxActor(nn.Module):
    def __init__(self, obs_dim: int, n_assets: int, hidden=(256, 256), temperature: float = 1.0):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_assets])
        self.temperature = temperature

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.net(obs)
        z = logits / max(self.temperature, 1e-6)
        weights = torch.softmax(z, dim=-1)
        return weights


class Critic(nn.Module):
    def __init__(self, obs_dim: int, n_assets: int, hidden=(256, 256)):
        super().__init__()
        self.q = mlp([obs_dim + n_assets, *hidden, 1], act=nn.GELU)

    def forward(self, obs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, weights], dim=-1)
        q = self.q(x)
        return q.squeeze(-1)


class TwinCritic(nn.Module):
    def __init__(self, obs_dim: int, n_assets: int, hidden=(256, 256)):
        super().__init__()
        self.q1 = Critic(obs_dim, n_assets, hidden)
        self.q2 = Critic(obs_dim, n_assets, hidden)

    def forward(self, obs: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, weights), self.q2(obs, weights)
