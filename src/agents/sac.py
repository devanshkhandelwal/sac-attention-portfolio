import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ..nets.actor_critic import SoftmaxActor, TwinCritic


class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buf.append((s, a, r, s2, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(np.array(a), dtype=torch.float32),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.array(s2), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


@dataclass
class SACConfig:
    obs_dim: int
    n_assets: int
    actor_hidden: Tuple[int, int] = (256, 256)
    critic_hidden: Tuple[int, int] = (256, 256)
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 1_000_000
    target_entropy: Optional[float] = None  # default to -log(n_assets)
    temperature_init: float = 1.0
    exploration_noise: float = 0.1  # Add exploration noise to actions


class SACAgent:
    def __init__(self, cfg: SACConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = SoftmaxActor(cfg.obs_dim, cfg.n_assets, cfg.actor_hidden).to(self.device)
        self.critic = TwinCritic(cfg.obs_dim, cfg.n_assets, cfg.critic_hidden).to(self.device)
        self.critic_tgt = TwinCritic(cfg.obs_dim, cfg.n_assets, cfg.critic_hidden).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.lr)

        self.replay = ReplayBuffer(cfg.buffer_size)

        # Entropy temperature alpha (log-param for positivity)
        target_ent = cfg.target_entropy if cfg.target_entropy is not None else -float(np.log(cfg.n_assets))
        self.target_entropy = torch.tensor(target_ent, device=self.device)
        self.log_alpha = torch.tensor(np.log(0.1), device=self.device, requires_grad=True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            w = self.actor(o)
        self.actor.train()
        w = w.squeeze(0).cpu().numpy()
        
        # Add exploration noise to prevent mode collapse
        if not greedy and self.cfg.exploration_noise > 0:
            # Add noise to logits before softmax to encourage exploration
            noise = np.random.normal(0, self.cfg.exploration_noise, w.shape)
            w_noisy = w + noise
            # Renormalize to ensure valid weights
            w_noisy = np.maximum(w_noisy, 0)
            w = w_noisy / (w_noisy.sum() + 1e-12)
        
        return w

    def update(self):
        if len(self.replay) < self.cfg.batch_size:
            return None

        s, a, r, s2, d = self.replay.sample(self.cfg.batch_size)
        s = s.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)

        # Critic target
        with torch.no_grad():
            a2 = self.actor(s2)
            q1_t, q2_t = self.critic_tgt(s2, a2)
            q_min = torch.min(q1_t, q2_t)
            v = q_min + self.alpha * (-torch.sum(a2 * torch.log(a2 + 1e-12), dim=-1))
            y = r + (1.0 - d) * self.cfg.gamma * v

        # Critic loss
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Actor loss (maximize Q + entropy) -> minimize negative
        a_pi = self.actor(s)
        q1_pi, q2_pi = self.critic(s, a_pi)
        q_min_pi = torch.min(q1_pi, q2_pi)
        ent = -torch.sum(a_pi * torch.log(a_pi + 1e-12), dim=-1)
        actor_loss = -(q_min_pi + self.alpha * ent).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # Alpha temperature update
        alpha_loss = (-(self.log_alpha * (ent.detach() - self.target_entropy))).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Target update
        with torch.no_grad():
            for p_tgt, p in zip(self.critic_tgt.parameters(), self.critic.parameters()):
                p_tgt.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
        }

    def state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_tgt': self.critic_tgt.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'alpha_opt': self.alpha_opt.state_dict(),
            'log_alpha': self.log_alpha,
            'cfg': self.cfg
        }

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_tgt.load_state_dict(state_dict['critic_tgt'])
        self.actor_opt.load_state_dict(state_dict['actor_opt'])
        self.critic_opt.load_state_dict(state_dict['critic_opt'])
        self.alpha_opt.load_state_dict(state_dict['alpha_opt'])
        self.log_alpha = state_dict['log_alpha']

    def eval(self):
        self.actor.eval()
        self.critic.eval()
        self.critic_tgt.eval()

    def train(self):
        self.actor.train()
        self.critic.train()
        self.critic_tgt.train()
