"""
Conditional MLP denoiser for latent vectors (not images -> MLP, not U-Net).

adaLN-Zero conditioning (DiT-style): a conditioning vector y = time_emb +
cond_emb modulates each residual MLP block via (shift, scale, gate) with the
gate/output layers zero-initialised, so the net starts near identity -> stable
training. Predicts either noise eps (DDPM) or velocity v (flow matching); the
output shape is the latent dim either way.
"""
import math

import torch
import torch.nn as nn


def timestep_embedding(t, dim, max_period=10000.0):
    """Sinusoidal embedding of continuous t in [0,1]. Returns [N, dim]."""
    t = t.float() * 1000.0
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    a = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class AdaLNBlock(nn.Module):
    def __init__(self, hidden, mlp_ratio=4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_ratio * hidden), nn.SiLU(),
            nn.Linear(mlp_ratio * hidden, hidden))
        self.ada = nn.Linear(hidden, 3 * hidden)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)  # adaLN-Zero

    def forward(self, h, y):
        shift, scale, gate = self.ada(y).chunk(3, dim=-1)
        return h + gate * self.mlp(self.norm(h) * (1 + scale) + shift)


class MLPDenoiser(nn.Module):
    def __init__(self, latent_dim, hidden=256, depth=6, time_dim=None):
        super().__init__()
        time_dim = time_dim or hidden
        self.hidden = hidden
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.time_dim = time_dim
        self.in_proj = nn.Linear(latent_dim, hidden)
        self.blocks = nn.ModuleList([AdaLNBlock(hidden) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(hidden, elementwise_affine=False)
        self.ada_out = nn.Linear(hidden, 2 * hidden)
        self.out = nn.Linear(hidden, latent_dim)
        for m in (self.ada_out, self.out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, z_t, t, cond_emb):
        """z_t [N,d], t [N] in [0,1], cond_emb [N,hidden] -> prediction [N,d]."""
        y = self.time_mlp(timestep_embedding(t, self.time_dim)) + cond_emb
        h = self.in_proj(z_t)
        for blk in self.blocks:
            h = blk(h, y)
        shift, scale = self.ada_out(y).chunk(2, dim=-1)
        h = self.norm_out(h) * (1 + scale) + shift
        return self.out(h)
