"""LDMNet = condition embedder + MLP denoiser, bundled for easy save/load."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn

from conditioning import ConditionEmbedder
from models.denoiser import MLPDenoiser


class LDMNet(nn.Module):
    def __init__(self, latent_dim, codec, hidden=256, depth=6):
        super().__init__()
        self.embedder = ConditionEmbedder(codec, hidden)
        self.denoiser = MLPDenoiser(latent_dim, hidden=hidden, depth=depth)

    def embed(self, cond, drop=None):
        return self.embedder(cond, drop)

    def denoise(self, z_t, t, cond_emb):
        return self.denoiser(z_t, t, cond_emb)
