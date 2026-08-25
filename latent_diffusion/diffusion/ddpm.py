"""
Gaussian diffusion (DDPM) with eps-prediction + DDIM sampling, on latent vectors.

The model interface is a closure `predict(z_t, t_norm) -> eps_hat` where t_norm
is the timestep normalized to [0,1]; conditioning / classifier-free guidance
live inside that closure (built in train.py / sample.py), so this class is pure
diffusion math.
"""
import numpy as np
import torch


def _cosine_alpha_bar(num_steps, s=0.008):
    steps = np.arange(num_steps + 1)
    f = np.cos(((steps / num_steps + s) / (1 + s)) * np.pi / 2) ** 2
    return f / f[0]                                  # alpha_bar[0..T], =1 at 0


class GaussianDiffusion:
    def __init__(self, num_steps=1000, device="cpu"):
        self.T = num_steps
        ab = _cosine_alpha_bar(num_steps)
        self.alpha_bar = torch.tensor(ab[1:], dtype=torch.float32, device=device)  # [T]
        self.device = device

    def _ab(self, t_idx):
        return self.alpha_bar[t_idx]

    def training_loss(self, predict, z0):
        N = z0.shape[0]
        t = torch.randint(0, self.T, (N,), device=z0.device)
        ab = self._ab(t).unsqueeze(-1)
        noise = torch.randn_like(z0)
        z_t = ab.sqrt() * z0 + (1 - ab).sqrt() * noise
        eps = predict(z_t, t.float() / self.T)
        return ((eps - noise) ** 2).mean()

    @torch.no_grad()
    def ddim_sample(self, predict, shape, steps=50, eta=0.0):
        """Deterministic (eta=0) DDIM. Returns z0 [shape]."""
        dev = self.alpha_bar.device
        z = torch.randn(shape, device=dev)
        ts = torch.linspace(self.T - 1, 0, steps, device=dev).long()
        for i in range(steps):
            t = ts[i]
            ab_t = self.alpha_bar[t]
            eps = predict(z, torch.full((shape[0],), float(t) / self.T, device=dev))
            z0 = (z - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            if i == steps - 1:
                z = z0
            else:
                ab_next = self.alpha_bar[ts[i + 1]]
                sigma = eta * ((1 - ab_next) / (1 - ab_t) * (1 - ab_t / ab_next)).clamp(min=0).sqrt()
                z = ab_next.sqrt() * z0 + (1 - ab_next - sigma ** 2).clamp(min=0).sqrt() * eps
                if eta > 0:
                    z = z + sigma * torch.randn_like(z)
        return z
