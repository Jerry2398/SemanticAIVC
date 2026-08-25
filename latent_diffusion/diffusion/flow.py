"""
Rectified flow / conditional flow matching on latent vectors.

Interpolation:  z_t = (1 - t) * z0 + t * noise,  t ~ U(0,1)
  -> t=0 is data, t=1 is standard noise.
Target velocity:  v = d z_t / dt = noise - z0   (constant along the path).
The model predicts v; sampling integrates the ODE dz = v dt from t=1 (noise)
back to t=0 (data) with an Euler solver.

Same `predict(z_t, t_norm)` closure interface as ddpm.GaussianDiffusion, so
train/sample code is shared; only method='flow' vs 'ddpm' switches the object.
"""
import torch


class RectifiedFlow:
    def __init__(self, device="cpu"):
        self.device = device

    def training_loss(self, predict, z0):
        N = z0.shape[0]
        t = torch.rand(N, device=z0.device)
        noise = torch.randn_like(z0)
        z_t = (1 - t).unsqueeze(-1) * z0 + t.unsqueeze(-1) * noise
        v = predict(z_t, t)
        target = noise - z0
        return ((v - target) ** 2).mean()

    @torch.no_grad()
    def sample(self, predict, shape, steps=50, device=None):
        dev = device or self.device
        z = torch.randn(shape, device=dev)          # t = 1 (noise)
        times = torch.linspace(1.0, 0.0, steps + 1, device=dev)
        for k in range(steps):
            t_cur, t_next = times[k], times[k + 1]
            v = predict(z, torch.full((shape[0],), float(t_cur), device=dev))
            z = z + v * (t_next - t_cur)            # Euler step (t_next - t_cur < 0)
        return z
