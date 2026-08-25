"""Langevin negative sampling from p_alpha(z_a | z_b, a)  (paper Eq. 13-15).

    z^{k+1} = z^k - (eta/2) [ grad_za E_alpha(z^k, z_b, a) + z^k ] + sqrt(eta) * eps

(+z^k is -grad log p_0 for p_0 = N(0, I)). The chain is initialised from ANOTHER cell's
positive z_a in the same batch -- a representation that is valid for some other
(z_b, a) but is a negative for this one -- which is a far more informative starting
point than white noise. Energy parameters are never updated here; gradients are taken
w.r.t. z_a only.
"""
import torch


@torch.enable_grad()
def sample_negative(model, z_b, a_e, init, steps=20, step_size=0.1,
                    noise_scale=1.0, clamp=10.0):
    """z_b / a_e / init are treated as fixed conditions. Returns z_a^(K), detached."""
    z_b, a_e = z_b.detach(), a_e.detach()
    net = model.energy_net
    was_training = net.training
    net.eval()                                        # no dropout/BN noise inside the chain
    frozen = [p.requires_grad for p in net.parameters()]
    for p in net.parameters():
        p.requires_grad_(False)

    z = init.detach().clone()
    for _ in range(steps):
        z = z.detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(model.energy(z, z_b, a_e).sum(), z)
        z = z - 0.5 * step_size * (grad + z)
        if noise_scale:
            z = z + noise_scale * (step_size ** 0.5) * torch.randn_like(z)
        if clamp:
            z = z.clamp(-clamp, clamp)

    for p, r in zip(net.parameters(), frozen):
        p.requires_grad_(r)
    if was_training:
        net.train()
    return z.detach()
