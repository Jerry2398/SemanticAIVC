"""
Negative-Binomial VAE for Tahoe HVG counts (scVI/scANVI-style likelihood).

Same MLP backbone as the Gaussian `SimpleVAE` (encoder/decoder, latent_dim,
hidden, depth) -- ONLY the observation model changes:

  encoder : x_lognorm -> (mu, logvar) ; z = mu + eps*exp(.5*logvar)   (input still log-norm)
  decoder : z -> gene logits -> softmax = rho (proportions, sum_g rho=1)
  counts  : x_g ~ NB(mean = library_size * rho_g, dispersion = theta_g)
            theta_g is a per-gene learnable dispersion (shared across cells, scVI default).
            library_size = observed total HVG counts of the cell (conditioned on, not modelled).
  loss    : -E[log NB(x | rate, theta)] + beta * KL(q(z|x) || N(0,I))

`decode(z)` deliberately returns the reconstruction in LOG-NORM space
(log1p(1e4 * rho)) so the exact same eval helpers (common.collect_recon /
reconstruction_metrics / generative_mean_match) apply -> metrics are directly
comparable to the Gaussian VAE. Training uses `decode_rho`/`nb_loss` on raw counts.
"""
import torch
import torch.nn as nn


class NBVAE(nn.Module):
    def __init__(self, n_genes, latent_dim=32, hidden=512, depth=2, dropout=0.1):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim

        def block(i, o):
            return [nn.Linear(i, o), nn.LayerNorm(o), nn.SiLU(), nn.Dropout(dropout)]

        enc = block(n_genes, hidden)
        for _ in range(depth - 1):
            enc += block(hidden, hidden)
        self.enc = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_lv = nn.Linear(hidden, latent_dim)

        dec = block(latent_dim, hidden)
        for _ in range(depth - 1):
            dec += block(hidden, hidden)
        self.dec = nn.Sequential(*dec)
        self.fc_out = nn.Linear(hidden, n_genes)              # gene logits -> softmax = rho
        self.log_theta = nn.Parameter(torch.zeros(n_genes))   # per-gene dispersion (theta = exp)

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_lv(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode_rho(self, z):
        return torch.softmax(self.fc_out(self.dec(z)), dim=-1)  # [B,G], rows sum to 1

    def decode(self, z, target_sum=1e4):
        """Reconstruction in LOG-NORM space (for eval reuse with the Gaussian model)."""
        return torch.log1p(self.decode_rho(z) * target_sum)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode_rho(z), mu, logvar

    @staticmethod
    def nb_loss(x_counts, rho, log_theta, lib, mu, logvar, beta=1.0, eps=1e-8):
        """NB NLL (scVI log_nb_positive parameterization) + beta*KL, per-cell summed over genes."""
        theta = torch.exp(log_theta)                            # [G]
        rate = lib.unsqueeze(1) * rho                           # [B,G] NB mean = lib * rho
        log_theta_rate = torch.log(theta + rate + eps)
        ll = (theta * (torch.log(theta + eps) - log_theta_rate)
              + x_counts * (torch.log(rate + eps) - log_theta_rate)
              + torch.lgamma(x_counts + theta)
              - torch.lgamma(theta)
              - torch.lgamma(x_counts + 1.0))
        nll = -ll.sum(dim=1).mean()                             # per-cell, avg batch
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
        return nll + beta * kl, nll, kl
