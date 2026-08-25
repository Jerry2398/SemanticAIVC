"""
Simplest scGen-style Gaussian VAE for HVG expression.

encoder: x (log-norm HVG) -> (mu, logvar) ; z = mu + eps*exp(.5*logvar)
decoder: z -> x_hat (log-norm HVG, linear output -> Gaussian likelihood)
loss   : reconstruction (MSE, summed over genes) + beta * KL(q(z|x) || N(0,I))

Latent dim is a hyperparameter (--latent_dim). Kept deliberately minimal.
"""
import torch
import torch.nn as nn


class SimpleVAE(nn.Module):
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
        self.fc_out = nn.Linear(hidden, n_genes)

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_lv(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.fc_out(self.dec(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss(x, x_hat, mu, logvar, beta=1.0):
        recon = ((x_hat - x) ** 2).sum(dim=1).mean()                       # per-cell SSE, avg batch
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
        return recon + beta * kl, recon, kl
