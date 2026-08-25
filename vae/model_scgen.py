"""
scGen-ALIGNED Gaussian VAE.

Same NN *dimensions* as our SimpleVAE (latent 32 / hidden 512 / depth 2 -- kept
fixed by request), but every other architectural/loss setting is matched to the
real scGen source (scvi-tools `SCGENVAE` / `SCGEN`):

  - normalization : BatchNorm  (scGen use_batch_norm="both")   [was LayerNorm]
  - activation    : LeakyReLU  (scGen activation_fn)           [was SiLU]
  - dropout       : 0.2        (scGen wrapper default)         [was 0.1]
  - observation   : linear output, Gaussian unit variance      [same]
  - loss          : 0.5*SSE + 0.5*kl_weight*KL, kl_weight=5e-5  [was recon + 1.0*KL]
                    (exactly scGen SCGENVAE.loss / get_reconstruction_loss)

Data, step budget, optimizer and lr are deliberately kept identical to our other
runs so the comparison isolates the scGen settings.
"""
import torch
import torch.nn as nn


class SCGenLikeVAE(nn.Module):
    def __init__(self, n_genes, latent_dim=32, hidden=512, depth=2, dropout=0.2):
        super().__init__()
        self.n_genes = n_genes
        self.latent_dim = latent_dim

        def block(i, o):
            return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.LeakyReLU(), nn.Dropout(dropout)]

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
    def loss(x, x_hat, mu, logvar, kl_weight=5e-5):
        """scGen loss: (0.5*SSE + 0.5*kl_weight*KL).mean(). Returns (loss, recon, kl)."""
        recon = ((x_hat - x) ** 2).sum(dim=1)                              # per-cell SSE
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1))    # per-cell KL
        loss = (0.5 * recon + 0.5 * kl_weight * kl).mean()
        return loss, recon.mean(), kl.mean()
