"""
Tahoe-VAE decoder adapter: latent z -> log-normalized HVG expression.

Wraps the three Tahoe-trained VAEs (Gaussian SimpleVAE, NBVAE, scGen-aligned
SCGenLikeVAE). All three expose `.decode(z)` returning reconstruction in LOG-NORM
space (log1p(1e4*.)), matching semantic_study's obsm['X_expr'] -- so predicted
and real expression are directly comparable in evaluate_expression.py.

Runs in the squidiff env (torch 2.5) since these are our own checkpoints; no
legacy env needed. `name` selects the checkpoint:
    tahoe_vae        -> models/tahoe_vae/vae.pt        (Gaussian, SimpleVAE)
    tahoe_vae_nb     -> models/tahoe_vae_nb/vae.pt     (NBVAE)
    tahoe_vae_scgen  -> models/tahoe_vae_scgen/vae.pt  (SCGenLikeVAE)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VAE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "vae")
sys.path.insert(0, _VAE_DIR)

from decoders.base import Decoder  # noqa: E402

MODELS_DIR = "/scratch/Projects/CFP-03/CFP03-CF-130/yuchen.yan/models"


def _build_model(ckpt):
    """Instantiate the right VAE class from the checkpoint's 'likelihood' tag."""
    import torch  # noqa: F401
    a = ckpt["args"]
    lik = ckpt.get("likelihood", "gaussian")
    ng = ckpt["n_genes"]
    if lik == "nb":
        from model_nb import NBVAE
        return NBVAE(ng, a["latent_dim"], a["hidden"], a["depth"])
    if lik == "gaussian_scgen":
        from model_scgen import SCGenLikeVAE
        return SCGenLikeVAE(ng, a["latent_dim"], a["hidden"], a["depth"], a.get("dropout", 0.2))
    from model import SimpleVAE
    return SimpleVAE(ng, a["latent_dim"], a["hidden"], a["depth"])


class TahoeVAEDecoder(Decoder):
    def __init__(self, name="tahoe_vae", ckpt_path=None):
        import torch
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_path = ckpt_path or os.path.join(MODELS_DIR, name, "vae.pt")
        ck = torch.load(ckpt_path, map_location=self.dev)
        self.model = _build_model(ck).to(self.dev)
        self.model.load_state_dict(ck["model"])
        self.model.eval()

    def decode(self, z):
        torch = self.torch
        zt = torch.as_tensor(np.asarray(z), dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            x = self.model.decode(zt)              # log-norm HVG reconstruction
        return np.asarray(x.detach().cpu().numpy(), dtype=np.float32)
