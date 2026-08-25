"""
scGen decoder adapter: latent z -> log-normalized expression.

scGen is a Gaussian VAE trained on log-normalized X, so its decoder maps z
directly to a reconstructed log-norm expression profile (no library size, no
condition needed). Loads the scGen model saved by semantic_study
(models[/<dataset>]_scgen).

MUST run in the scGen env (/scratch/yuchen.yan/envs/sem_scgen_py39). Written for
scgen 2.1.0 / scvi-tools 0.16.4; the generative call there is
module.generative(z) -> {"px": <Normal or tensor>}. If your version differs,
adjust the marked line.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
SEM = C.SEM

from decoders.base import Decoder  # noqa: E402


class SCGENDecoder(Decoder):
    def __init__(self, model_dir=None, adata_path=None):
        import scanpy as sc
        import scgen
        import torch
        self.torch = torch
        adata = sc.read_h5ad(adata_path or SEM.TRAIN_DATA_PATH)   # scGen used X = log-norm
        self.model = scgen.SCGEN.load(model_dir or SEM.models_dir("scgen"), adata=adata)
        self.model.module.eval()

    def decode(self, z):
        torch = self.torch
        m = self.model.module
        dev = next(m.parameters()).device
        zt = torch.as_tensor(np.asarray(z), dtype=torch.float32, device=dev)
        with torch.no_grad():
            gen = m.generative(zt)                     # VERIFY: scgen 2.1.0 generative(z)
            px = gen["px"] if isinstance(gen, dict) else gen
            if isinstance(px, torch.Tensor):           # scGen returns the reconstruction tensor
                rec = px
            elif hasattr(px, "loc"):                   # a Normal distribution
                rec = px.loc
            else:                                      # distribution with .mean property
                mm = px.mean
                rec = mm() if callable(mm) else mm
        return np.asarray(rec.detach().cpu().numpy(), dtype=np.float32)   # log-norm expression
