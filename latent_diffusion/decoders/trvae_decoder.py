"""
trVAE decoder adapter: latent z -> expression profile.

trVAE is a CONDITIONAL VAE that integrates OVER a condition (batch/plate). Its
latent is condition-invariant, so decoding needs a *reference* condition to
decode into (default: the first condition seen in training). The scArches
decoder returns the reconstruction (NB rate if trained recon_loss='nb', i.e. an
expression profile; log-norm mean if 'mse').

MUST run in the trVAE env (/scratch/yuchen.yan/envs/sem_trvae). Written for
scArches 0.6.1; the decoder call `tm.decoder(z, c_onehot)` and the
condition_encoder mapping are version-specific -> VERIFY-marked below.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
SEM = C.SEM

from decoders.base import Decoder  # noqa: E402


class TRVAEDecoder(Decoder):
    def __init__(self, model_dir=None, adata_path=None, ref_condition=None):
        import scanpy as sc
        import torch
        from scarches.models import TRVAE
        self.torch = torch
        adata = sc.read_h5ad(adata_path or SEM.TRAIN_DATA_PATH)
        # checkpoint was saved on GPU; this eval runs CPU-only -> map to CPU so
        # torch.load doesn't try (and fail) to deserialize onto a CUDA device.
        map_loc = None if torch.cuda.is_available() else torch.device("cpu")
        self.model = TRVAE.load(model_dir or SEM.models_dir("trvae"), adata, map_location=map_loc)
        self.tm = self.model.model                       # underlying torch module
        self.tm.eval()
        # condition label -> index
        self.cond_enc = getattr(self.model, "condition_encoder", None) \
            or {c: i for i, c in enumerate(self.model.conditions_)}
        self.n_cond = len(self.cond_enc)
        self.ref = ref_condition or list(self.cond_enc)[0]

    def decode(self, z):
        torch = self.torch
        dev = next(self.tm.parameters()).device
        zt = torch.as_tensor(np.asarray(z), dtype=torch.float32, device=dev)
        c = torch.zeros(zt.shape[0], self.n_cond, device=dev)
        c[:, self.cond_enc[self.ref]] = 1.0
        with torch.no_grad():
            out = self.tm.decoder(zt, c)                 # VERIFY: scArches 0.6.1 decoder(z, c)
            recon = out[0] if isinstance(out, (tuple, list)) else out
        r = np.asarray(recon.detach().cpu().numpy(), dtype=np.float32)
        # nb-trained decoder outputs a count-scale rate -> log1p to match X_expr
        # (log-norm). If trained recon_loss='mse' (already log-norm), pass --raw.
        return np.log1p(np.clip(r, 0, None)).astype(np.float32)
