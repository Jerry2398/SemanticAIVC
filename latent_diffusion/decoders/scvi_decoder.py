"""
scVI decoder adapter: latent z -> gene-expression profile.

Loads the scVI model saved by semantic_study (models[/<dataset>]_scvi, via
--save_model in extract_scvi.py) with its training adata, then runs the scVI
generative decoder on arbitrary latent vectors. Returns px_scale -- the
library-normalized mean expression profile per cell (sums to ~1 over genes),
which is the natural library-independent output for a *generated* cell.

MUST run in the scVI env (/scratch/yuchen.yan/envs/sem_scvi). The exact decoder
call is scvi-version-specific (written for scvi-tools 1.3.x); if your version
differs, adjust the marked line.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
SEM = C.SEM

from decoders.base import Decoder  # noqa: E402


class SCVIDecoder(Decoder):
    def __init__(self, model_dir=None, adata_path=None, target_library=1e4):
        import scanpy as sc
        import scvi
        import torch
        self.torch = torch
        model_dir = model_dir or SEM.models_dir("scvi")
        adata = sc.read_h5ad(adata_path or SEM.TRAIN_DATA_PATH)
        if "counts" in adata.layers:            # scVI was set up on the counts layer
            adata.X = adata.layers["counts"].copy()
        self.model = scvi.model.SCVI.load(model_dir, adata=adata)
        self.model.module.eval()
        self.log_library = float(np.log(target_library))

    def decode(self, z):
        torch = self.torch
        m = self.model.module
        dev = next(m.parameters()).device
        zt = torch.as_tensor(np.asarray(z), dtype=torch.float32, device=dev)
        lib = torch.full((zt.shape[0], 1), self.log_library, device=dev)
        with torch.no_grad():
            # scvi-tools 1.3.x: decoder(dispersion, z, library, *cat_list) ->
            # (px_scale, px_r, px_rate, px_dropout). Even with batch_key=None scVI
            # registers ONE batch category, so the decoder's FCLayers still expect
            # a batch index in cat_list -- pass batch 0 for every generated cell
            # (omitting it raises "nb. categorical args ... doesn't match init").
            batch = torch.zeros((zt.shape[0], 1), dtype=torch.long, device=dev)
            px_scale = m.decoder(m.dispersion, zt, lib, batch)[0]  # normalized profile (sums~1)
        # -> log-norm space to match X_expr: log1p(px_scale * target_library)
        return np.log1p(px_scale.cpu().numpy() * float(np.exp(self.log_library))).astype(np.float32)
