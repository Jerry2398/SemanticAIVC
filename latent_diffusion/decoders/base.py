"""Decoder-adapter interface: generated latent z (N,d) -> gene expression.

CONTRACT: decode() returns LOG-NORMALIZED expression (N, n_genes), in the same
space as semantic_study's obsm['X_expr'] (normalize_total(1e4)+log1p), so
predicted and real expression are directly comparable in evaluate_expression.py.
scGen already reconstructs log-norm; scVI/trVAE adapters log-transform their
count-scale output to match.
"""
import numpy as np


class Decoder:
    def decode(self, z: np.ndarray) -> np.ndarray:  # -> (N, n_genes), log-normalized
        raise NotImplementedError
