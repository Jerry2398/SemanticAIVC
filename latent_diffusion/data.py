"""
Data plumbing for the LDM: load VAE latents + encode conditions.

- load_latents(vae, split): reads the semantic_study embedding .h5ad, returns
  (z [N,d] float32, obs DataFrame). Latent dim d is inferred here.
- LatentScaler: standardize z with TRAIN stats (mean/std); invert before decode.
  (Standard practice for LDMs -- keeps the diffusion target ~N(0,1).)
- CondCodec: fit categorical vocabularies + continuous transforms on TRAIN obs,
  apply to any split. Encodes obs -> {name: LongTensor|FloatTensor}. Reserves a
  null id per categorical (last index) for classifier-free guidance. Serializable.
"""
import json

import numpy as np
import scanpy as sc


def load_latents(emb_path):
    ad = sc.read_h5ad(emb_path)
    X = ad.X
    z = (X.toarray() if hasattr(X, "toarray") else np.asarray(X)).astype(np.float32)
    return z, ad.obs.copy()


class LatentScaler:
    def __init__(self, mu, sd):
        self.mu = np.asarray(mu, np.float32)
        self.sd = np.asarray(sd, np.float32)

    @classmethod
    def fit(cls, z):
        return cls(z.mean(0), z.std(0) + 1e-6)

    def transform(self, z):
        return (z - self.mu) / self.sd

    def inverse(self, z):
        return z * self.sd + self.mu

    def to_dict(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["mu"], np.float32), np.asarray(d["sd"], np.float32))


class CondCodec:
    """Encodes the conditioning obs columns. Categorical -> int id in
    [0, cardinality); a separate 'null' id == cardinality is used for CFG.
    Continuous -> optional log10(x+1) then standardize (train stats)."""

    def __init__(self, spec, log10_names):
        self.spec = spec                       # list of (name, kind, obs_col)
        self.log10_names = set(log10_names)
        self.cats = {}                         # name -> list of category strings
        self.cont = {}                         # name -> (mean, std)

    def fit(self, obs):
        for name, kind, col in self.spec:
            if kind == "categorical":
                self.cats[name] = sorted(obs[col].astype(str).unique().tolist())
            else:
                v = obs[col].astype(float).values
                if name in self.log10_names:
                    v = np.log10(v + 1.0)
                self.cont[name] = (float(v.mean()), float(v.std() + 1e-6))
        return self

    def cardinality(self, name):
        return len(self.cats[name])            # embedding table needs +1 for null

    def encode(self, obs):
        """obs -> dict {name: np.int64[N] (categorical) | np.float32[N] (continuous)}."""
        out = {}
        for name, kind, col in self.spec:
            if kind == "categorical":
                idx = {c: i for i, c in enumerate(self.cats[name])}
                null = len(self.cats[name])
                out[name] = np.array([idx.get(str(v), null) for v in obs[col]], dtype=np.int64)
            else:
                v = obs[col].astype(float).values
                if name in self.log10_names:
                    v = np.log10(v + 1.0)
                m, s = self.cont[name]
                out[name] = ((v - m) / s).astype(np.float32)
        return out

    # --- for sampling: build a condition dict for explicit requested values ---
    def encode_value(self, name, value, n):
        kind = dict((s[0], s[1]) for s in self.spec)[name]
        if kind == "categorical":
            null = len(self.cats[name])
            idx = {c: i for i, c in enumerate(self.cats[name])}.get(str(value), null)
            return np.full(n, idx, dtype=np.int64)
        v = float(value)
        if name in self.log10_names:
            v = np.log10(v + 1.0)
        m, s = self.cont[name]
        return np.full(n, (v - m) / s, dtype=np.float32)

    def to_dict(self):
        return {"spec": self.spec, "log10_names": sorted(self.log10_names),
                "cats": self.cats, "cont": self.cont}

    @classmethod
    def from_dict(cls, d):
        c = cls([tuple(s) for s in d["spec"]], d["log10_names"])
        c.cats, c.cont = d["cats"], {k: tuple(v) for k, v in d["cont"].items()}
        return c
