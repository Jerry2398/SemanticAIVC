"""
Streaming mini-batch Dataset over Tahoe-100M for out-of-core VAE training.

Only the current shard is held in memory; cells are yielded as dense 5000-HVG
count minibatches (+ obs conditions). Never loads the full dataset. Works with a
torch DataLoader (batch_size=None, since batching is done here) and multiple
workers (shards are partitioned across workers).

  from tahoe_dataset import TahoeStream
  ds = TahoeStream(split="train", batch_size=512)
  for counts, cond in torch.utils.data.DataLoader(ds, batch_size=None, num_workers=4):
      ...  # counts: FloatTensor [B, 5000] raw counts; cond: dict col -> list[str]
"""
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

import tahoe_common as T


class TahoeStream(IterableDataset):
    def __init__(self, split="train", batch_size=512, shuffle=True,
                 shard_shuffle_seed=0, cond_cols=("cell_line_id", "drug", "moa-fine"),
                 val_every=20, max_shards=None):
        super().__init__()
        self.paths = T.list_shards(split, val_every=val_every)
        if max_shards:
            self.paths = self.paths[:max_shards]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = shard_shuffle_seed
        self.cond_cols = list(cond_cols)
        gm = T.gene_metadata()
        self.n_genes = T.n_vocab(gm)
        self.hvg_tokens = T.load_hvg()["token_ids"]          # (5000,) token ids
        self.epoch = 0

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        paths = list(self.paths)
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(paths)
        wi = get_worker_info()
        if wi is not None:                                    # partition shards across workers
            paths = paths[wi.id::wi.num_workers]
        rng = np.random.default_rng(self.seed + self.epoch + (wi.id if wi else 0) + 1)
        for p in paths:
            X, obs = T.shard_to_csr(p, self.n_genes, want_obs=True)
            n = X.shape[0]
            order = rng.permutation(n) if self.shuffle else np.arange(n)
            for s in range(0, n, self.batch_size):
                b = order[s:s + self.batch_size]
                counts = X[b][:, self.hvg_tokens].toarray().astype(np.float32)  # [B, 5000]
                cond = {c: obs[c].values[b].astype(str).tolist() for c in self.cond_cols if c in obs}
                yield torch.from_numpy(counts), cond
