"""
Condition embedding for the denoiser, with classifier-free guidance (CFG).

Each conditioning variable -> an H-dim embedding; they are summed into one
condition vector. Categorical vars use an embedding table (with a reserved
`null` row = cardinality, so unseen/held-out categories are valid). Continuous
vars (e.g. dose) go through a small MLP.

CFG: with prob `p_uncond` during training the whole condition vector is replaced
by a single learned `null` embedding (joint drop) -> gives a clean unconditional
path. At sampling we run cond & uncond and combine with a guidance scale.
"""
import torch
import torch.nn as nn


class ConditionEmbedder(nn.Module):
    def __init__(self, codec, hidden):
        super().__init__()
        self.spec = codec.spec
        self.embs = nn.ModuleDict()
        self.conts = nn.ModuleDict()
        for name, kind, _ in codec.spec:
            if kind == "categorical":
                self.embs[name] = nn.Embedding(codec.cardinality(name) + 1, hidden)
            else:
                self.conts[name] = nn.Sequential(
                    nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.null = nn.Parameter(torch.zeros(hidden))

    def forward(self, cond, drop=None):
        """cond: {name: LongTensor[N] | FloatTensor[N]}; drop: BoolTensor[N] or None."""
        s = None
        for name, kind, _ in self.spec:
            if kind == "categorical":
                e = self.embs[name](cond[name])
            else:
                e = self.conts[name](cond[name].unsqueeze(-1))
            s = e if s is None else s + e
        if drop is not None:
            s = torch.where(drop.unsqueeze(-1), self.null.unsqueeze(0).expand_as(s), s)
        return s

    def null_like(self, n, device):
        return self.null.unsqueeze(0).expand(n, -1).to(device)
