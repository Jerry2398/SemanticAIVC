"""Config loader: YAML + dotted CLI overrides (e.g. --set langevin.steps=40)."""
import os
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "config.yaml")


class Cfg(dict):
    """dict with attribute access, recursively."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k)
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k, v):
        self[k] = v


def _cast(s):
    for f in (int, float):
        try:
            return f(s)
        except ValueError:
            pass
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    return s


def load(path=None, overrides=()):
    d = yaml.safe_load(open(path or DEFAULT))
    for ov in overrides or ():
        key, val = ov.split("=", 1)
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = _cast(val)
    return Cfg(d)


def add_cfg_args(ap):
    ap.add_argument("--config", default=DEFAULT)
    ap.add_argument("--set", nargs="*", default=[], metavar="k.k=v",
                    help="dotted config overrides, e.g. --set langevin.steps=40 train.max_steps=50000")
    return ap
