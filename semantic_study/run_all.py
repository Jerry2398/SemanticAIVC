"""
Run the full semantic study and collect every metric into results/summary.json.

Assumes the embedding .h5ad files already exist (run extract/ first, or the
submit script). Each metric is a self-contained module invoked here in-process;
individual metrics can also be run standalone from the command line.

  $ENV run_all.py                 # all metrics
  $ENV run_all.py --only ger ctd  # a subset
"""
import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "decodability"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "organization"))

import config as C

# metric name -> (module path, results json basename)
METRICS = {
    "ger":    ("decodability.ger", "ger"),
    "ctd":    ("decodability.ctd", "ctd"),
    "dd":     ("decodability.dd", "dd"),
    "dose_d": ("decodability.dose_d", "dose_d"),
    "moa_d":  ("decodability.moa_d", "moa_d"),
    "egc":    ("organization.egc", "egc"),
    "lnp":    ("organization.lnp", "lnp"),
    "cog":    ("organization.cog", "cog"),
    "pgc":    ("organization.pgc", "pgc"),
}

# decodability metrics train a probe (need both splits); organization metrics
# only read the val embeddings.
NEEDS_TRAIN = {"ger", "ctd", "dd", "dose_d", "moa_d"}


def main():
    ap = argparse.ArgumentParser()
    # default to the metrics applicable to the selected dataset (config.METRICS);
    # e.g. PBMC drops dose_d/moa_d/pgc (single perturbation, no dose/MOA).
    _default_only = [m for m in C.METRICS if m in METRICS]
    ap.add_argument("--only", nargs="+", choices=list(METRICS), default=_default_only)
    ap.add_argument("--encoder", default="squidiff",
                    help="which encoder's embeddings to evaluate (scvi/scgen/trvae/... "
                         "-- resolved via config.emb_paths). Metrics that read expr/labels "
                         "work unchanged since every extractor writes the same format.")
    args = ap.parse_args()

    train_emb, val_emb = C.emb_paths(args.encoder)
    for p in (train_emb, val_emb):
        if not os.path.exists(p):
            print(f"!!! missing embedding {p} -- run the extractor for '{args.encoder}' first.")
    # namespace results under results/<encoder>/ (read by save_results)
    os.environ["SEMANTIC_ENCODER"] = args.encoder
    out_dir = os.path.join(C.RESULTS_DIR, args.encoder)
    print(f"evaluating encoder='{args.encoder}'\n  train={train_emb}\n  val={val_emb}\n  results -> {out_dir}")

    for name in args.only:
        mod_path, _ = METRICS[name]
        print(f"\n{'='*70}\n>>> {name.upper()}\n{'='*70}")
        # forward the encoder's embedding paths (only the flags each metric has).
        old_argv = sys.argv
        sys.argv = [mod_path, "--val_emb", val_emb]
        if name in NEEDS_TRAIN:
            sys.argv += ["--train_emb", train_emb]
        try:
            mod = importlib.import_module(mod_path)
            mod.main()
        except Exception as e:  # keep going; one metric failing shouldn't sink the rest
            print(f"!!! {name} FAILED: {type(e).__name__}: {e}")
        finally:
            sys.argv = old_argv

    # collect: merge every metric json that exists on disk (so a partial
    # `--only` rerun refreshes those metrics without dropping the others).
    summary = {}
    for name, (_, base) in METRICS.items():
        path = os.path.join(out_dir, f"{base}.json")
        if os.path.exists(path):
            with open(path) as f:
                summary[name] = json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'='*70}\nSUMMARY ({args.encoder}) written to {out}\n{'='*70}")
    _print_table(summary)


def _print_table(summary):
    def g(m, k):
        return summary.get(m, {}).get(k, "-")
    rows = [
        ("GER  gene-wise Pearson",   g("ger", "gene_wise_pearson")),
        ("GER  cell-wise Pearson",   g("ger", "cell_wise_pearson")),
        ("GER  gene-wise R2",        g("ger", "gene_wise_r2")),
        ("GER  cell-wise R2",        g("ger", "cell_wise_r2")),
        ("CTD  accuracy",            g("ctd", "accuracy")),
        ("CTD  macro-F1",            g("ctd", "macro_f1")),
        ("DD   accuracy",            g("dd", "accuracy")),
        ("DD   macro-F1",            g("dd", "macro_f1")),
        ("DoseD MAE (log10)",        g("dose_d", "mae")),
        ("DoseD R2",                 g("dose_d", "r2")),
        ("MOA-D accuracy",           g("moa_d", "accuracy")),
        ("MOA-D macro-F1",           g("moa_d", "macro_f1")),
        ("EGC  distance corr",       g("egc", "distance_correlation")),
        ("LNP  mean Jaccard",        g("lnp", "mean_jaccard")),
        ("PGC  silhouette",          g("pgc", "silhouette")),
        ("PGC  within/between",      g("pgc", "within_between_ratio")),
    ]
    print(f"\n{'metric':32s} value")
    print("-" * 46)
    for name, val in rows:
        v = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"{name:32s} {v}")


if __name__ == "__main__":
    main()
