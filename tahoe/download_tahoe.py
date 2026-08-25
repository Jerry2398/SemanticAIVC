"""
Download all of Tahoe-100M (data/*.parquet + metadata/) to /scratch.

~3388 shards x ~71 MB ~= 241 GB. Uses snapshot_download with the xet backend
DISABLED (it hangs on this cluster; plain HTTPS works and is resumable). Caches
to /scratch, NOT $HOME (which is nearly full).

Run on a node WITH internet (the submit/login node), in the background:
  nohup python download_tahoe.py > /scratch/yuchen.yan/tahoe_download.log 2>&1 &
"""
import os

os.environ["HF_HUB_DISABLE_XET"] = "1"          # xet backend hangs here -> use HTTPS
os.environ.setdefault("HF_HOME", "/scratch/yuchen.yan/hf_cache")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import snapshot_download

import tahoe_common as T

if __name__ == "__main__":
    os.makedirs(T.TAHOE_DIR, exist_ok=True)
    path = snapshot_download(
        repo_id="tahoebio/Tahoe-100M",
        repo_type="dataset",
        local_dir=T.TAHOE_DIR,
        # all cell shards + the small useful metadata; SKIP the 1026-file
        # pseudobulk-DE folder and the giant per-cell obs_metadata (obs is in
        # each shard already) -> not needed for VAE training.
        allow_patterns=[
            "data/*.parquet",
            "metadata/gene_metadata.parquet",
            "metadata/gene_vocabulary.json",
            "metadata/cell_line_metadata.parquet",
            "metadata/drug_metadata.parquet",
        ],
        max_workers=8,
    )
    print("TAHOE_DOWNLOAD_DONE ->", path)
