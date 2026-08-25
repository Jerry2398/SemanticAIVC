"""Build the cell-line (context c) vocabulary from Tahoe; drug vocab already exists.

  python build_vocab.py
-> <tahoe>/cell_line_vocab.json  {"cell_lines": [...], "cell_line_to_id": {...}, "n": 50}
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tahoe"))

import pyarrow.parquet as pq
import tahoe_common as T

OUT = os.path.join(T.TAHOE_DIR, "cell_line_vocab.json")


def main(every=100):
    shards = T.list_shards("all")
    seen = set()
    for p in shards[::every]:
        seen |= set(pq.read_table(p, columns=["cell_line_id"]).to_pandas()["cell_line_id"].astype(str))
    cls = sorted(seen)
    json.dump({"cell_lines": cls, "cell_line_to_id": {c: i for i, c in enumerate(cls)}, "n": len(cls)},
              open(OUT, "w"))
    print(f"scanned {len(shards[::every])} shards -> {len(cls)} cell lines; wrote {OUT}")


if __name__ == "__main__":
    main()
