"""
Build the Cell-Ontology pairwise-distance matrix that COG (organization/cog.py)
consumes via --ontology_csv, for the Kang PBMC cell types.

Maps each cell_type -> a Cell Ontology (CL) term, loads the CL ontology with
obonet, builds the undirected is_a graph, and takes the shortest-path (number of
is_a hops) between every pair of terms. Writes a cell_type x cell_type CSV.

Edit CL_MAP if you change the label set / dataset.

ENV needs obonet + networkx:  pip install obonet networkx
Run: HDF5_USE_FILE_LOCKING=FALSE python build_ontology_distances.py
"""
import argparse
import os

import numpy as np
import pandas as pd

# Kang PBMC cell_type -> Cell Ontology term
CL_MAP = {
    "CD4 T cells":        "CL:0000624",  # CD4-positive T cell
    "CD8 T cells":        "CL:0000625",  # CD8-positive T cell
    "B cells":            "CL:0000236",  # B cell
    "NK cells":           "CL:0000623",  # natural killer cell
    "CD14+ Monocytes":    "CL:0000860",  # classical monocyte
    "FCGR3A+ Monocytes":  "CL:0000875",  # non-classical monocyte
    "Dendritic cells":    "CL:0000451",  # dendritic cell
    "Megakaryocytes":     "CL:0000556",  # megakaryocyte
}
CL_OBO_URL = "http://purl.obolibrary.org/obo/cl/cl-basic.obo"


def main():
    import obonet
    import networkx as nx

    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--obo", default=CL_OBO_URL, help="cl-basic.obo URL or local path")
    args = ap.parse_args()

    import config as C
    out_csv = args.out_csv or C.ONTOLOGY_CSV

    print("loading Cell Ontology from", args.obo)
    g = obonet.read_obo(args.obo)                       # directed multigraph
    # undirected graph over is_a edges only -> tree-like hop distance
    isa = nx.Graph()
    isa.add_nodes_from(g.nodes())
    for u, v, key in g.edges(keys=True):
        if key == "is_a":
            isa.add_edge(u, v)

    names = list(CL_MAP.keys())
    terms = [CL_MAP[n] for n in names]
    for t in terms:
        if t not in isa:
            raise KeyError(f"{t} not in ontology graph (check CL id / obo version)")

    n = len(names)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = nx.shortest_path_length(isa, terms[i], terms[j])
    df = pd.DataFrame(D, index=names, columns=names).astype(int)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv)
    print("wrote", out_csv)
    print(df)


if __name__ == "__main__":
    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    main()
