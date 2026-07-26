"""
Regenerate dashboard_data.json for Karate and Football (post-tie-break alignment).

Preserves existing Jazz and Tokyo City entries.
Maps dashboard short names to my spec-correct columns:
    CKS  -> LKS
    LDC  -> LDC
    Jaccard -> Jaccard (reverse=False)
    LLBC -> LLBCe
    LLBME -> LLBMEe1 (Shannon entropy form per new_instructions.md)
    CI   -> CI_e_mul_skin (preserves the historical dashboard CI series)
    IE   -> dropped (deprecated per new_instructions.md §3)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src" / "utils"))
np.random.seed(42)

from network_utils import NetworkDismantler  # noqa: E402
from network_utils_iter import get_iterative_curve  # noqa: E402

sys.path.insert(0, str(ROOT / "Karate" / "CI"))
sys.path.insert(0, str(ROOT / "Karate" / "LDC"))
sys.path.insert(0, str(ROOT / "Karate" / "Jaccard"))
sys.path.insert(0, str(ROOT / "Karate" / "CKS"))
sys.path.insert(0, str(ROOT / "Karate" / "LLBC"))

from collective_influence import rank_ci  # noqa: E402
from LDC_link_degree_centrality import rank_cdc  # noqa: E402
from jaccard_index import rank_jaccard  # noqa: E402
from CKS_Link_K_Shell import rank_lks  # noqa: E402
from LLBC_contrast import rank_llbc_me  # noqa: E402

# dashboard_short -> (rank_function, my_column_name, reverse)
METRICS = {
    "LDC":     (rank_cdc,        "LDC",            True),
    "Jaccard": (rank_jaccard,    "Jaccard",        False),
    "CKS":     (rank_lks,        "LKS",            True),
    "LLBC":    (rank_llbc_me,    "LLBCe",          True),
    "LLBME":   (rank_llbc_me,    "LLBMEe1",        True),
    "CI":      (rank_ci,         "CI_e_mul_skin",  True),
}


def load_graph(dataset: str) -> nx.Graph:
    if dataset == "Karate":
        G = nx.karate_club_graph()
    elif dataset == "Football":
        G = nx.read_gml(str(ROOT / "datasets" / "football" / "football.gml"), label="id")
    else:
        raise ValueError(dataset)
    return nx.convert_node_labels_to_integers(G, label_attribute="old_label")


def static_curve(G: nx.Graph, rank_fn, col: str, reverse: bool):
    df = rank_fn(G)
    dism = NetworkDismantler(G)
    rgc = dism.get_static_curve(df, col, reverse=reverse)
    x = np.linspace(0, 1, len(rgc)).tolist()
    auc = float(np.trapezoid(rgc, x))
    return {"x": x, "y": list(rgc), "auc": auc}


def iter_curve(G: nx.Graph, rank_fn, col: str, reverse: bool):
    rgc = get_iterative_curve(G, rank_fn, col, reverse=reverse, step_size=1)
    x = np.linspace(0, 1, len(rgc)).tolist()
    auc = float(np.trapezoid(rgc, x))
    return {"x": x, "y": rgc, "auc": auc}


def main() -> None:
    t0 = time.time()
    json_path = ROOT / "dashboard_app" / "dashboard_data.json"
    with open(json_path) as f:
        data = json.load(f)

    for dataset in ("Karate", "Football"):
        print(f"\n=== {dataset} ===")
        G = load_graph(dataset)
        print(f"  graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        for mode_key, runner in (("Static", static_curve), ("Iterative", iter_curve)):
            print(f"  {mode_key}:")
            data.setdefault(dataset, {}).setdefault(mode_key, {})
            for short_name, (rank_fn, col, reverse) in METRICS.items():
                ts = time.time()
                entry = runner(G, rank_fn, col, reverse=reverse)
                data[dataset][mode_key][short_name] = entry
                print(f"    {short_name:8s} (col={col:14s}) auc={entry['auc']:.4f}  ({time.time()-ts:.1f}s)")
            # Drop deprecated IE if present
            data[dataset][mode_key].pop("IE", None)

    with open(json_path, "w") as f:
        json.dump(data, f)
    print(f"\nWrote {json_path}  total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
