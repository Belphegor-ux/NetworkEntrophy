"""
Compute one chunk of dashboard data and write a partial JSON.

Usage:
    python update_dashboard_chunk.py --chunk {kar|fs|fi-cheap|fi-llbc|fi-llbme}

Chunks:
    kar       Karate static + iterative, all 6 metrics
    fs        Football static, all 6 metrics
    fi-cheap  Football iterative, LDC/Jaccard/CKS/CI
    fi-llbc   Football iterative, LLBC only (~3 hr)
    fi-llbme  Football iterative, LLBME only (~3 hr)

Output:
    _dash_<chunk>.json  with the curves to merge.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

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


def static_curve(G, rank_fn, col, reverse):
    df = rank_fn(G)
    rgc = NetworkDismantler(G).get_static_curve(df, col, reverse=reverse)
    x = np.linspace(0, 1, len(rgc)).tolist()
    return {"x": x, "y": list(rgc), "auc": float(np.trapezoid(rgc, x))}


def iter_curve(G, rank_fn, col, reverse):
    rgc = get_iterative_curve(G, rank_fn, col, reverse=reverse, step_size=1)
    x = np.linspace(0, 1, len(rgc)).tolist()
    return {"x": x, "y": rgc, "auc": float(np.trapezoid(rgc, x))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chunk", required=True,
                   choices=["kar", "fs", "fi-cheap", "fi-llbc", "fi-llbme"])
    args = p.parse_args()
    t0 = time.time()
    out: dict = {}

    if args.chunk == "kar":
        G = load_graph("Karate")
        out["Karate"] = {"Static": {}, "Iterative": {}}
        for name, (fn, col, rev) in METRICS.items():
            ts = time.time()
            out["Karate"]["Static"][name] = static_curve(G, fn, col, rev)
            print(f"  Karate Static {name:8s} auc={out['Karate']['Static'][name]['auc']:.4f}  ({time.time()-ts:.1f}s)", flush=True)
        for name, (fn, col, rev) in METRICS.items():
            ts = time.time()
            out["Karate"]["Iterative"][name] = iter_curve(G, fn, col, rev)
            print(f"  Karate Iter   {name:8s} auc={out['Karate']['Iterative'][name]['auc']:.4f}  ({time.time()-ts:.1f}s)", flush=True)

    elif args.chunk == "fs":
        G = load_graph("Football")
        out["Football"] = {"Static": {}}
        for name, (fn, col, rev) in METRICS.items():
            ts = time.time()
            out["Football"]["Static"][name] = static_curve(G, fn, col, rev)
            print(f"  Football Static {name:8s} auc={out['Football']['Static'][name]['auc']:.4f}  ({time.time()-ts:.1f}s)", flush=True)

    elif args.chunk == "fi-cheap":
        G = load_graph("Football")
        out["Football"] = {"Iterative": {}}
        for name in ("LDC", "Jaccard", "CKS", "CI"):
            fn, col, rev = METRICS[name]
            ts = time.time()
            out["Football"]["Iterative"][name] = iter_curve(G, fn, col, rev)
            print(f"  Football Iter {name:8s} auc={out['Football']['Iterative'][name]['auc']:.4f}  ({time.time()-ts:.1f}s)", flush=True)

    elif args.chunk == "fi-llbc":
        G = load_graph("Football")
        out["Football"] = {"Iterative": {}}
        ts = time.time()
        out["Football"]["Iterative"]["LLBC"] = iter_curve(G, rank_llbc_me, "LLBCe", True)
        print(f"  Football Iter LLBC  auc={out['Football']['Iterative']['LLBC']['auc']:.4f}  ({time.time()-ts:.1f}s)", flush=True)

    elif args.chunk == "fi-llbme":
        G = load_graph("Football")
        out["Football"] = {"Iterative": {}}
        ts = time.time()
        out["Football"]["Iterative"]["LLBME"] = iter_curve(G, rank_llbc_me, "LLBMEe1", True)
        print(f"  Football Iter LLBME auc={out['Football']['Iterative']['LLBME']['auc']:.4f}  ({time.time()-ts:.1f}s)", flush=True)

    out_path = ROOT / f"_dash_{args.chunk}.json"
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\nWrote {out_path}  total={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
