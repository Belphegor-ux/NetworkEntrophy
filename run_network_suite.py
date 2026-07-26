"""
Unified all-metrics dismantling suite for arbitrary edge-list networks.

Runs the full 9-metric static + iterative dismantling analysis on a single
network given as a two-column edge-list CSV (header `i,j`), in the same style
as PowerGrid_City/run_city_analysis.py. Reuses the shared drivers and the
PowerGrid rank functions so the metric definitions stay in one place.

Outputs (to results_<name>/):
  Static:
    <name>_metrics.csv        - i, j, <9 metric values> (sortable record)
    plot_<metric>.png         - per-metric static RGC curve
  Iterative:
    plot_iterative_<group>.png       - per rank-group iterative RGC curves
    <name>_iter_order.csv            - i, j, order_<metric>... (removal order)
    <name>_iter_<metric>_rgc.csv     - Fraction_Removed, RGC per metric
    aucs_<name>.csv                  - static + iterative AUC per metric

Usage:
    python run_network_suite.py --csv "Networks to check/dolphins.csv" --name dolphins
    python run_network_suite.py --csv "Networks to check/2transport_edges_re.csv" \
        --name transport --step-size 5
    python run_network_suite.py --csv ... --name ... --no-iterative
"""
import argparse
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src", "utils"))
sys.path.insert(0, os.path.join(ROOT, "PowerGrid_City"))

from network_utils import run_and_plot  # noqa: E402
from network_utils_iter import run_iterative_benchmark  # noqa: E402
from run_city_analysis import (  # noqa: E402
    rank_ci,
    rank_jaccard,
    rank_ldc,
    rank_llbc_me,
    rank_lks,
)

# (label, rank_func, reverse-spec). reverse may be a bool or a per-column dict.
# Jaccard distance: lower = more critical bridge -> remove low-first (reverse=False).
# LLBMEe1: negative, smallest = most critical -> reverse=False; LLBCe stays high-first.
RANK_GROUPS = [
    ("ldc", rank_ldc, True),
    ("jaccard", rank_jaccard, False),
    ("lks", rank_lks, True),
    ("ci", rank_ci, True),
    ("llbc", rank_llbc_me, {"LLBCe": True, "LLBMEe1": False}),
]


def load_lcc_graph(csv_path: str) -> nx.Graph:
    """Load an `i,j` edge-list CSV into the largest connected component."""
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    src, dst = cols[0], cols[1]
    G = nx.from_pandas_edgelist(df, source=src, target=dst)
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_nodes() == 0:
        raise ValueError(f"No edges loaded from {csv_path}")
    lcc_nodes = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc_nodes).copy()
    G = nx.convert_node_labels_to_integers(G, label_attribute="old_label")
    return G


def write_static(G: nx.Graph, name: str, out_dir: str) -> dict:
    """Compute all metrics, write the values CSV + per-metric static plots."""
    metric_dfs = [func(G) for _, func, _ in RANK_GROUPS]
    combined = metric_dfs[0]
    for extra in metric_dfs[1:]:
        combined = combined.merge(extra, on=["i", "j"])
    combined.to_csv(os.path.join(out_dir, f"{name}_metrics.csv"), index=False)

    aucs = {}
    for label, func, reverse in RANK_GROUPS:
        plot_path = os.path.join(out_dir, f"plot_{label}.png")
        results = run_and_plot(G, f"{name} {label}", func, plot_path, reverse=reverse)
        for metric_col, res in results.items():
            aucs[metric_col] = res["auc"]
    return aucs


def write_iterative(G: nx.Graph, name: str, out_dir: str, step_size: int) -> dict:
    """Run iterative dismantling: plots, removal-order CSVs, RGC curve CSVs."""
    aucs = {}
    order_frames = []
    for label, func, reverse in RANK_GROUPS:
        plot_path = os.path.join(out_dir, f"plot_iterative_{label}.png")
        order_path = os.path.join(out_dir, f"{name}_iter_order_{label}.csv")
        results = run_iterative_benchmark(
            G, f"{name} {label}", func, filename=plot_path,
            reverse=reverse, step_size=step_size, order_csv_path=order_path,
        )
        order_frames.append(pd.read_csv(order_path))
        for metric_col, res in results.items():
            aucs[metric_col] = res["auc"]
            rgc_df = pd.DataFrame({"Fraction_Removed": res["x"], "RGC": res["y"]})
            rgc_df.to_csv(
                os.path.join(out_dir, f"{name}_iter_{metric_col}_rgc.csv"), index=False
            )

    # Merge per-group removal-order files into one wide CSV, then drop the parts.
    merged = order_frames[0]
    for frame in order_frames[1:]:
        merged = merged.merge(frame, on=["i", "j"])
    merged.to_csv(os.path.join(out_dir, f"{name}_iter_order.csv"), index=False)
    for label, _, _ in RANK_GROUPS:
        part = os.path.join(out_dir, f"{name}_iter_order_{label}.csv")
        if os.path.exists(part):
            os.remove(part)
    return aucs


def main() -> None:
    parser = argparse.ArgumentParser(description="All-metrics dismantling suite.")
    parser.add_argument("--csv", required=True, help="Path to i,j edge-list CSV.")
    parser.add_argument("--name", required=True, help="Short network name for outputs.")
    parser.add_argument("--out-dir", default=None, help="Output dir (default results_<name>/).")
    parser.add_argument("--step-size", type=int, default=1, help="Edges per iterative recompute.")
    parser.add_argument("--no-iterative", action="store_true", help="Skip iterative analysis.")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(ROOT, f"results_{args.name}")
    os.makedirs(out_dir, exist_ok=True)

    G = load_lcc_graph(args.csv)
    print(f"[{args.name}] LCC: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    static_aucs = write_static(G, args.name, out_dir)
    iter_aucs = write_iterative(G, args.name, out_dir, args.step_size) if not args.no_iterative else {}

    metrics = sorted(set(static_aucs) | set(iter_aucs))
    summary = pd.DataFrame(
        {
            "metric": metrics,
            "auc_static": [static_aucs.get(m) for m in metrics],
            "auc_iterative": [iter_aucs.get(m) for m in metrics],
        }
    )
    summary.to_csv(os.path.join(out_dir, f"aucs_{args.name}.csv"), index=False)
    print(f"[{args.name}] done -> {out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
