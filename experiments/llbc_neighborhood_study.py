"""
Neighborhood-size study for LLBCe / LLBMEe1 edge-criticality metrics.

Parameterizes the central domain order k of the LLBC subset-betweenness:
    Gamma_k(e=(u,v)) = { w : dist(w, u) <= k or dist(w, v) <= k }
so k=1 is the current first-order domain {u,v} U N(u) U N(v) (Eq. 8 of
MDPI Entropy 26(4) 315), k=2 adds neighbors-of-Gamma_1, etc.

For each (network, k) it computes, ON THE FULL GRAPH (no subgraph extraction):
    LLBCe(e)    = edge_betweenness_centrality_subset(G, Gamma_k, Gamma_k,
                  normalized=False)[e] / (|Gamma_k| * (|Gamma_k| - 1))
    LLBMEe1(e)  = -raw(e) * sum_{e2 in N(e)} log(max(raw(e2), 1e-10))
                  (0 when raw(e) <= 0)  -- inverted criticality, smallest =
                  most critical (matches PowerGrid_City/run_city_analysis.py).

Then runs STATIC dismantling (src/utils/network_utils.NetworkDismantler):
LLBCe removed high-first (reverse=True), LLBMEe1 removed smallest-first
(reverse=False). Lower RGC-AUC = better metric.

Node iteration is explicitly sorted everywhere to sidestep the known
tie-breaking non-determinism of edge_betweenness_centrality_subset.

Per-cell wall-clock budget: if the extrapolated (or actual) compute time for a
(network, k) cell exceeds BUDGET_S seconds, the cell is skipped and recorded.

Outputs (results_llbc_neighborhood/):
    aucs.csv                      network, metric, k, auc, runtime_s, status
    scores_<network>_k<k>.csv     i, j, LLBCe, LLBMEe1 (per completed cell)
    rgc_<network>_<metric>.png    RGC comparison plot, curves for k=1/2/3
    spearman.csv                  rank correlation of scores k=1 vs k=2/3
    sanity_karate.csv             LLBCe at full-cover k vs full edge betweenness

Usage:
    .venv/Scripts/python.exe experiments/llbc_neighborhood_study.py
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "utils"))

from network_utils import NetworkDismantler  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results_llbc_neighborhood")
BUDGET_S = 900.0          # 15 min per (network, k) cell
PILOT_EDGES = 20          # edges used to extrapolate cell runtime
K_VALUES = (1, 2, 3)
LOG_FLOOR = 1e-10

NETWORKS = {
    "karate": os.path.join(ROOT, "Networks to check", "karate.csv"),
    "dolphins": os.path.join(ROOT, "Networks to check", "dolphins.csv"),
    "football": os.path.join(ROOT, "Networks to check", "football.csv"),
    "jazz": os.path.join(ROOT, "_derived_jazz.csv"),
}


def load_lcc_graph(csv_path: str) -> nx.Graph:
    """Load an i,j edge-list CSV: undirected, self-loops dropped, LCC only."""
    df = pd.read_csv(csv_path)
    src, dst = list(df.columns)[:2]
    G = nx.from_pandas_edgelist(df, source=src, target=dst)
    G.remove_edges_from(nx.selfloop_edges(G))
    lcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc).copy()
    G = nx.convert_node_labels_to_integers(G, ordering="sorted",
                                           label_attribute="old_label")
    return G


def gamma_k(G: nx.Graph, u, v, k: int) -> list:
    """Sorted node list of the order-k central domain of edge (u, v)."""
    nodes = set(nx.single_source_shortest_path_length(G, u, cutoff=k))
    nodes |= set(nx.single_source_shortest_path_length(G, v, cutoff=k))
    return sorted(nodes)


def compute_llbc_llbme(G: nx.Graph, k: int, budget_s: float = BUDGET_S):
    """LLBCe/LLBMEe1 with an order-k central domain.

    Returns (df, elapsed_s, status) where status is 'ok' or a skip reason;
    df is None when skipped.
    """
    edges = sorted(tuple(sorted(e)) for e in G.edges())
    llbc_raw, llbc_e = {}, {}
    t0 = time.perf_counter()

    for idx, (u, v) in enumerate(edges):
        fc_nodes = gamma_k(G, u, v, k)
        ebc = nx.edge_betweenness_centrality_subset(
            G, sources=fc_nodes, targets=fc_nodes, normalized=False
        )
        raw = ebc.get((u, v), ebc.get((v, u), 0))
        denom = len(fc_nodes) * (len(fc_nodes) - 1)
        llbc_raw[(u, v)] = raw
        llbc_e[(u, v)] = raw / denom if denom > 0 else 0.0

        elapsed = time.perf_counter() - t0
        if idx + 1 == PILOT_EDGES:
            projected = elapsed / PILOT_EDGES * len(edges)
            if projected > budget_s:
                return None, elapsed, (
                    f"skipped: projected {projected:.0f}s > {budget_s:.0f}s budget"
                )
        if elapsed > budget_s:
            return None, elapsed, (
                f"skipped: exceeded {budget_s:.0f}s budget at edge "
                f"{idx + 1}/{len(edges)}"
            )

    # Mapping entropy on RAW betweenness (reference formula, k-agnostic).
    llbme = {}
    for u, v in edges:
        neigh = [tuple(sorted((u, n))) for n in sorted(G.neighbors(u)) if n != v]
        neigh += [tuple(sorted((v, n))) for n in sorted(G.neighbors(v)) if n != u]
        sum_log = sum(np.log(max(llbc_raw.get(e2, LOG_FLOOR), LOG_FLOOR))
                      for e2 in neigh)
        val = llbc_raw[(u, v)]
        llbme[(u, v)] = (-val * sum_log) if val > 0 else 0.0

    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(
        [{"i": u, "j": v, "LLBCe": llbc_e[(u, v)], "LLBMEe1": llbme[(u, v)]}
         for u, v in edges]
    )
    return df, elapsed, "ok"


def auc_of(rgc: list) -> float:
    x = np.linspace(0, 1, len(rgc))
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trap(rgc, x))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation with average ranks for ties (no scipy)."""
    def ranks(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(len(x), dtype=float)
        # average tied ranks
        sx = x[order]
        i = 0
        while i < len(sx):
            j = i
            while j + 1 < len(sx) and sx[j + 1] == sx[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = ranks(np.asarray(a, dtype=float)), ranks(np.asarray(b, dtype=float))
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def sanity_check_karate(G: nx.Graph) -> pd.DataFrame:
    """At k covering the whole graph, LLBCe must match full edge betweenness."""
    k_full = nx.diameter(G)  # Gamma_k = V for every edge
    df, _, status = compute_llbc_llbme(G, k_full, budget_s=1e9)
    assert status == "ok"
    full_ebc = nx.edge_betweenness_centrality(G, normalized=False)
    rows = []
    for _, r in df.iterrows():
        e = (int(r["i"]), int(r["j"]))
        ref = full_ebc.get(e, full_ebc.get((e[1], e[0]), 0.0))
        rows.append({"i": e[0], "j": e[1], "LLBCe_kfull": r["LLBCe"],
                     "full_EBC_raw": ref})
    out = pd.DataFrame(rows)
    rho = spearman(out["LLBCe_kfull"].to_numpy(), out["full_EBC_raw"].to_numpy())
    # With Gamma_k = V the denominator N(N-1) is constant, so LLBCe should be
    # exactly proportional to full raw EBC.
    ratio = out["full_EBC_raw"] / out["LLBCe_kfull"].replace(0, np.nan)
    out.attrs["spearman"] = rho
    out.attrs["k_full"] = k_full
    out.attrs["ratio_cv"] = float(np.nanstd(ratio) / np.nanmean(ratio))
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    auc_rows, spearman_rows = [], []
    curves = {}   # (net, metric) -> {k: rgc}
    scores = {}   # (net, k) -> df

    for net, path in NETWORKS.items():
        G = load_lcc_graph(path)
        print(f"\n=== {net}: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges ===")
        dismantler = NetworkDismantler(G)

        for k in K_VALUES:
            print(f"  k={k} ...", flush=True)
            df, elapsed, status = compute_llbc_llbme(G, k)
            if df is None:
                print(f"    {status} ({elapsed:.1f}s spent)")
                for metric in ("LLBCe", "LLBMEe1"):
                    auc_rows.append({"network": net, "metric": metric, "k": k,
                                     "auc": np.nan, "runtime_s": round(elapsed, 2),
                                     "status": status})
                continue

            scores[(net, k)] = df
            df.to_csv(os.path.join(OUT_DIR, f"scores_{net}_k{k}.csv"),
                      index=False)
            for metric, reverse in (("LLBCe", True), ("LLBMEe1", False)):
                rgc = dismantler.get_static_curve(df, metric, reverse=reverse)
                a = auc_of(rgc)
                curves.setdefault((net, metric), {})[k] = rgc
                auc_rows.append({"network": net, "metric": metric, "k": k,
                                 "auc": round(a, 5),
                                 "runtime_s": round(elapsed, 2), "status": "ok"})
                print(f"    {metric}: AUC={a:.4f}  ({elapsed:.1f}s)")

        # Spearman between k=1 and higher-k scores
        base = scores.get((net, 1))
        if base is not None:
            for k in K_VALUES[1:]:
                other = scores.get((net, k))
                if other is None:
                    continue
                merged = base.merge(other, on=["i", "j"],
                                    suffixes=("_k1", f"_k{k}"))
                for metric in ("LLBCe", "LLBMEe1"):
                    rho = spearman(merged[f"{metric}_k1"].to_numpy(),
                                   merged[f"{metric}_k{k}"].to_numpy())
                    spearman_rows.append({"network": net, "metric": metric,
                                          "k_pair": f"1v{k}",
                                          "spearman": round(rho, 4)})

    pd.DataFrame(auc_rows).to_csv(os.path.join(OUT_DIR, "aucs.csv"), index=False)
    pd.DataFrame(spearman_rows).to_csv(os.path.join(OUT_DIR, "spearman.csv"),
                                       index=False)

    # RGC comparison plots: one per (network, metric), curves for each k.
    for (net, metric), by_k in curves.items():
        plt.figure(figsize=(9, 5.5))
        for k in sorted(by_k):
            rgc = by_k[k]
            x = np.linspace(0, 1, len(rgc))
            plt.plot(x, rgc, linewidth=2,
                     label=f"k={k} (AUC={auc_of(rgc):.3f})")
        plt.title(f"{net}: {metric} static dismantling vs neighborhood order k")
        plt.xlabel("Fraction of Edges Removed")
        plt.ylabel("Relative Giant Component Size")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"rgc_{net}_{metric}.png"), dpi=200)
        plt.close()

    # Sanity check on karate: full-cover k -> full edge betweenness.
    G_karate = load_lcc_graph(NETWORKS["karate"])
    sanity = sanity_check_karate(G_karate)
    sanity.to_csv(os.path.join(OUT_DIR, "sanity_karate.csv"), index=False)
    print(f"\nSanity (karate, k={sanity.attrs['k_full']} = diameter): "
          f"Spearman(LLBCe_kfull, full EBC) = {sanity.attrs['spearman']:.6f}, "
          f"ratio CV = {sanity.attrs['ratio_cv']:.2e}")
    with open(os.path.join(OUT_DIR, "sanity_karate_summary.txt"), "w") as fh:
        fh.write(
            f"k_full={sanity.attrs['k_full']}\n"
            f"spearman={sanity.attrs['spearman']:.8f}\n"
            f"ratio_cv={sanity.attrs['ratio_cv']:.3e}\n"
        )

    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
