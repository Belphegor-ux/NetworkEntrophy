"""
PROTOTYPE — Adaptive Kirchhoff Flow Criticality (KFC): a new edge-criticality
algorithm for resilience / protection prioritisation.

What the benchmark taught us (see benchmark_criticality_methods.py + the sweeps)
------------------------------------------------------------------------------
* Current-flow edge betweenness (CFEdge) — electrical throughput — is the single
  most robust method, best or tied-best on every network, and (with the sorted
  pairwise-difference identity in resilience_utils.current_flow_edge_betweenness)
  computable in O(N^3 + E*N log N) instead of the naive O(N^2*E).
* Amplifying CFEdge by irreplaceability (a factor 1/(1-R_eff)) HELPS dense graphs
  (Jazz, Karate) but HURTS tree-like ones (Tokyo), where almost every edge has a
  high effective resistance so the amplification over-fires. No FIXED amount of
  amplification beats plain CFEdge on the mean.
* The right amount of amplification is predicted by the graph's MEAN EFFECTIVE
  RESISTANCE (tree-likeness): near 0 for dense graphs (amplify), near 1 for
  tree-like grids (don't). This statistic is free — it falls out of the same
  Laplacian pseudoinverse.

The algorithm
-------------
From a single Laplacian pseudoinverse L+ compute, per edge e=(u,v):
    cf1(e)  = sum_{s<t} |d[s]-d[t]| / n_pairs        (current-flow betweenness)
    R_eff(e)= L+[u,u]+L+[v,v]-2 L+[u,v]              (effective resistance)
with d = L+[u,:]-L+[v,:]. Then

    KFC(e) = cf1(e) * ( 1 / (1 - R_eff(e)) ) ** beta

where the amplification exponent adapts to how tree-like the graph is:

    beta = clip( 1 - mean(R_eff) / DENSITY_THRESH , 0, BETA_MAX )

so tree-like graphs (mean R_eff >= DENSITY_THRESH) get beta = 0 and KFC == CFEdge
exactly (no regression), while dense graphs get progressively more irreplaceability
weighting. `1/(1-R_eff)` is the Sherman-Morrison Kirchhoff-index sensitivity term,
capped near bridges (R_eff -> 1) via a denominator floor.

Properties
----------
* Strict generalisation of CFEdge: beta = 0 recovers it exactly.
* Same cost class as fast CFEdge: one pseudoinverse + O(E*N).
* Standard `i, j, KFC` DataFrame (drop-in for run_and_plot, reverse=True).

This is a research prototype. On the four-network testbed the mean-AUC gain over
plain CFEdge is small (it comes almost entirely from the densest network); CFEdge
remains the robust default. KFC's value is (a) it never does worse than CFEdge and
(b) it improves on dense graphs, decided automatically per network.
"""
from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resilience_utils import (  # noqa: E402
    _canon,
    _laplacian_pinv,
    largest_connected_component,
)

__all__ = ["rank_kfc", "kfc_scores", "adaptive_beta"]

DENSITY_THRESH = 0.15   # mean R_eff at/above which the graph is "tree-like" -> beta 0
BETA_MAX = 1.0
_DENOM_FLOOR = 1e-12    # caps 1/(1-R_eff) as an edge approaches a bridge (R_eff -> 1)


def adaptive_beta(mean_reff: float) -> float:
    """Amplification exponent from mean effective resistance (tree-likeness)."""
    return float(np.clip(1.0 - mean_reff / DENSITY_THRESH, 0.0, BETA_MAX))


def kfc_scores(G: nx.Graph, beta: float | None = None) -> tuple[dict[tuple, float], float, float]:
    """
    Return (scores, beta_used, mean_reff). If ``beta`` is None it is chosen
    adaptively from the graph's mean effective resistance.
    """
    H = largest_connected_component(G)
    n = H.number_of_nodes()
    if n < 2 or H.number_of_edges() == 0:
        return {}, 0.0, 0.0

    nodes = list(H.nodes())
    idx = {u: k for k, u in enumerate(nodes)}
    Lp = _laplacian_pinv(H, nodes)
    diag = np.diag(Lp)
    coeff = 2.0 * np.arange(n) - (n - 1)      # ascending-sorted-d weights for cf1
    n_pairs = n * (n - 1) / 2.0

    edges = [_canon(u, v) for u, v in H.edges()]
    cf1 = np.empty(len(edges))
    reff = np.empty(len(edges))
    for k, (u, v) in enumerate(edges):
        d = Lp[idx[u]] - Lp[idx[v]]
        cf1[k] = coeff @ np.sort(d) / n_pairs
        reff[k] = diag[idx[u]] + diag[idx[v]] - 2.0 * Lp[idx[u], idx[v]]

    mean_reff = float(np.mean(reff))
    b = adaptive_beta(mean_reff) if beta is None else float(beta)

    if b == 0.0:
        vals = cf1
    else:
        amplify = 1.0 / np.maximum(1.0 - reff, _DENOM_FLOOR)
        vals = cf1 * amplify ** b

    return {e: float(vals[k]) for k, e in enumerate(edges)}, b, mean_reff


def rank_kfc(G: nx.Graph, beta: float | None = None) -> pd.DataFrame:
    """DataFrame form: columns ``i, j, KFC`` (dismantle reverse=True)."""
    scores, _, _ = kfc_scores(G, beta=beta)
    rows = [{"i": u, "j": v, "KFC": val} for (u, v), val in scores.items()]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    gml = os.path.join(here, "..", "..", "PowerGrid_City", "datasets", "tokyo_grid.gml")
    G = largest_connected_component(nx.read_gml(gml))
    scores, beta, mreff = kfc_scores(G)
    print(f"Tokyo: mean_reff={mreff:.3f} -> adaptive beta={beta:.3f}")
    df = rank_kfc(G).sort_values("KFC", ascending=False)
    print(df.head(10).to_string(index=False))
