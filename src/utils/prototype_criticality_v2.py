"""
PROTOTYPE — Generalized Kirchhoff Flow Criticality, v2 (``rank_kfc_v2``).

Why v1 failed off-circuit (see results_comparison/kfc_v2_diagnosis.md)
----------------------------------------------------------------------
v1's amplifier ``(1/(1-R_eff))**beta`` has two degeneracies:

* **Bridge saturation** — at a bridge R_eff = 1 exactly, so the term explodes
  to the denominator floor (1e12). On tree-like graphs (Tokyo: 56 bridges)
  every bridge — including structurally worthless leaf edges — jumps to the
  top of the ranking, wrecking the dismantling curve (AUC 0.179 -> 0.324 at
  beta=0.25). This forced the mean-R_eff auto-tuning to be so conservative
  that beta = 0 on five of six benchmark networks, i.e. v1 IS CFEdge almost
  everywhere by construction.
* **Scale degeneracy** — on dense graphs all R_eff are tiny (Jazz: 0.04-0.10)
  so the amplifier spans only ~[1.04, 1.11]^beta and barely perturbs the
  ranking; on quasi-regular graphs (Football: cv(R_eff) = 0.09) R_eff carries
  no signal at all and any amplification is pure noise.

The v2 generalisation
---------------------
What actually marks a critical edge on modular (social) networks is whether
it BRIDGES communities — and the empirical sweep showed the same is true on
the Tokyo grid (its 500 kV inter-region trunks are inter-community edges).
v2 therefore fuses current-flow throughput with community bridging instead of
raw effective resistance:

    KFC_v2(e) = cf1(e) + w * [e is inter-community]      (two-tier score)

where cf1 is exact current-flow edge betweenness (same sorted pairwise-
difference identity as v1 / resilience_utils) and the tier shift
``w = max(cf1)`` strictly separates the classes: ALL inter-community edges
rank before ALL intra-community edges, each tier internally ordered by cf1.
This is the ``gamma -> infinity`` limit of the blend
``cf1 * (1 + gamma * inter)``; the finite-gamma sweep saturates to it by
gamma ~ 4 on every benchmark network, so the limit is the parameter-free
default (``community_weight=None``). A finite ``community_weight`` gives the
multiplicative blend, and ``community_weight = 0`` recovers CFEdge EXACTLY
(strict generalisation, unit-tested).

Communities come from ``prototype_criticality_fast.detect_communities``
(Louvain, fixed seed -> deterministic; 'leiden' = documented fallback).
A single-community partition degrades gracefully to pure CFEdge.

Benchmark (static dismantling AUC, lower = better; seed 42):

    network   CFEdge   KFC(v1)  KFC_v2
    karate    0.5439   0.5439   0.4244
    dolphins  0.4243   0.4243   0.3374
    lesmis    0.3367   0.3367   0.2852
    football  0.3449   0.3449   0.3326
    jazz      0.5690   0.5670   0.4104
    tokyo     0.1788   0.1788   0.1093

KFC_v2 is strictly better than CFEdge and v1 on all six networks and is
seed-robust (max seed-to-seed AUC spread ~0.015). Pure NumPy + NetworkX,
input graph never mutated, standard ``i, j, KFC_v2`` frame (drop-in for
``run_and_plot``, dismantle ``reverse=True``).
"""
from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype_criticality_fast import (  # noqa: E402
    _grounded_inverse,
    _weighted_pairwise_absdiff,
    community_representatives,
    detect_communities,
)
from resilience_utils import (  # noqa: E402
    _canon,
    _laplacian_pinv,
    largest_connected_component,
)

__all__ = ["kfc_v2_scores", "rank_kfc_v2", "kfc_v2_fast_scores", "rank_kfc_v2_fast"]

_DEFAULT_SEED = 42
_DEFAULT_REP_FRACTION = 0.5
_DEFAULT_MIN_CENTERS = 8
# ~1 GB of float64 for the E x R working block in kfc_v2_fast_scores.
_CHUNK_BUDGET_FLOATS = 130_000_000


def kfc_v2_scores(
    G: nx.Graph,
    community_weight: float | None = None,
    method: str = "louvain",
    seed: int = _DEFAULT_SEED,
) -> tuple[dict[tuple, float], dict]:
    """
    Community-tiered current-flow criticality.

    ``community_weight=None`` (default) — strict two-tier score
    ``cf1 + max(cf1) * inter`` (the gamma->inf limit: every inter-community
    edge outranks every intra-community edge, cf1 order within each tier).
    A finite value ``g`` gives the blend ``cf1 * (1 + g * inter)``;
    ``g = 0`` recovers exact CFEdge values.

    Returns ``(scores, info)`` with info keys ``n_communities``,
    ``frac_inter``, ``community_weight`` (None means tiered), ``method``.
    Works on the largest connected component; input untouched.
    """
    H = largest_connected_component(G)
    n = H.number_of_nodes()
    info = {"n_communities": 0, "frac_inter": 0.0,
            "community_weight": community_weight, "method": method}
    if n < 2 or H.number_of_edges() == 0:
        return {}, info

    nodes = list(H.nodes())
    idx = {u: k for k, u in enumerate(nodes)}
    Lp = _laplacian_pinv(H, nodes)
    coeff = 2.0 * np.arange(n) - (n - 1)
    n_pairs = n * (n - 1) / 2.0

    edges = [_canon(u, v) for u, v in H.edges()]
    cf1 = np.empty(len(edges))
    for k, (u, v) in enumerate(edges):
        d = Lp[idx[u]] - Lp[idx[v]]
        cf1[k] = coeff @ np.sort(d) / n_pairs

    if community_weight is not None and float(community_weight) == 0.0:
        # Degenerate setting: exact CFEdge (no partition needed).
        info["n_communities"] = 1
        return {e: float(cf1[k]) for k, e in enumerate(edges)}, info

    parts = detect_communities(H, method=method, seed=seed)
    membership = {v: ci for ci, comm in enumerate(parts) for v in comm}
    inter = np.array([1.0 if membership[u] != membership[v] else 0.0
                      for u, v in edges])
    info["n_communities"] = len(parts)
    info["frac_inter"] = float(inter.mean())

    if community_weight is None:
        vals = cf1 + float(cf1.max()) * inter        # strict tier separation
    else:
        vals = cf1 * (1.0 + float(community_weight) * inter)

    return {e: float(vals[k]) for k, e in enumerate(edges)}, info


def _community_centers(G: nx.Graph, parts: list[set]) -> list:
    """Center of each community: max closeness of the community subgraph
    (ties by sorted node label — deterministic). Same rule as
    ``community_representatives``' first pick."""
    centers = []
    for comm in parts:
        clo = nx.closeness_centrality(G.subgraph(comm))
        centers.append(min(clo, key=lambda v: (-clo[v], str(v))))
    return centers


def _intercommunity_mincut_edges(
    H: nx.Graph, parts: list[set], membership: dict
) -> set[tuple]:
    """
    Min-cut membership restricted to ADJACENT community pairs.

    For every pair of communities joined by at least one edge, compute the
    minimum edge cut between their centers (max-flow) on the FULL graph and
    collect the cut edges. These are the true bottlenecks of inter-community
    exchange. Cost: one max-flow per adjacent community pair — typically a
    handful of calls, versus MinCutCrit's thousands of sampled node pairs.
    """
    centers = _community_centers(H, parts)
    adjacent: set[tuple[int, int]] = set()
    for u, v in H.edges():
        cu, cv = membership[u], membership[v]
        if cu != cv:
            adjacent.add((min(cu, cv), max(cu, cv)))

    cut_edges: set[tuple] = set()
    for ca, cb in sorted(adjacent):
        a, b = centers[ca], centers[cb]
        if a == b:
            continue
        for u, v in nx.minimum_edge_cut(H, a, b):
            cut_edges.add(_canon(u, v))
    return cut_edges


def kfc_v2_fast_scores(
    G: nx.Graph,
    community_weight: float | None = None,
    method: str = "louvain",
    seed: int = _DEFAULT_SEED,
    rep_fraction: float = _DEFAULT_REP_FRACTION,
    min_centers: int = _DEFAULT_MIN_CENTERS,
    mincut: bool = False,
) -> tuple[dict[tuple, float], dict]:
    """
    Cluster-accelerated KFC_v2: the v2 community tier on top of the
    representative-pair current-flow approximation of
    ``prototype_criticality_fast`` (one shared community detection pass).

    ``rep_fraction=1.0`` makes the approximation exact, so the result equals
    ``kfc_v2_scores`` EXACTLY (unit-tested). ``community_weight`` semantics
    match ``kfc_v2_scores`` (None = strict two-tier; 0 = approximate CFEdge).

    ``mincut=True`` adds a THIRD tier: inter-community edges that lie in a
    minimum edge cut between adjacent community centers (max-flow, see
    ``_intercommunity_mincut_edges``) outrank all other inter-community
    edges — ordering: intra < inter < inter-mincut, cf order within tiers.

    Returns ``(scores, info)``; info records ``n_communities``, ``n_reps``,
    ``frac_inter``, ``n_mincut_edges``, ``fallback``.
    """
    H = largest_connected_component(G)
    n = H.number_of_nodes()
    info = {"n_communities": 0, "n_reps": 0, "frac_inter": 0.0,
            "n_mincut_edges": 0, "community_weight": community_weight,
            "method": method, "fallback": False}
    if n < 2 or H.number_of_edges() == 0:
        return {}, info

    parts = detect_communities(H, method=method, seed=seed)
    reps = community_representatives(H, parts, rep_fraction, min_centers)
    info["n_communities"] = len(parts)
    info["n_reps"] = len(reps)
    if len(reps) < 2:
        scores, exact_info = kfc_v2_scores(
            H, community_weight=community_weight, method=method, seed=seed)
        info.update({"frac_inter": exact_info["frac_inter"], "fallback": True})
        return scores, info

    nodes = list(H.nodes())
    idx = {u: k for k, u in enumerate(nodes)}
    M = _grounded_inverse(H, nodes)
    edges = [_canon(u, v) for u, v in H.edges()]
    U = np.array([idx[u] for u, _ in edges])
    V = np.array([idx[v] for _, v in edges])

    r_idx = np.array([idx[v] for v, _ in reps])
    w = np.array([wt for _, wt in reps])
    # Edge-axis chunking: D is E x R float64 and its pairwise reduction makes
    # several same-shaped temporaries, so cap the working set (~1 GB budget).
    # Rows are independent — chunked results are identical to the one-shot.
    chunk = max(1, int(_CHUNK_BUDGET_FLOATS / max(1, len(r_idx))))
    cf = np.empty(len(edges))
    for s in range(0, len(edges), chunk):
        sl = slice(s, min(s + chunk, len(edges)))
        D = M[np.ix_(U[sl], r_idx)] - M[np.ix_(V[sl], r_idx)]
        cf[sl] = _weighted_pairwise_absdiff(D, w)

    if community_weight is not None and float(community_weight) == 0.0:
        return {e: float(cf[k]) for k, e in enumerate(edges)}, info

    membership = {v: ci for ci, comm in enumerate(parts) for v in comm}
    inter = np.array([1.0 if membership[u] != membership[v] else 0.0
                      for u, v in edges])
    info["frac_inter"] = float(inter.mean())

    tier = inter.copy()
    if mincut:
        cut_edges = _intercommunity_mincut_edges(H, parts, membership)
        cutmark = np.array([1.0 if e in cut_edges else 0.0 for e in edges])
        info["n_mincut_edges"] = int(cutmark.sum())
        tier = inter + cutmark * inter    # cut tier only within inter edges

    if community_weight is None:
        vals = cf + float(cf.max()) * tier
    else:
        vals = cf * (1.0 + float(community_weight) * tier)

    return {e: float(vals[k]) for k, e in enumerate(edges)}, info


def rank_kfc_v2_fast(
    G: nx.Graph,
    community_weight: float | None = None,
    method: str = "louvain",
    seed: int = _DEFAULT_SEED,
    rep_fraction: float = _DEFAULT_REP_FRACTION,
    min_centers: int = _DEFAULT_MIN_CENTERS,
    mincut: bool = False,
) -> pd.DataFrame:
    """DataFrame form: columns ``i, j, KFC_v2_fast`` (dismantle reverse=True)."""
    scores, _ = kfc_v2_fast_scores(
        G, community_weight=community_weight, method=method, seed=seed,
        rep_fraction=rep_fraction, min_centers=min_centers, mincut=mincut)
    rows = [{"i": u, "j": v, "KFC_v2_fast": val}
            for (u, v), val in scores.items()]
    return pd.DataFrame(rows)


def rank_kfc_v2(
    G: nx.Graph,
    community_weight: float | None = None,
    method: str = "louvain",
    seed: int = _DEFAULT_SEED,
) -> pd.DataFrame:
    """DataFrame form: columns ``i, j, KFC_v2`` (dismantle reverse=True)."""
    scores, _ = kfc_v2_scores(G, community_weight=community_weight,
                              method=method, seed=seed)
    rows = [{"i": u, "j": v, "KFC_v2": val} for (u, v), val in scores.items()]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    G = nx.karate_club_graph()
    scores, info = kfc_v2_scores(G)
    print(info)
    print(rank_kfc_v2(G).sort_values("KFC_v2", ascending=False)
          .head(8).to_string(index=False))
