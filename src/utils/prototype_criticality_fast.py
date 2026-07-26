"""
PROTOTYPE — Cluster-accelerated Kirchhoff Flow Criticality (``rank_kfc_fast``).

Idea
----
Exact KFC (``prototype_criticality.kfc_scores``) accumulates current-flow
throughput over ALL C(N,2) source-sink pairs via the sorted pairwise-difference
identity (O(E * N log N) after one Laplacian pseudoinverse). This module
approximates that accumulation using a community-stratified set of
REPRESENTATIVE nodes as sources/targets:

1.  Detect communities with Louvain (``nx.community.louvain_communities``,
    fixed ``seed`` for determinism) or a Leiden-style variant (see below).
2.  In each community rank nodes by CLOSENESS CENTRALITY of the community
    subgraph (ties broken by sorted node label — deterministic). The top node
    is the community CENTER; it anchors the representative set. Because
    center-only pairs systematically starve peripheral edges of current,
    additional representatives are taken at evenly spaced quantiles of the
    closeness ordering (so the community's periphery is sampled too), up to
    ``max(min_centers, ceil(rep_fraction * |C|))`` per community. Each
    representative carries weight ``|C| / k_C`` (it stands in for that many
    nodes of its community).
3.  Source/target pairs are ALL unordered pairs of distinct representatives
    (cross- and intra-community), pair (s, t) weighted ``w_s * w_t``. The
    weighted sum of |pairwise potential differences| collapses to the same
    sorted-vector identity as exact CFEdge, restricted to the R representative
    columns: per edge O(R log R) instead of O(N log N), fully vectorised
    across edges. With ``rep_fraction=1.0`` every node is a representative
    with weight 1 and the result equals exact CFEdge EXACTLY (unit-tested).
4.  Fuse with irreplaceability exactly like exact KFC:
    ``KFC_fast(e) = cf_approx(e) * (1/(1-R_eff(e)))**beta``, with the SAME
    adaptive ``beta = clip(1 - mean(R_eff)/DENSITY_THRESH, 0, 1)`` computed
    from the EXACT per-edge effective resistances (O(E) once the inverse
    exists — they are not approximated).

Leiden availability
-------------------
``leidenalg``/``igraph`` are NOT installed in this venv, and networkx 3.6's
``nx.community.leiden_communities`` is a dispatch-only stub (raises
``NotImplementedError`` without a GPU backend). Per project constraints no
heavy dependency is added; ``method='leiden'`` runs a documented FALLBACK:
Louvain seeded partition + the two defining Leiden repairs —
(a) split internally-disconnected communities into connected components
(Leiden's "well-connected communities" guarantee), and (b) one deterministic
greedy local-move refinement sweep (sorted node order, standard unweighted
modularity delta) followed by a re-split. This captures Leiden's quality fix
over Louvain without new dependencies; it is NOT the full Leiden algorithm.

Where the time is actually saved (honest accounting)
----------------------------------------------------
* The dense linear-algebra step is NOT avoided: the KFC fusion term needs
  per-edge effective resistances, which require (a column basis of) the
  Laplacian inverse anyway. We compute ``np.linalg.inv`` of the grounded
  (N-1)x(N-1) Laplacian instead of ``np.linalg.pinv``'s SVD — same O(N^3)
  class, several-fold smaller constant.
* The genuine asymptotic saving is the accumulation: O(E * R log R) vectorised
  (R = representatives, ~rep_fraction*N) versus exact's per-edge Python-loop
  O(E * N log N).
* The new preprocessing cost is Louvain/Leiden + per-community closeness
  (roughly O(E)-ish, but pure-Python constant-heavy). On the small benchmark
  networks (N <= 200) this OVERHEAD EXCEEDS THE SAVING — exact KFC is already
  milliseconds there and ``rank_kfc_fast`` is slower end-to-end. The method
  pays off as N grows (the measured crossover is around N ~ 1000; see
  ``results_comparison/kfc_fast_notes.md``). Rank agreement with exact KFC is
  Spearman ~0.7-0.97 at the default ``rep_fraction=0.5``.

Pure NumPy + NetworkX (no scipy). Never mutates the input graph (all work on
an LCC copy). Returns the standard ``i, j, KFC_fast`` DataFrame — drop-in for
``run_and_plot`` (dismantle ``reverse=True``).
"""
from __future__ import annotations

import math
import os
import sys
from typing import Hashable

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prototype_criticality import _DENOM_FLOOR, adaptive_beta  # noqa: E402
from resilience_utils import (  # noqa: E402
    _canon,
    largest_connected_component,
)

__all__ = [
    "detect_communities",
    "community_representatives",
    "kfc_fast_scores",
    "rank_kfc_fast",
]

_DEFAULT_SEED = 42
_DEFAULT_REP_FRACTION = 0.5
_DEFAULT_MIN_CENTERS = 8
_MODULARITY_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Community detection                                                          #
# --------------------------------------------------------------------------- #
def _split_disconnected(G: nx.Graph, parts: list[set]) -> list[set]:
    """Leiden repair (a): split internally-disconnected communities."""
    out: list[set] = []
    for comm in parts:
        sub = G.subgraph(comm)
        if len(comm) <= 1 or nx.is_connected(sub):
            out.append(set(comm))
        else:
            out.extend(set(c) for c in nx.connected_components(sub))
    return out


def _local_move_refinement(G: nx.Graph, parts: list[set]) -> list[set]:
    """
    Leiden repair (b): one deterministic greedy local-move sweep. Nodes are
    visited in sorted order; a node moves to the adjacent community with the
    largest strictly-positive modularity gain (ties keep the lowest community
    index, i.e. the earliest in sorted order). Standard unweighted delta-Q:
    ``dQ = (k_{v,B} - k_{v,A\\v})/m - k_v*(Sig_B - (Sig_A - k_v))/(2 m^2)``.
    """
    m = G.number_of_edges()
    if m == 0:
        return [set(p) for p in parts]

    membership = {v: ci for ci, comm in enumerate(parts) for v in comm}
    sigma = {ci: float(sum(G.degree(v) for v in comm))
             for ci, comm in enumerate(parts)}

    for v in sorted(G.nodes(), key=str):
        a = membership[v]
        kv = float(G.degree(v))
        k_to: dict[int, float] = {}
        for w in G.neighbors(v):
            if w != v:
                k_to[membership[w]] = k_to.get(membership[w], 0.0) + 1.0
        k_va = k_to.get(a, 0.0)

        best_gain, best_comm = 0.0, a
        for b in sorted(k_to):
            if b == a:
                continue
            gain = ((k_to[b] - k_va) / m
                    - kv * (sigma[b] - (sigma[a] - kv)) / (2.0 * m * m))
            if gain > best_gain + _MODULARITY_EPS:
                best_gain, best_comm = gain, b
        if best_comm != a:
            membership[v] = best_comm
            sigma[a] -= kv
            sigma[best_comm] += kv

    grouped: dict[int, set] = {}
    for v, ci in membership.items():
        grouped.setdefault(ci, set()).add(v)
    return [c for c in grouped.values() if c]


def detect_communities(
    G: nx.Graph, method: str = "louvain", seed: int = _DEFAULT_SEED
) -> list[set]:
    """
    Partition ``G`` into communities (deterministic given ``seed``).

    ``method='louvain'`` — ``nx.community.louvain_communities(seed=seed)``.
    ``method='leiden'``  — Louvain + Leiden-style refinement FALLBACK (module
    docstring): split disconnected communities, one greedy local-move sweep,
    re-split.
    """
    if method not in ("louvain", "leiden"):
        raise ValueError(f"unknown community method {method!r}; "
                         "expected 'louvain' or 'leiden'.")
    parts = [set(c) for c in nx.community.louvain_communities(G, seed=seed)]
    if method == "leiden":
        parts = _split_disconnected(G, parts)
        parts = _local_move_refinement(G, parts)
        parts = _split_disconnected(G, parts)
    # Deterministic community order regardless of set/dict iteration order.
    return sorted(parts, key=lambda c: min(str(v) for v in c))


def community_representatives(
    G: nx.Graph,
    parts: list[set],
    rep_fraction: float = _DEFAULT_REP_FRACTION,
    min_centers: int = _DEFAULT_MIN_CENTERS,
) -> list[tuple[Hashable, float]]:
    """
    Weighted representative nodes per community.

    Nodes of each community are ordered by closeness centrality of the
    community SUBGRAPH (descending; ties by sorted node label). The first node
    is the community CENTER; ``k_C = min(|C|, max(min_centers,
    ceil(rep_fraction*|C|)))`` representatives are picked at evenly spaced
    quantiles of that ordering (always including the center at quantile 0).
    Each carries weight ``|C| / k_C``. Returns ``[(node, weight), ...]``.
    """
    reps: list[tuple[Hashable, float]] = []
    for comm in parts:
        sub = G.subgraph(comm)
        clo = nx.closeness_centrality(sub)
        ordered = sorted(clo, key=lambda v: (-clo[v], str(v)))
        k = min(len(ordered), max(min_centers, math.ceil(rep_fraction * len(ordered))))
        if k == 1:
            chosen = [ordered[0]]
        else:
            qidx = sorted({int(round(q * (len(ordered) - 1)))
                           for q in np.linspace(0.0, 1.0, k)})
            chosen = [ordered[i] for i in qidx]
        w = len(comm) / len(chosen)
        reps.extend((v, w) for v in chosen)
    return reps


# --------------------------------------------------------------------------- #
# Cluster-approximated KFC                                                     #
# --------------------------------------------------------------------------- #
def _grounded_inverse(H: nx.Graph, nodes: list) -> np.ndarray:
    """
    N x N matrix M with the last node grounded (row/col g = 0) and the
    remaining block the inverse of the grounded Laplacian. For any s, t, u, v:
    ``R_eff(u,v) = M[u,u]+M[v,v]-2M[u,v]`` and the pair-(s,t) potential drop
    across edge (u,v) is ``(M[u,s]-M[u,t]) - (M[v,s]-M[v,t])`` — identical to
    the pseudoinverse expressions (the grounding constant cancels), at a
    fraction of ``np.linalg.pinv``'s SVD cost.
    """
    A = nx.to_numpy_array(H, nodelist=nodes, weight=None)
    L = np.diag(A.sum(axis=1)) - A
    n = len(nodes)
    M = np.zeros((n, n))
    M[: n - 1, : n - 1] = np.linalg.inv(L[: n - 1, : n - 1])
    return M


def _weighted_pairwise_absdiff(D: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Per row of ``D`` (edges x reps): ``sum_{a<b} w_a w_b |D[a] - D[b]|``
    divided by the total pair weight — the weighted-mean pair current. Uses
    the sorted prefix-sum identity, vectorised over rows: O(E * R log R).
    """
    order = np.argsort(D, axis=1)
    Ds = np.take_along_axis(D, order, axis=1)
    Ws = w[order]
    cw = np.cumsum(Ws, axis=1)
    below = cw - Ws                       # weight strictly below each entry
    tot = cw[:, -1:]
    acc = (Ws * Ds * below).sum(axis=1) - (Ws * Ds * (tot - cw)).sum(axis=1)
    total_pair_weight = (float(w.sum()) ** 2 - float((w ** 2).sum())) / 2.0
    return acc / total_pair_weight


def kfc_fast_scores(
    G: nx.Graph,
    beta: float | None = None,
    method: str = "louvain",
    seed: int = _DEFAULT_SEED,
    rep_fraction: float = _DEFAULT_REP_FRACTION,
    min_centers: int = _DEFAULT_MIN_CENTERS,
) -> tuple[dict[tuple, float], dict]:
    """
    Cluster-approximated KFC. Returns ``(scores, info)``; ``info`` records
    ``beta``, ``mean_reff``, ``n_communities``, ``n_reps``, ``method`` and
    ``fallback``. Works on the largest connected component (input untouched).
    Falls back to exact ``kfc_scores`` when fewer than 2 representatives exist
    (info["fallback"] = True). ``beta=None`` -> same adaptive exponent as
    exact KFC, from EXACT effective resistances.
    """
    H = largest_connected_component(G)
    n = H.number_of_nodes()
    if n < 2 or H.number_of_edges() == 0:
        return {}, {"beta": 0.0, "mean_reff": 0.0, "n_communities": 0,
                    "n_reps": 0, "method": method, "fallback": False}

    parts = detect_communities(H, method=method, seed=seed)
    reps = community_representatives(H, parts, rep_fraction, min_centers)
    if len(reps) < 2:
        from prototype_criticality import kfc_scores
        scores, b, mreff = kfc_scores(H, beta=beta)
        return scores, {"beta": b, "mean_reff": mreff,
                        "n_communities": len(parts), "n_reps": len(reps),
                        "method": method, "fallback": True}

    nodes = list(H.nodes())
    idx = {u: k for k, u in enumerate(nodes)}
    M = _grounded_inverse(H, nodes)

    edges = [_canon(u, v) for u, v in H.edges()]
    U = np.array([idx[u] for u, _ in edges])
    V = np.array([idx[v] for _, v in edges])

    # Exact effective resistances (O(E)) -> the SAME adaptive beta as exact KFC.
    diag = np.diag(M)
    reff = diag[U] + diag[V] - 2.0 * M[U, V]
    mean_reff = float(np.mean(reff))
    b = adaptive_beta(mean_reff) if beta is None else float(beta)

    # Weighted representative-pair current accumulation (sorted identity).
    r_idx = np.array([idx[v] for v, _ in reps])
    w = np.array([wt for _, wt in reps])
    D = M[np.ix_(U, r_idx)] - M[np.ix_(V, r_idx)]     # edges x reps potentials
    cf_approx = _weighted_pairwise_absdiff(D, w)

    if b == 0.0:
        vals = cf_approx
    else:
        vals = cf_approx * (1.0 / np.maximum(1.0 - reff, _DENOM_FLOOR)) ** b

    scores = {e: float(vals[k]) for k, e in enumerate(edges)}
    info = {"beta": b, "mean_reff": mean_reff, "n_communities": len(parts),
            "n_reps": len(reps), "method": method, "fallback": False}
    return scores, info


def rank_kfc_fast(
    G: nx.Graph,
    beta: float | None = None,
    method: str = "louvain",
    seed: int = _DEFAULT_SEED,
    rep_fraction: float = _DEFAULT_REP_FRACTION,
    min_centers: int = _DEFAULT_MIN_CENTERS,
) -> pd.DataFrame:
    """DataFrame form: columns ``i, j, KFC_fast`` (dismantle reverse=True)."""
    scores, _ = kfc_fast_scores(G, beta=beta, method=method, seed=seed,
                                rep_fraction=rep_fraction,
                                min_centers=min_centers)
    rows = [{"i": u, "j": v, "KFC_fast": val} for (u, v), val in scores.items()]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    G = nx.karate_club_graph()
    for meth in ("louvain", "leiden"):
        scores, info = kfc_fast_scores(G, method=meth)
        print(meth, info)
        print(rank_kfc_fast(G, method=meth).sort_values(
            "KFC_fast", ascending=False).head(5).to_string(index=False))
