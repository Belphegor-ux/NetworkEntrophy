"""
Resilience / criticality utilities for infrastructure network robustness.

Purpose
-------
Identify the structurally critical edges and nodes of a network so they can be
**prioritised for protection, redundancy, and N-1 / N-k contingency planning**.
This is the flow/spectral complement to the topological dismantling metrics in
``network_utils.py``: where those rank edges by neighbourhood/entropy heuristics,
these rank them by explicit max-flow / min-cut membership, electrical
(current-flow) throughput, and single-failure (N-1) impact.

All edge-level ``rank_*`` functions return the standard ``i, j, <metric>``
DataFrame, so they are drop-in compatible with ``network_utils.run_and_plot``
and ``network_utils_iter.run_iterative_benchmark``.

Design notes
------------
* Pure NumPy + NetworkX. The electrical (current-flow) and spectral
  (algebraic-connectivity / Fiedler) metrics are computed directly from the
  Laplacian pseudoinverse / eigendecomposition, so **scipy is not required**.
  For a few-hundred-node grid the dense linear algebra is instantaneous.
* Current-flow metrics use **unit conductances** — a purely topological
  electrical model. A true power-flow criticality analysis would weight each
  line by its admittance (1 / reactance); the OSM-derived extract used here
  does not carry reliable per-line reactance, so unit conductances are the
  honest choice. This is documented rather than silently assumed.
* Everything is deterministic (no RNG) given a fixed node ordering.

Metric summary
--------------
Edge-level (higher = more critical, dismantle reverse=True):
    EBC          shortest-path edge betweenness (topological load proxy)
    CFEdge       current-flow (electrical) edge betweenness — grid-appropriate
    MaxFlowCrit  flow betweenness: unit-capacity max-flow load summed over pairs
    BridgeImpact single-edge N-1 impact: giant-component fraction lost on removal

Node-level:
    degree, NodeBetweenness, CFNode (current-flow node betweenness),
    is_articulation_point, in_min_node_cut

Global (max-flow / min-cut / spectral):
    edge_connectivity, node_connectivity, minimum_edge_cut, minimum_node_cut,
    global_min_cut (Stoer-Wagner), backbone (2-core) min-cut,
    algebraic_connectivity (Fiedler value), Fiedler partition, bridge/AP counts
"""
from __future__ import annotations

import warnings
from typing import Any, Hashable

import networkx as nx
import numpy as np
import pandas as pd

__all__ = [
    "largest_connected_component",
    "current_flow_betweenness",
    "current_flow_edge_betweenness",
    "rank_edge_betweenness",
    "rank_current_flow_edge",
    "rank_effective_resistance",
    "rank_maxflow_criticality",
    "rank_bridge_impact",
    "combined_edge_criticality",
    "node_resilience_table",
    "algebraic_connectivity",
    "fiedler_partition",
    "global_resilience_summary",
]

# Above this number of unordered node pairs the exact all-pairs max-flow load
# becomes expensive; we then sample and LOG the sample size (never silently cap).
_MAXFLOW_EXACT_PAIR_LIMIT = 20_000


def _canon(u: Hashable, v: Hashable) -> tuple:
    """Canonical, type-safe ordering of an undirected edge's endpoints."""
    return (u, v) if str(u) <= str(v) else (v, u)


def largest_connected_component(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component as a new graph (self-loops removed)."""
    H = G.copy()
    H.remove_edges_from(nx.selfloop_edges(H))
    if H.number_of_nodes() == 0 or nx.is_connected(H):
        return H
    lcc_nodes = max(nx.connected_components(H), key=len)
    return H.subgraph(lcc_nodes).copy()


# --------------------------------------------------------------------------- #
# Electrical (current-flow) betweenness — NumPy Laplacian pseudoinverse        #
# --------------------------------------------------------------------------- #
def _laplacian_pinv(G: nx.Graph, nodelist: list) -> np.ndarray:
    """Moore-Penrose pseudoinverse of the (unit-conductance) Laplacian."""
    A = nx.to_numpy_array(G, nodelist=nodelist, weight=None)
    L = np.diag(A.sum(axis=1)) - A
    return np.linalg.pinv(L)


def current_flow_betweenness(
    G: nx.Graph,
) -> tuple[dict[tuple, float], dict[Hashable, float]]:
    """
    Current-flow (electrical) betweenness for every edge and node.

    Models the graph as a resistor network with unit conductance on each edge.
    For a unit current injected at ``s`` and extracted at ``t`` the potential
    vector is ``phi = L+ (e_s - e_t)`` and the current on edge (u, v) is
    ``phi[u] - phi[v]``. Betweenness accumulates the absolute current over all
    unordered source-sink pairs (Newman 2005), normalised to a per-pair mean.

    Unlike shortest-path betweenness, current-flow spreads load across *all*
    paths in proportion to how well they conduct — the physically meaningful
    notion of "critical corridor" for an electrical grid.

    Requires a connected graph. Returns ``(edge_cfb, node_cfb)`` dicts keyed by
    canonical edge tuple and node label respectively.
    """
    if G.number_of_nodes() < 2:
        return {}, {n: 0.0 for n in G.nodes()}
    if not nx.is_connected(G):
        raise ValueError(
            "current_flow_betweenness requires a connected graph; "
            "call largest_connected_component(G) first."
        )

    nodes = list(G.nodes())
    idx = {u: k for k, u in enumerate(nodes)}
    n = len(nodes)
    Lp = _laplacian_pinv(G, nodes)

    edges = [_canon(u, v) for u, v in G.edges()]
    U = np.array([idx[u] for u, v in edges])
    V = np.array([idx[v] for u, v in edges])

    edge_acc = np.zeros(len(edges))
    node_acc = np.zeros(n)

    for a in range(n):
        col_a = Lp[:, a]
        for b in range(a + 1, n):
            phi = col_a - Lp[:, b]
            ecur = np.abs(phi[U] - phi[V])            # |current| on each edge
            edge_acc += ecur
            # Node throughput = half the sum of incident edge currents,
            # except source/target carry the full injected unit current.
            inc = (np.bincount(U, ecur, minlength=n)
                   + np.bincount(V, ecur, minlength=n))
            tau = 0.5 * inc
            tau[a] = 1.0
            tau[b] = 1.0
            node_acc += tau

    n_pairs = n * (n - 1) / 2.0
    edge_cfb = {e: float(edge_acc[k] / n_pairs) for k, e in enumerate(edges)}
    node_cfb = {nodes[k]: float(node_acc[k] / n_pairs) for k in range(n)}
    return edge_cfb, node_cfb


# --------------------------------------------------------------------------- #
# Edge-level rank_* functions (i, j, <metric>) — drop-in for run_and_plot      #
# --------------------------------------------------------------------------- #
def rank_edge_betweenness(G: nx.Graph) -> pd.DataFrame:
    """Shortest-path edge betweenness centrality (topological load proxy)."""
    ebc = nx.edge_betweenness_centrality(G, normalized=True)
    rows = []
    for (u, v), val in ebc.items():
        cu, cv = _canon(u, v)
        rows.append({"i": cu, "j": cv, "EBC": float(val)})
    return pd.DataFrame(rows)


def current_flow_edge_betweenness(G: nx.Graph) -> dict[tuple, float]:
    """
    Current-flow (electrical) edge betweenness — fast exact edge-only path.

    Identical values to the edge dict of ``current_flow_betweenness`` but without
    the O(N^2 * E) all-pairs accumulation. For edge (u, v) let
    ``d = L+[u,:] - L+[v,:]``; the edge's summed absolute current over all pairs
    is ``sum_{s<t} |d[s] - d[t]|``, which for sorted ``d`` equals
    ``sum_i (2i - (n-1)) * d_sorted[i]``. That collapses the per-edge cost to
    O(N log N), giving overall **O(N^3 + E*N log N)** — the same best-in-class
    effectiveness at a fraction of the runtime of the naive accumulation.
    """
    H = largest_connected_component(G)
    if H.number_of_nodes() < 2:
        return {}
    if not nx.is_connected(H):
        raise ValueError("current_flow_edge_betweenness requires a connected graph.")
    nodes = list(H.nodes())
    idx = {u: k for k, u in enumerate(nodes)}
    n = len(nodes)
    Lp = _laplacian_pinv(H, nodes)
    coeff = 2.0 * np.arange(n) - (n - 1)   # weights for the ascending-sorted d
    n_pairs = n * (n - 1) / 2.0
    out: dict[tuple, float] = {}
    for u, v in H.edges():
        d = np.sort(Lp[idx[u]] - Lp[idx[v]])
        out[_canon(u, v)] = float(coeff @ d / n_pairs)
    return out


def rank_current_flow_edge(G: nx.Graph) -> pd.DataFrame:
    """Current-flow (electrical) edge betweenness — the grid-appropriate load."""
    edge_cfb = current_flow_edge_betweenness(G)
    rows = [{"i": u, "j": v, "CFEdge": val} for (u, v), val in edge_cfb.items()]
    return pd.DataFrame(rows)


def rank_effective_resistance(G: nx.Graph) -> pd.DataFrame:
    """
    Edge effective-resistance criticality (``EffRes``) — irreplaceability.

    For edge (u, v), ``R_eff = L+[u,u] + L+[v,v] - 2 L+[u,v]`` (u,v indexed into
    the Laplacian pseudoinverse). Electrically it is the resistance between the
    endpoints through the *whole* network: a bridge has R_eff = 1 (no parallel
    path), an edge with many redundant paths has R_eff -> 0. High R_eff = few
    alternatives = critical (dismantle ``reverse=True``).

    Efficiency: this is the flow/spectral criticality that costs only **O(E)**
    after a single Laplacian pseudoinverse, versus the **O(N^2 * E)** all-pairs
    accumulation of current-flow betweenness (``rank_current_flow_edge``) and the
    all-pairs max-flow of ``rank_maxflow_criticality``. It also admits a
    near-linear-time (1 +/- eps) Spielman-Srivastava random-projection
    approximation for grids too large for a dense solve — which the two
    accumulation-based metrics do not.
    """
    H = largest_connected_component(G)
    nodes = list(H.nodes())
    idx = {u: k for k, u in enumerate(nodes)}
    Lp = _laplacian_pinv(H, nodes)
    rows = []
    for u, v in H.edges():
        a, b = idx[u], idx[v]
        r_eff = Lp[a, a] + Lp[b, b] - 2.0 * Lp[a, b]
        cu, cv = _canon(u, v)
        rows.append({"i": cu, "j": cv, "EffRes": float(r_eff)})
    return pd.DataFrame(rows)


def rank_maxflow_criticality(
    G: nx.Graph, max_pairs: int = _MAXFLOW_EXACT_PAIR_LIMIT, seed: int = 42
) -> pd.DataFrame:
    """
    Min-cut membership criticality via max-flow (the user's mini-max-flow ask).

    For each source-sink pair a *minimum edge cut* is computed (max-flow /
    min-cut theorem, unit capacities) and every edge on that cut is tallied.
    ``MinCutCrit(e)`` is the fraction of pairs for which ``e`` lies on the
    minimum cut — i.e. how often removing ``e`` is part of the cheapest way to
    sever a source from a sink. A bridge lies on the min-cut of *every* pair it
    separates, so it dominates; this is precisely "which edges are the network's
    bottlenecks", unlike flow *load* which concentrates on hub-incident edges.

    Caveat: a minimum cut need not be unique, so which cut is returned is
    algorithm-dependent; the resulting ranking is robust and, for genuine
    single-edge bottlenecks (bridges), unambiguous.

    Exact over all pairs when C(n, 2) <= ``max_pairs``; otherwise a uniform
    random sample of ``max_pairs`` pairs is used and the sample size is logged
    (never silently truncated).
    """
    H = largest_connected_component(G)
    nodes = list(H.nodes())
    n = len(nodes)
    edges = [_canon(u, v) for u, v in H.edges()]
    acc = {e: 0.0 for e in edges}

    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        pairs = [(nodes[a], nodes[b]) for a in range(n) for b in range(a + 1, n)]
        n_used = total_pairs
    else:
        rng = np.random.default_rng(seed)
        chosen: set[tuple[int, int]] = set()
        while len(chosen) < max_pairs:
            a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
            if a != b:
                chosen.add((min(a, b), max(a, b)))
        pairs = [(nodes[a], nodes[b]) for a, b in chosen]
        n_used = len(pairs)
        print(
            f"  [min-cut] {total_pairs} pairs > limit {max_pairs}; "
            f"sampling {n_used} pairs (seed={seed}). Values are estimates."
        )

    for s, t in pairs:
        for (u, v) in nx.minimum_edge_cut(H, s, t):
            acc[_canon(u, v)] += 1.0

    rows = [
        {"i": u, "j": v, "MinCutCrit": acc[(u, v)] / n_used}
        for (u, v) in edges
    ]
    return pd.DataFrame(rows)


def rank_bridge_impact(G: nx.Graph) -> pd.DataFrame:
    """
    Single-edge N-1 impact: fraction of the giant component lost when this one
    edge is removed. Non-bridges score 0 (the component stays connected); a
    bridge scores (size of the smaller severed side) / N. This directly ranks
    the single points of failure by how catastrophic their loss would be.
    """
    H = largest_connected_component(G)
    N = H.number_of_nodes()
    base = 1.0  # H is connected by construction
    bridge_set = {_canon(u, v) for u, v in nx.bridges(H)}

    rows = []
    for u, v in H.edges():
        e = _canon(u, v)
        if e in bridge_set:
            work = H.copy()
            work.remove_edge(*e)
            gc = max(len(c) for c in nx.connected_components(work)) / N
            impact = base - gc
        else:
            impact = 0.0
        rows.append({"i": e[0], "j": e[1], "BridgeImpact": float(impact)})
    return pd.DataFrame(rows)


def combined_edge_criticality(G: nx.Graph, **maxflow_kwargs: Any) -> pd.DataFrame:
    """Merge all edge-level resilience metrics into one ``i, j, ...`` frame.

    Reduces to the largest connected component *once* up front so all four
    metrics are computed over the same edge set. (``rank_edge_betweenness``
    would otherwise score full-graph edges while the flow/spectral rankers use
    the LCC, leaving NaN rows for non-LCC edges after the merge.) The inner
    merges then operate on a single shared, canonical edge set.
    """
    H = largest_connected_component(G)
    df = rank_edge_betweenness(H)
    for ranker in (rank_current_flow_edge, rank_bridge_impact):
        df = df.merge(ranker(H), on=["i", "j"], how="inner")
    df = df.merge(rank_maxflow_criticality(H, **maxflow_kwargs),
                  on=["i", "j"], how="inner")
    return df


# --------------------------------------------------------------------------- #
# Node-level criticality                                                        #
# --------------------------------------------------------------------------- #
def node_resilience_table(G: nx.Graph) -> pd.DataFrame:
    """
    Per-node criticality table for identifying the substations/junctions whose
    loss most degrades connectivity (candidates for hardening / backup).

    Columns: node, degree, NodeBetweenness, CFNode (current-flow node
    betweenness), is_articulation_point, in_min_node_cut.
    """
    H = largest_connected_component(G)
    betw = nx.betweenness_centrality(H, normalized=True)
    _, cf_node = current_flow_betweenness(H)
    aps = set(nx.articulation_points(H))
    try:
        min_node_cut = set(nx.minimum_node_cut(H))
    except nx.NetworkXError:
        # Only NetworkX's own "no cut exists" style errors degrade gracefully;
        # programming errors (NameError/AttributeError) must still surface.
        min_node_cut = set()

    rows = []
    for node in H.nodes():
        rows.append({
            "node": node,
            "degree": H.degree(node),
            "NodeBetweenness": float(betw.get(node, 0.0)),
            "CFNode": float(cf_node.get(node, 0.0)),
            "is_articulation_point": node in aps,
            "in_min_node_cut": node in min_node_cut,
        })
    df = pd.DataFrame(rows)
    return df.sort_values("CFNode", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# Spectral robustness (NumPy — no scipy)                                        #
# --------------------------------------------------------------------------- #
def _laplacian_spectrum(G: nx.Graph, nodelist: list) -> tuple[np.ndarray, np.ndarray]:
    A = nx.to_numpy_array(G, nodelist=nodelist, weight=None)
    L = np.diag(A.sum(axis=1)) - A
    vals, vecs = np.linalg.eigh(L)  # symmetric → ascending real eigenvalues
    return vals, vecs


def algebraic_connectivity(G: nx.Graph) -> float:
    """
    Second-smallest Laplacian eigenvalue (Fiedler value) of the LCC — a global
    robustness index. Larger = harder to fragment; a value near 0 means the
    network is close to disconnection.
    """
    H = largest_connected_component(G)
    if H.number_of_nodes() < 2:
        return 0.0
    vals, _ = _laplacian_spectrum(H, list(H.nodes()))
    return float(vals[1])


def fiedler_partition(G: nx.Graph) -> tuple[list, list]:
    """
    Spectral bisection from the sign of the Fiedler vector — the network's
    "weakest" large-scale cut (the two halves that are hardest to keep joined).
    """
    H = largest_connected_component(G)
    nodes = list(H.nodes())
    if len(nodes) < 2:
        return nodes, []
    _, vecs = _laplacian_spectrum(H, nodes)
    fied = vecs[:, 1]
    side_a = [nodes[k] for k in range(len(nodes)) if fied[k] >= 0]
    side_b = [nodes[k] for k in range(len(nodes)) if fied[k] < 0]
    return side_a, side_b


def _unit_min_cut(H: nx.Graph):
    """Stoer-Wagner global min-cut with forced unit edge weights.

    ``nx.stoer_wagner`` defaults to ``weight='weight'``; forcing unit weights
    keeps the global min-cut consistent with the unit-capacity
    ``edge_connectivity`` and the module's unit-conductance model even when the
    input graph happens to carry a ``weight`` edge attribute.
    """
    U = nx.Graph()
    U.add_nodes_from(H.nodes())
    U.add_edges_from(((u, v) for u, v in H.edges()), weight=1)
    return nx.stoer_wagner(U)


def _two_core_backbone(G: nx.Graph) -> nx.Graph:
    """Largest connected component of the 2-core (stub/pendant lines pruned)."""
    core = nx.k_core(G, k=2)
    if core.number_of_nodes() == 0:
        return core
    return largest_connected_component(core)


def global_resilience_summary(G: nx.Graph) -> dict[str, Any]:
    """
    Whole-network min-cut / max-flow / spectral robustness summary.

    Reports both the raw global min-cut (often dominated by pendant stubs in
    real extracts) and the min-cut of the 2-core *backbone*, which is the
    meaningful structural bottleneck once dead-end lines are pruned.
    """
    H = largest_connected_component(G)
    summary: dict[str, Any] = {
        "n_nodes": H.number_of_nodes(),
        "n_edges": H.number_of_edges(),
    }

    summary["edge_connectivity"] = int(nx.edge_connectivity(H))
    summary["node_connectivity"] = int(nx.node_connectivity(H))
    summary["minimum_edge_cut"] = [list(_canon(u, v)) for u, v in nx.minimum_edge_cut(H)]
    try:
        summary["minimum_node_cut"] = sorted(nx.minimum_node_cut(H), key=str)
    except Exception:
        summary["minimum_node_cut"] = []

    cut_value, (part_a, part_b) = _unit_min_cut(H)
    summary["global_min_cut_value"] = int(cut_value)
    summary["global_min_cut_partition_sizes"] = [len(part_a), len(part_b)]

    backbone = _two_core_backbone(H)
    if backbone.number_of_nodes() >= 2 and nx.is_connected(backbone):
        b_val, (b_a, b_b) = _unit_min_cut(backbone)
        summary["backbone_n_nodes"] = backbone.number_of_nodes()
        summary["backbone_n_edges"] = backbone.number_of_edges()
        summary["backbone_min_cut_value"] = int(b_val)
        summary["backbone_min_cut_partition_sizes"] = [len(b_a), len(b_b)]
        summary["backbone_edge_connectivity"] = int(nx.edge_connectivity(backbone))
    else:
        summary["backbone_n_nodes"] = backbone.number_of_nodes()
        summary["backbone_min_cut_value"] = None

    summary["algebraic_connectivity"] = algebraic_connectivity(H)
    side_a, side_b = fiedler_partition(H)
    summary["fiedler_partition_sizes"] = [len(side_a), len(side_b)]

    summary["n_bridges"] = len(list(nx.bridges(H)))
    summary["n_articulation_points"] = len(list(nx.articulation_points(H)))
    try:
        summary["diameter"] = int(nx.diameter(H))
    except Exception:
        summary["diameter"] = None

    return summary


if __name__ == "__main__":
    warnings.warn(
        "resilience_utils is a shared library module. Import its rank_* / "
        "global_resilience_summary functions from an analysis driver such as "
        "PowerGrid_City/run_resilience_analysis.py.",
        stacklevel=1,
    )
