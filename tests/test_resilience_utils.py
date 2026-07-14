"""
Correctness tests for src/utils/resilience_utils.py on graphs with known
ground-truth criticality structure. Runnable with pytest OR as a plain script:

    .venv/Scripts/python.exe tests/test_resilience_utils.py
"""
import os
import sys

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "utils")))
import resilience_utils as ru  # noqa: E402


def _col(df, i, j, col):
    """Fetch a metric value for canonical edge (i, j) regardless of stored order."""
    ci, cj = ru._canon(i, j)
    row = df[(df["i"] == ci) & (df["j"] == cj)]
    assert len(row) == 1, f"edge {(ci, cj)} not uniquely present"
    return float(row.iloc[0][col])


# --------------------------------------------------------------------------- #
# Barbell: two K5 cliques joined by a single bridge edge (4, 5).                #
# The bridge must dominate every criticality metric.                           #
# --------------------------------------------------------------------------- #
def test_barbell_bridge_is_dominant():
    G = nx.barbell_graph(5, 0)          # bridge is (4, 5)
    bridge = ru._canon(4, 5)

    summary = ru.global_resilience_summary(G)
    assert summary["edge_connectivity"] == 1
    assert summary["node_connectivity"] == 1
    assert summary["global_min_cut_value"] == 1
    assert summary["n_bridges"] == 1
    assert sorted(summary["minimum_edge_cut"]) == [list(bridge)]

    # BridgeImpact: removing (4,5) severs 5 of 10 nodes -> 0.5; all others 0.
    bi = ru.rank_bridge_impact(G)
    assert abs(_col(bi, 4, 5, "BridgeImpact") - 0.5) < 1e-9
    others = bi[~((bi["i"] == bridge[0]) & (bi["j"] == bridge[1]))]
    assert (others["BridgeImpact"].abs() < 1e-9).all()

    # Current-flow edge betweenness: bridge carries all cross-clique current.
    cf = ru.rank_current_flow_edge(G)
    assert _col(cf, 4, 5, "CFEdge") == cf["CFEdge"].max()

    # Min-cut membership: bridge lies on the min-cut of every cross-clique pair.
    mf = ru.rank_maxflow_criticality(G)
    assert _col(mf, 4, 5, "MinCutCrit") == mf["MinCutCrit"].max()

    # Articulation points are exactly the two bridge endpoints.
    nt = ru.node_resilience_table(G)
    aps = set(nt[nt["is_articulation_point"]]["node"])
    assert aps == {4, 5}


# --------------------------------------------------------------------------- #
# Cycle C6: 2-edge-connected, no bridges/APs, fully symmetric.                  #
# --------------------------------------------------------------------------- #
def test_cycle_is_robust_and_symmetric():
    G = nx.cycle_graph(6)
    summary = ru.global_resilience_summary(G)
    assert summary["edge_connectivity"] == 2
    assert summary["node_connectivity"] == 2
    assert summary["n_bridges"] == 0
    assert summary["n_articulation_points"] == 0

    bi = ru.rank_bridge_impact(G)
    assert (bi["BridgeImpact"].abs() < 1e-9).all()      # no single edge disconnects

    cf = ru.rank_current_flow_edge(G)
    assert cf["CFEdge"].std() < 1e-9                    # all edges equivalent by symmetry


# --------------------------------------------------------------------------- #
# Path P5: every edge is a bridge; centre carries most flow; ends the least.   #
# --------------------------------------------------------------------------- #
def test_path_center_is_most_critical():
    G = nx.path_graph(5)                # 0-1-2-3-4
    summary = ru.global_resilience_summary(G)
    assert summary["n_bridges"] == 4
    assert summary["n_articulation_points"] == 3        # nodes 1, 2, 3

    cf = ru.rank_current_flow_edge(G)
    # Central edges (1,2)/(2,3) carry more current than the end edges (0,1)/(3,4).
    assert _col(cf, 1, 2, "CFEdge") > _col(cf, 0, 1, "CFEdge")
    assert _col(cf, 2, 3, "CFEdge") > _col(cf, 3, 4, "CFEdge")

    # BridgeImpact: end edge (0,1) severs only node 0 -> 1/5; centre severs 2/5.
    bi = ru.rank_bridge_impact(G)
    assert abs(_col(bi, 0, 1, "BridgeImpact") - 0.2) < 1e-9
    assert abs(_col(bi, 1, 2, "BridgeImpact") - 0.4) < 1e-9

    # Node betweenness: middle node 2 is the most central.
    nt = ru.node_resilience_table(G)
    top_node = nt.sort_values("NodeBetweenness", ascending=False).iloc[0]["node"]
    assert top_node == 2


# --------------------------------------------------------------------------- #
# Schema + integrity: rank_* produce the standard i,j frame with no NaNs.       #
# --------------------------------------------------------------------------- #
def test_schema_and_integrity():
    G = nx.karate_club_graph()
    for ranker in (ru.rank_edge_betweenness, ru.rank_current_flow_edge,
                   ru.rank_maxflow_criticality, ru.rank_bridge_impact):
        df = ranker(G)
        assert list(df.columns[:2]) == ["i", "j"]
        assert len(df) == G.number_of_edges()
        assert not df.isnull().any().any()
        assert not (df["i"] == df["j"]).any()

    combined = ru.combined_edge_criticality(G)
    assert len(combined) == G.number_of_edges()
    assert not combined.isnull().any().any()

    # Fiedler value of a connected graph is strictly positive.
    assert ru.algebraic_connectivity(G) > 1e-9


def test_current_flow_equals_pair_separation_on_tree():
    # On a tree every source-sink pair's unit current flows entirely along the
    # unique path, so current-flow betweenness of an edge equals the number of
    # node pairs it separates = product of the two subtree sizes. This checks
    # the NumPy Laplacian-pseudoinverse numerics against a closed form.
    T = nx.balanced_tree(2, 3)              # deterministic 15-node binary tree
    N = T.number_of_nodes()
    n_pairs = N * (N - 1) / 2.0
    cf = ru.rank_current_flow_edge(T)
    for _, r in cf.iterrows():
        work = T.copy()
        work.remove_edge(r["i"], r["j"])
        side = len(next(iter(nx.connected_components(work))))
        expected = side * (N - side) / n_pairs
        assert abs(r["CFEdge"] - expected) < 1e-9, \
            f"edge {(r['i'], r['j'])}: {r['CFEdge']} vs analytical {expected}"


def test_current_flow_node_equals_path_membership_on_tree():
    # Node-level mirror of the edge closed form: on a tree, CFNode(v) = (number
    # of unordered pairs whose unique path contains v, endpoints included)/n_pairs.
    # Pins the hand-rolled tau numerics (bincount + endpoint overrides).
    T = nx.balanced_tree(2, 3)
    N = T.number_of_nodes()
    n_pairs = N * (N - 1) / 2.0
    _, cfn = ru.current_flow_betweenness(T)
    nodes = list(T.nodes())
    for v in nodes:
        cnt = sum(1 for ia, a in enumerate(nodes) for b in nodes[ia + 1:]
                  if v in nx.shortest_path(T, a, b))
        assert abs(cfn[v] - cnt / n_pairs) < 1e-9, f"CFNode[{v}]={cfn[v]} vs {cnt / n_pairs}"

    _, cfn3 = ru.current_flow_betweenness(nx.path_graph(3))
    assert abs(cfn3[1] - 1.0) < 1e-9          # middle node
    assert abs(cfn3[0] - 2.0 / 3.0) < 1e-9    # end node

    # Report-facing sort order: on the barbell the top-2 CFNode rows are the
    # bridge endpoints {4, 5}.
    nt = ru.node_resilience_table(nx.barbell_graph(5, 0))
    assert set(nt.head(2)["node"]) == {4, 5}


def test_fiedler_partition_barbell():
    G = nx.barbell_graph(5, 0)
    side_a, side_b = ru.fiedler_partition(G)
    assert {frozenset(side_a), frozenset(side_b)} == \
        {frozenset(range(5)), frozenset(range(5, 10))}
    summary = ru.global_resilience_summary(G)
    assert sorted(summary["fiedler_partition_sizes"]) == [5, 5]


def test_algebraic_connectivity_closed_forms():
    # Exact Fiedler values on graphs with known Laplacian spectra. The path is
    # essential — cycle/complete spectra are degenerate (vals[1]==vals[2]) and
    # would not distinguish a vals[2] off-by-one.
    assert abs(ru.algebraic_connectivity(nx.cycle_graph(6)) - 1.0) < 1e-9
    assert abs(ru.algebraic_connectivity(nx.complete_graph(5)) - 5.0) < 1e-9
    assert abs(ru.algebraic_connectivity(nx.path_graph(5))
               - 2 * (1 - np.cos(np.pi / 5))) < 1e-9
    assert abs(ru.algebraic_connectivity(nx.karate_club_graph())
               - 0.4685252267013929) < 1e-6


def test_edge_betweenness_values():
    # Normalized tree edge betweenness = (separated pairs)/C(N,2).
    eb = ru.rank_edge_betweenness(nx.path_graph(5))
    assert abs(_col(eb, 0, 1, "EBC") - 0.4) < 1e-9
    assert abs(_col(eb, 1, 2, "EBC") - 0.6) < 1e-9
    # The bridge is the max-betweenness edge of the barbell.
    ebb = ru.rank_edge_betweenness(nx.barbell_graph(5, 0))
    assert _col(ebb, 4, 5, "EBC") == ebb["EBC"].max()


def test_mincut_membership_value_barbell():
    # Exact value + strict dominance (not the vacuous "member == column max").
    mf = ru.rank_maxflow_criticality(nx.barbell_graph(5, 0))
    bridge = ru._canon(4, 5)
    mcc = _col(mf, 4, 5, "MinCutCrit")
    assert abs(mcc - 25.0 / 45.0) < 1e-9       # bridge on the unique min cut of all 25 cross pairs
    others = mf[~((mf["i"] == bridge[0]) & (mf["j"] == bridge[1]))]
    assert (others["MinCutCrit"] < mcc).all()


def test_maxflow_sampling_branch_deterministic():
    # Force the sampling branch (45 pairs > max_pairs=20) and pin its contract.
    G = nx.barbell_graph(5, 0)
    m1 = ru.rank_maxflow_criticality(G, max_pairs=20, seed=7)
    m2 = ru.rank_maxflow_criticality(G, max_pairs=20, seed=7)
    a = m1.sort_values(["i", "j"]).reset_index(drop=True)
    b = m2.sort_values(["i", "j"]).reset_index(drop=True)
    assert (a["MinCutCrit"].values == b["MinCutCrit"].values).all()   # seed determinism
    assert len(m1) == G.number_of_edges() and not m1.isnull().any().any()
    assert m1["MinCutCrit"].max() <= 1.0 + 1e-12                      # normalized by n_used
    assert _col(m1, 4, 5, "MinCutCrit") == m1["MinCutCrit"].max()     # bridge still dominates


def test_min_node_cut_barbell():
    nt = ru.node_resilience_table(nx.barbell_graph(5, 0))
    cut_nodes = set(nt[nt["in_min_node_cut"]]["node"])
    # Minimum node cut is a single bridge endpoint; {4} and {5} are both valid.
    assert len(cut_nodes) == 1 and cut_nodes <= {4, 5}


def test_dirty_graph_lcc_and_disconnected():
    # Two unequal components + a self-loop exercise the LCC-selection and
    # self-loop-stripping paths and the disconnected-input contract.
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])          # triangle (LCC)
    G.add_edges_from([(10, 11)])                        # smaller component
    G.add_edge(0, 0)                                    # self-loop
    H = ru.largest_connected_component(G)
    assert set(H.nodes()) == {0, 1, 2}
    assert H.number_of_edges() == 3                     # self-loop stripped

    # current_flow_betweenness refuses disconnected input rather than guessing.
    try:
        ru.current_flow_betweenness(G)
        raised = False
    except ValueError:
        raised = True
    assert raised

    # combined_edge_criticality reduces to the LCC once -> no NaN rows.
    combined = ru.combined_edge_criticality(G)
    assert len(combined) == 3
    assert not combined.isnull().any().any()


def test_fast_current_flow_edge_matches_reference():
    # The O(E*N log N) sorted-identity edge path must equal the edge dict of the
    # O(N^2*E) all-pairs current_flow_betweenness, exactly (same metric).
    for G in (nx.karate_club_graph(), nx.les_miserables_graph(), nx.path_graph(7)):
        ref_edges, _ = ru.current_flow_betweenness(ru.largest_connected_component(G))
        fast = ru.current_flow_edge_betweenness(G)
        assert set(ref_edges) == set(fast)
        for e in ref_edges:
            assert abs(ref_edges[e] - fast[e]) < 1e-9, f"{e}: {ref_edges[e]} vs {fast[e]}"


def test_effective_resistance_closed_forms():
    # A bridge has effective resistance exactly 1 (no parallel path); it is also
    # the max-EffRes edge of the barbell.
    er = ru.rank_effective_resistance(nx.barbell_graph(5, 0))
    assert abs(_col(er, 4, 5, "EffRes") - 1.0) < 1e-9
    assert _col(er, 4, 5, "EffRes") == er["EffRes"].max()

    # Every edge of a cycle C_n has effective resistance (n-1)/n (one edge in
    # series with the length-(n-1) parallel path).
    erc = ru.rank_effective_resistance(nx.cycle_graph(6))
    assert (abs(erc["EffRes"] - 5.0 / 6.0) < 1e-9).all()


def test_global_min_cut_ignores_edge_weights():
    # Stoer-Wagner must use unit capacities regardless of any 'weight' attribute,
    # staying consistent with edge_connectivity and the unit-conductance model.
    G = nx.cycle_graph(6)
    for u, v in G.edges():
        G[u][v]["weight"] = 1000.0
    summary = ru.global_resilience_summary(G)
    assert summary["edge_connectivity"] == 2
    assert summary["global_min_cut_value"] == 2       # not 2000 (weighted)


def test_run_and_plot_integration_smoke():
    # The production path: feed the 4-metric combined frame through the real
    # run_and_plot / NetworkDismantler used by the Tokyo driver.
    import tempfile
    from network_utils import run_and_plot

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "smoke.png")
        results = run_and_plot(nx.karate_club_graph(), "smoke",
                               ru.combined_edge_criticality, out, reverse=True)
        assert set(results) == {"EBC", "CFEdge", "BridgeImpact", "MinCutCrit"}
        for m, res in results.items():
            assert 0.0 <= res["auc"] <= 1.0 and np.isfinite(res["auc"]), m
        assert os.path.exists(out)


def test_kfc_generalizes_current_flow():
    # PROTOTYPE: adaptive Kirchhoff Flow Criticality. beta=0 must recover CFEdge
    # exactly (strict generalisation, no regression on tree-like graphs), and the
    # adaptive beta must be 0 for a tree-like graph, >0 for a dense one.
    import prototype_criticality as pc

    G = nx.path_graph(8)  # tree -> all edges R_eff=1 -> mean_reff high -> beta 0
    cf = ru.current_flow_edge_betweenness(G)
    kfc0, beta, mean_reff = pc.kfc_scores(G, beta=0.0)
    assert set(cf) == set(kfc0)
    for e in cf:
        assert abs(cf[e] - kfc0[e]) < 1e-9

    _, beta_tree, mrf_tree = pc.kfc_scores(nx.path_graph(8))
    assert beta_tree == 0.0 and mrf_tree > pc.DENSITY_THRESH

    # K_n has edge effective resistance 2/n, so a large complete graph is dense
    # enough (mean R_eff = 2/n < DENSITY_THRESH) to trigger amplification.
    _, beta_dense, mrf_dense = pc.kfc_scores(nx.complete_graph(20))
    assert mrf_dense < pc.DENSITY_THRESH and beta_dense > 0.0

    df = pc.rank_kfc(nx.karate_club_graph())
    assert list(df.columns) == ["i", "j", "KFC"] and not df.isnull().any().any()


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
