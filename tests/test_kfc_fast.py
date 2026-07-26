"""
Correctness tests for src/utils/prototype_criticality_fast.py (cluster-
accelerated KFC). Runnable with pytest OR as a plain script:

    .venv/Scripts/python.exe tests/test_kfc_fast.py
"""
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "utils")))
import prototype_criticality as pc            # noqa: E402
import prototype_criticality_fast as pcf      # noqa: E402
import resilience_utils as ru                 # noqa: E402


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation via pandas (no scipy in this repo)."""
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    return float(j["a"].rank().corr(j["b"].rank()))


def _indexed(df: pd.DataFrame, col: str) -> pd.Series:
    return df.set_index(["i", "j"])[col]


# --------------------------------------------------------------------------- #
# Exactness: full sampling (rep_fraction=1) makes every node a representative  #
# with weight 1, so the weighted sorted identity must recover exact values.    #
# --------------------------------------------------------------------------- #
def test_full_sampling_recovers_exact():
    for G in (nx.karate_club_graph(), nx.path_graph(8)):
        # beta=0 -> pure current flow: must equal fast-exact CFEdge.
        cf = ru.current_flow_edge_betweenness(G)
        fast, info = pcf.kfc_fast_scores(G, beta=0.0, rep_fraction=1.0,
                                         min_centers=1)
        assert not info["fallback"]
        assert set(fast) == set(cf)
        for e in cf:
            assert abs(fast[e] - cf[e]) < 1e-9, f"{e}: {fast[e]} vs {cf[e]}"

        # adaptive beta -> must equal exact KFC (same beta, same fusion).
        ex_scores, ex_beta, ex_mreff = pc.kfc_scores(G)
        fast_b, info_b = pcf.kfc_fast_scores(G, rep_fraction=1.0, min_centers=1)
        assert abs(info_b["beta"] - ex_beta) < 1e-9
        assert abs(info_b["mean_reff"] - ex_mreff) < 1e-9
        for e in ex_scores:
            assert abs(fast_b[e] - ex_scores[e]) < 1e-9


# --------------------------------------------------------------------------- #
# Determinism: identical values across two runs, for both methods.             #
# --------------------------------------------------------------------------- #
def test_determinism_across_runs():
    G = nx.karate_club_graph()
    for meth in ("louvain", "leiden"):
        d1 = rank_sorted(pcf.rank_kfc_fast(G, method=meth, seed=42))
        d2 = rank_sorted(pcf.rank_kfc_fast(G, method=meth, seed=42))
        assert (d1["i"] == d2["i"]).all() and (d1["j"] == d2["j"]).all()
        assert np.allclose(d1["KFC_fast"].values, d2["KFC_fast"].values,
                           rtol=0, atol=0), meth  # bit-identical


def rank_sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["i", "j"], key=lambda s: s.astype(str)).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Approximation quality: high Spearman vs exact KFC on karate (empirically     #
# 0.727 louvain / 0.744 leiden at the default rep_fraction=0.5).               #
# --------------------------------------------------------------------------- #
def test_spearman_vs_exact_karate():
    G = nx.karate_club_graph()
    exact = _indexed(pc.rank_kfc(G), "KFC")
    for meth in ("louvain", "leiden"):
        fast = _indexed(pcf.rank_kfc_fast(G, method=meth), "KFC_fast")
        rho = _spearman(exact, fast)
        assert rho > 0.7, f"{meth}: Spearman {rho:.3f} <= 0.7"


# --------------------------------------------------------------------------- #
# Schema: standard i, j, KFC_fast frame, one row per LCC edge, no NaNs.        #
# --------------------------------------------------------------------------- #
def test_schema_and_integrity():
    G = nx.karate_club_graph()
    df = pcf.rank_kfc_fast(G)
    assert list(df.columns) == ["i", "j", "KFC_fast"]
    assert len(df) == G.number_of_edges()
    assert not df.isnull().any().any()
    assert not (df["i"] == df["j"]).any()
    assert (df["KFC_fast"] > 0).all()          # currents are strictly positive


# --------------------------------------------------------------------------- #
# Disconnected / dirty input: reduced to the LCC (self-loops stripped),        #
# consistent with resilience_utils / prototype_criticality.                    #
# --------------------------------------------------------------------------- #
def test_disconnected_input_uses_lcc():
    G = nx.Graph()
    G.add_edges_from(nx.karate_club_graph().edges())   # LCC (78 edges)
    G.add_edges_from([(100, 101), (101, 102)])         # smaller component
    G.add_edge(0, 0)                                   # self-loop
    df = pcf.rank_kfc_fast(G)
    assert len(df) == 78                               # LCC edges only
    assert not df.isnull().any().any()
    assert 100 not in set(df["i"]) | set(df["j"])

    # Never mutates the input.
    assert G.number_of_edges() == 78 + 2 + 1
    assert G.has_edge(0, 0)


# --------------------------------------------------------------------------- #
# Leiden fallback partition contract: covers all nodes, disjoint, and every    #
# community internally connected (the defining Leiden guarantee).              #
# --------------------------------------------------------------------------- #
def test_leiden_partition_well_connected():
    for G in (nx.karate_club_graph(), nx.les_miserables_graph()):
        H = ru.largest_connected_component(G)
        parts = pcf.detect_communities(H, method="leiden", seed=42)
        seen: set = set()
        for comm in parts:
            assert comm, "empty community"
            assert not (comm & seen), "communities overlap"
            seen |= comm
            assert nx.is_connected(H.subgraph(comm)), \
                "leiden fallback left an internally-disconnected community"
        assert seen == set(H.nodes())

    try:
        pcf.detect_communities(nx.karate_club_graph(), method="walktrap")
        raised = False
    except ValueError:
        raised = True
    assert raised


# --------------------------------------------------------------------------- #
# Representatives: centers are deterministic, quantile spread includes the     #
# community center (top closeness), weights sum to |C| per community.          #
# --------------------------------------------------------------------------- #
def test_representatives_weights_and_centers():
    H = ru.largest_connected_component(nx.karate_club_graph())
    parts = pcf.detect_communities(H, seed=42)
    reps = pcf.community_representatives(H, parts, rep_fraction=0.5, min_centers=3)
    total_w = sum(w for _, w in reps)
    assert abs(total_w - H.number_of_nodes()) < 1e-9   # weights partition N
    rep_nodes = [v for v, _ in reps]
    assert len(rep_nodes) == len(set(rep_nodes))       # no duplicates
    for comm in parts:                                 # center always included
        sub = H.subgraph(comm)
        clo = nx.closeness_centrality(sub)
        center = sorted(clo, key=lambda v: (-clo[v], str(v)))[0]
        assert center in rep_nodes


# --------------------------------------------------------------------------- #
# Tiny graphs: with every node a representative the result is exact; empty     #
# graphs return an empty score dict rather than raising.                       #
# --------------------------------------------------------------------------- #
def test_tiny_graph_exact():
    G = nx.path_graph(2)
    scores, info = pcf.kfc_fast_scores(G)
    exact, _, _ = pc.kfc_scores(G)
    assert set(scores) == set(exact)
    for e in exact:
        assert abs(scores[e] - exact[e]) < 1e-9

    empty, info0 = pcf.kfc_fast_scores(nx.Graph())
    assert empty == {} and info0["n_reps"] == 0


# --------------------------------------------------------------------------- #
# Integration: drop-in for the production run_and_plot driver.                 #
# --------------------------------------------------------------------------- #
def test_run_and_plot_integration_smoke():
    import tempfile
    from network_utils import run_and_plot

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "smoke_kfc_fast.png")
        results = run_and_plot(nx.karate_club_graph(), "kfc_fast_smoke",
                               pcf.rank_kfc_fast, out, reverse=True)
        assert set(results) == {"KFC_fast"}
        auc = results["KFC_fast"]["auc"]
        assert 0.0 <= auc <= 1.0 and np.isfinite(auc)
        assert os.path.exists(out)


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
