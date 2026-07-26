"""
Correctness tests for src/utils/prototype_criticality_v2.py (generalized
KFC_v2). Runnable with pytest OR as a plain script:

    .venv/Scripts/python.exe tests/test_kfc_v2.py
"""
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "utils")))
import prototype_criticality_v2 as pcv2       # noqa: E402
import prototype_criticality_fast as pcf      # noqa: E402
import resilience_utils as ru                 # noqa: E402


# --------------------------------------------------------------------------- #
# Strict generalisation: community_weight=0 recovers exact CFEdge VALUES.      #
# --------------------------------------------------------------------------- #
def test_cfedge_recovery_at_zero_weight():
    for G in (nx.karate_club_graph(), nx.path_graph(8), nx.cycle_graph(6)):
        cf = ru.current_flow_edge_betweenness(G)
        scores, info = pcv2.kfc_v2_scores(G, community_weight=0.0)
        assert set(scores) == set(cf)
        for e in cf:
            assert abs(scores[e] - cf[e]) < 1e-9, f"{e}: {scores[e]} vs {cf[e]}"


# --------------------------------------------------------------------------- #
# Single-community partition degrades gracefully to pure CFEdge (K6 -> one     #
# Louvain community, no inter edges, tier shift never fires).                  #
# --------------------------------------------------------------------------- #
def test_single_community_equals_cfedge():
    G = nx.complete_graph(6)
    scores, info = pcv2.kfc_v2_scores(G)
    assert info["n_communities"] == 1
    assert info["frac_inter"] == 0.0
    cf = ru.current_flow_edge_betweenness(G)
    for e in cf:
        assert abs(scores[e] - cf[e]) < 1e-9


# --------------------------------------------------------------------------- #
# Pinned closed-form: barbell(5,0) = two K5s joined by bridge (4,5). Louvain   #
# finds the two K5s; the bridge is the only inter-community edge and must      #
# rank first. Its current-flow component is exactly 25/45 (all 25 cross       #
# pairs push unit current through the bridge; 45 total pairs).                 #
# --------------------------------------------------------------------------- #
def test_barbell_pinned():
    G = nx.barbell_graph(5, 0)
    cf, _ = pcv2.kfc_v2_scores(G, community_weight=0.0)
    assert abs(cf[(4, 5)] - 25.0 / 45.0) < 1e-9

    scores, info = pcv2.kfc_v2_scores(G)
    assert info["n_communities"] == 2
    top = max(scores, key=scores.get)
    assert top == (4, 5), f"bridge not top-ranked: {top}"
    others = [v for e, v in scores.items() if e != (4, 5)]
    assert scores[(4, 5)] > max(others)


# --------------------------------------------------------------------------- #
# Tier property: with the default (tiered) score every inter-community edge    #
# outranks every intra-community edge, and within a tier order follows cf1.    #
# --------------------------------------------------------------------------- #
def test_tier_separation_karate():
    G = nx.karate_club_graph()
    H = ru.largest_connected_component(G)
    parts = pcf.detect_communities(H, seed=42)
    memb = {v: ci for ci, c in enumerate(parts) for v in c}
    scores, info = pcv2.kfc_v2_scores(G)
    cf, _ = pcv2.kfc_v2_scores(G, community_weight=0.0)

    inter = {e for e in scores if memb[e[0]] != memb[e[1]]}
    intra = set(scores) - inter
    assert inter and intra
    assert min(scores[e] for e in inter) > max(scores[e] for e in intra)
    # within-tier ordering identical to CFEdge ordering
    for tier in (inter, intra):
        t = sorted(tier)
        cfo = sorted(t, key=lambda e: cf[e])
        v2o = sorted(t, key=lambda e: scores[e])
        assert cfo == v2o


# --------------------------------------------------------------------------- #
# Finite-gamma blend: values follow cf1 * (1 + g * inter) exactly.             #
# --------------------------------------------------------------------------- #
def test_finite_gamma_formula():
    G = nx.karate_club_graph()
    g = 2.0
    scores, _ = pcv2.kfc_v2_scores(G, community_weight=g)
    cf, _ = pcv2.kfc_v2_scores(G, community_weight=0.0)
    H = ru.largest_connected_component(G)
    parts = pcf.detect_communities(H, seed=42)
    memb = {v: ci for ci, c in enumerate(parts) for v in c}
    for e, val in scores.items():
        expect = cf[e] * (1.0 + g * (memb[e[0]] != memb[e[1]]))
        assert abs(val - expect) < 1e-9


# --------------------------------------------------------------------------- #
# Determinism: bit-identical across runs (fixed seed).                         #
# --------------------------------------------------------------------------- #
def test_determinism_across_runs():
    G = nx.karate_club_graph()
    d1 = pcv2.rank_kfc_v2(G).sort_values(["i", "j"]).reset_index(drop=True)
    d2 = pcv2.rank_kfc_v2(G).sort_values(["i", "j"]).reset_index(drop=True)
    assert (d1["i"] == d2["i"]).all() and (d1["j"] == d2["j"]).all()
    assert np.allclose(d1["KFC_v2"].values, d2["KFC_v2"].values, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# Schema: standard i, j, KFC_v2 frame, one row per LCC edge, no NaNs.          #
# --------------------------------------------------------------------------- #
def test_schema_and_integrity():
    G = nx.karate_club_graph()
    df = pcv2.rank_kfc_v2(G)
    assert list(df.columns) == ["i", "j", "KFC_v2"]
    assert len(df) == G.number_of_edges()
    assert not df.isnull().any().any()
    assert not (df["i"] == df["j"]).any()
    assert (df["KFC_v2"] > 0).all()


# --------------------------------------------------------------------------- #
# Dirty / disconnected input: reduced to the LCC, self-loops stripped, and     #
# the input graph is never mutated.                                            #
# --------------------------------------------------------------------------- #
def test_disconnected_input_uses_lcc():
    G = nx.Graph()
    G.add_edges_from(nx.karate_club_graph().edges())   # LCC (78 edges)
    G.add_edges_from([(100, 101), (101, 102)])         # smaller component
    G.add_edge(0, 0)                                   # self-loop
    df = pcv2.rank_kfc_v2(G)
    assert len(df) == 78
    assert 100 not in set(df["i"]) | set(df["j"])
    assert G.number_of_edges() == 78 + 2 + 1           # input untouched
    assert G.has_edge(0, 0)

    empty, info0 = pcv2.kfc_v2_scores(nx.Graph())
    assert empty == {} and info0["n_communities"] == 0


# --------------------------------------------------------------------------- #
# Integration: drop-in for run_and_plot, and strictly better dismantling AUC   #
# than CFEdge on karate (0.4244 vs 0.5439 — deterministic at seed 42).         #
# --------------------------------------------------------------------------- #
def test_run_and_plot_integration_and_auc_gain():
    import tempfile
    from network_utils import run_and_plot

    G = nx.karate_club_graph()
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "smoke_kfc_v2.png")
        res = run_and_plot(G, "kfc_v2_smoke", pcv2.rank_kfc_v2, out, reverse=True)
        assert set(res) == {"KFC_v2"}
        auc_v2 = res["KFC_v2"]["auc"]
        assert 0.0 <= auc_v2 <= 1.0 and np.isfinite(auc_v2)
        assert os.path.exists(out)

        out2 = os.path.join(tmp, "smoke_cf.png")
        res_cf = run_and_plot(
            G, "cf_smoke",
            lambda g: pcv2.rank_kfc_v2(g, community_weight=0.0).rename(
                columns={"KFC_v2": "CFE"}),
            out2, reverse=True)
        assert auc_v2 < res_cf["CFE"]["auc"]


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
