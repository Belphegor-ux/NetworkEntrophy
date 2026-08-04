# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NetworkEntropy is a research codebase for **identifying critical edges in complex networks** using topological, entropy-based, and flow/spectral metrics. It applies the same suite of edge-ranking methods across four core benchmarks (Karate, Jazz, Football, Tokyo Power Grid) plus a wider conference set (dolphins, lesmis, baseball, transport, hermaphrodite).

Source paradigm: each metric ranks edges, then edges are removed in rank order ("dismantling"). The Relative Giant Component (RGC) curve and its AUC measure how effectively a metric identifies structurally critical links — **lower AUC = better**.

Spec: see `new_instructions.md` (defines the canonical 9-column output schema and the CI 4-variant requirement). All metric DataFrames must have columns `i, j, <metric>` as the standard. Nine classic metrics: `LDC, Jaccard, LKS, LLBCe, LLBMEe1, CI_e_av_skin, CI_e_mul_skin, CI_e_av_body, CI_e_mul_body` (CI×4 + 5 others). The flow/spectral and KFC families (below) are additive, not part of that 9-column set.

## Common Commands

```bash
# Static + iterative benchmarks per dataset (parallel via gemini-flow)
./run_karate.sh
./run_jazz.sh
./run_football.sh

# Tokyo grid pipeline (sequential)
python PowerGrid_City/download_and_filter_tokyo.py    # downloads + bbox-filters Japan grid → tokyo_grid.gml
python PowerGrid_City/run_city_analysis.py            # computes static metrics → results_city/tokyo_grid_metrics.csv + plots
python PowerGrid_City/run_iterative_analysis.py --method ldc      # iterative for one method
python PowerGrid_City/validate_metrics.py             # writes results_city/validation_report.md
python PowerGrid_City/run_resilience_analysis.py      # max-flow/min-cut/current-flow/spectral criticality → results_city/tokyo_resilience_*.{csv,md,png}

# Per-method scripts inside dataset dirs
python Football/CI/collective_influence.py            # static
python Football/CI/iterative_ci.py                    # iterative
# (same pattern for CKS, IE, Jaccard, LDC, LLBC, ME)

# Aggregate everything (cross-dataset summary)
python aggregate_results.py
python get_tokyo_aucs.py        # quick AUC table for Tokyo
python print_aucs.py            # print AUC summary

# Unified multi-network suite: full 9-metric static + iterative dismantling on any i,j edge-list CSV
python run_network_suite.py     # → results_<name>/: metrics.csv, iter_order.csv, iter_<metric>_rgc.csv, plots, aucs_<name>.csv

# Methodology benchmark: effectiveness (dismantling AUC) vs runtime, all methods, 4 core networks
python benchmark_criticality_methods.py                 # full run → results_comparison/
python benchmark_criticality_methods.py --report-only   # rebuild report/plots from saved CSVs (no re-benchmark)

# KFC family benchmarks (v1, fast, v2, v2-fast — see Architecture below)
python benchmark_kfc_comparison.py       # KFC v1 vs CFEdge vs other methods
python benchmark_kfc_fast.py             # KFC v1-fast validation
python benchmark_kfc_allnets.py          # KFC v2 / v2-fast across all conference networks (argparse CLI, GPU-aware)
python benchmark_kfc_static_vs_iterative.py

# Dashboard + Excel comparison rebuild (after running run_network_suite.py on new datasets)
python build_dashboard_and_comparison.py   # → dashboard_app/dashboard_data.{js,json}, static_vs_iterative_comparison.xlsx

# Tests (only tested modules — the rest of the repo has no test suite)
.venv/Scripts/python.exe tests/test_resilience_utils.py          # plain-script runner (no pytest needed)
.venv/Scripts/python.exe -m pytest tests/test_resilience_utils.py -k barbell   # a single test
.venv/Scripts/python.exe -m pytest tests/test_kfc_v2.py          # KFC v2 tests (strict-generalization, tiering)
.venv/Scripts/python.exe -m pytest tests/test_kfc_fast.py        # KFC fast-mode tests
```

Most of the codebase has **no linter or build step**, and the classic dismantling metrics are verified only via `validate_metrics.py` and visual inspection of `results_*/` plots. The **exception is `src/utils/resilience_utils.py`, `prototype_criticality.py`, and the KFC v2/fast family** (below), which have real test suites pinning headline metrics to closed-form/ground-truth values; they run with **or without pytest** (bottom `if __name__ == "__main__"` runner). Add a test whenever you touch any of those modules.

Python venv is at `.venv/` (Python 3.13, NetworkX, pandas, numpy, matplotlib, seaborn). Use `.venv/Scripts/python.exe` on Windows.

## Architecture

### Shared core: `src/utils/`
- `network_utils.py` — `NetworkDismantler` class (static dismantling), `run_and_plot()` (driver), `create_metric_df()` (standardizes scores dict → DataFrame with `i, j, metric` columns).
- `network_utils_iter.py` — `get_iterative_curve()` (recomputes ranks after each removal), `run_iterative_benchmark()`.
- `resilience_utils.py` — **flow/spectral criticality** for infrastructure protection / N-1 contingency (complement to the topological dismantling metrics). Edge `rank_*` funcs return the same `i, j, <metric>` frame (drop-in for `run_and_plot`): `EBC`, `CFEdge` (current-flow betweenness — fast-exact O(N³+E·N log N) via the sorted pairwise-difference identity on the Laplacian pseudoinverse), `MinCutCrit` (min-cut membership via max-flow), `BridgeImpact` (N-1), `EffRes` (effective resistance). Plus `node_resilience_table()` and `global_resilience_summary()` (edge/node connectivity, Stoer–Wagner global + 2-core backbone min-cut, algebraic connectivity, Fiedler). Pure NumPy + NetworkX.
- `prototype_criticality.py` — **KFC v1**: adaptive Kirchhoff Flow Criticality fusing current-flow throughput (`cf1`) with effective-resistance irreplaceability, exponent β auto-tuned by mean effective resistance. Strict generalisation of `CFEdge` (β=0 recovers it exactly). **Superseded by v2** — v1's auto-β collapses to 0 on 5/6 benchmark networks (numerically ≈ CFEdge) and saturates near bridges (R_eff → 1). See `results_comparison/kfc_v2_diagnosis.md` for the failure analysis that motivated v2.
- `prototype_criticality_v2.py` — **KFC v2** (`rank_kfc_v2`), the current standard: two-tier score `KFC_v2(e) = cf1(e) + max(cf1) · inter(e)`, where `cf1` is exact current-flow edge betweenness and `inter(e)` flags edges that bridge Louvain communities (fixed seed → deterministic). Strictly beats CFEdge and v1 on all six core benchmarks (mean AUC −21.4%, Tokyo −38.9%) at ~CFEdge runtime. `community_weight=0` recovers CFEdge exactly (unit-tested). Also `rank_kfc_v2_fast` (`kfc_v2_fast_scores`) — same formula, but `cf1` is approximated via community-stratified representative nodes (closeness-ranked, evenly spaced, weighted `|C|/k_C`) instead of all-pairs, dropping cost from O(N log N) to O(R log R) per edge; `rep_fraction=1.0` reproduces exact v2 to 1e-9. Only wins wall-clock above N≈10³ — below that, community-detection overhead exceeds the saving, so exact v2 is both faster and more accurate on the repo's benchmark networks. See `docs/kfc_v2_formula_explained.md` for a symbol-by-symbol writeup (with units) and `docs/kfc_v2_pitch_deck.html` for a slide overview.
- `prototype_criticality_fast.py` — KFC v1-fast (community-stratified acceleration of v1) and `detect_communities()` (Louvain, shared by v1-fast and v2/v2-fast).
- `kfc_linalg.py` — shared linear-algebra backend for the whole KFC family: NumPy by default, **optional CUDA** via `KFC_GPU=1` env var (only engages when torch + CUDA are available AND the matrix is ≥ `KFC_GPU_MIN_N` — default 1500 — below which transfer overhead loses to NumPy). `KFC_GPU_FP32=1` enables float32 (only after `check_gpu_equivalence.py` passes; default is float64). Any GPU failure falls back to NumPy silently and disables GPU for the rest of the process.

Every per-dataset, per-method script imports from `src/utils/` via a `sys.path.insert(...)` hack. Path resolution is **cwd-relative** in most scripts (`run_iterative_analysis.py` is the exception — uses `__file__`-based paths).

### Resilience analysis & method benchmarks
- `PowerGrid_City/run_resilience_analysis.py` → `results_city/tokyo_resilience_*.{csv,md}` + `plot_resilience_all.png` (joins line/substation names read-only from `datasets/japan_*.csv`; never writes `tokyo_grid_metrics.csv`).
- `benchmark_criticality_methods.py` → `results_comparison/` — compares every classic method on effectiveness (dismantling AUC) vs runtime across karate/tokyo/football/jazz.
- `benchmark_kfc_allnets.py` — the current KFC benchmark driver: v2 and v2-fast across all conference networks + Tokyo, GPU-aware via `kfc_linalg.py`.
- KFC reference docs: `results_comparison/kfc_algorithm_dfd.md` + `kfc_dfd_level{0,1}.svg` (data-flow diagrams), `results_comparison/prototype_kfc_findings.md` (the pre-v2 verdict — superseded, kept for history), `results_comparison/kfc_v2_diagnosis.md` (why v1 fails), `docs/kfc_fast_and_v2.md` (full v2/v2-fast math + benchmark writeup, also compiled as `.pdf`), `docs/kfc_paper.tex` (research paper, `pdflatex`-compilable), `docs/kfc_slides.html` and `docs/kfc_v2_pitch_deck.html` (self-contained slide decks).

### Dataset directory pattern
```
{Karate,Jazz,Football}/{CI,CKS,IE,Jaccard,LDC,LLBC,ME}/
    <Method>_<descriptor>.py        # static scoring + RGC curve
    iterative_<method>.py           # iterative scoring (recompute after each removal)
```
Each method directory is independent — duplication across Karate/Jazz/Football is intentional (per-dataset reproducibility). Edits to one dataset's method must usually be replicated to the others.

### Tokyo Power Grid (`PowerGrid_City/`)
The most-scrutinized pipeline. Departs from the per-method-folder pattern — all 9 classic metrics live in a single file.

```
download_and_filter_tokyo.py  →  datasets/tokyo_grid.gml (LCC)
                                       ↓
run_city_analysis.py          →  results_city/tokyo_grid_metrics.csv + plot_<method>.png
run_iterative_analysis.py     →  results_city/plot_iterative_<method>.png   (one method per CLI arg)
validate_metrics.py           →  results_city/validation_report.md
run_resilience_analysis.py    →  results_city/tokyo_resilience_*.{csv,md,png}   (flow/spectral, separate from the above)
```

Data source: ComplexNetTSP/Power_grids Japan CSVs (raw OSM `power=*` extraction, NOT GridKit — GridKit is Europe + N. America only). Bbox `lon[138.3,141.0] lat[34.8,37.1]` → 156 nodes, 190 edges (LCC). All retained nodes are 50 Hz (TEPCO/J-Power), no Chubu (60 Hz) leak despite the loose bbox. 56 bridges, 54 articulation points in the LCC — this is why KFC_v2's community-bridging signal (its 500 kV inter-region trunks are inter-community edges) is especially effective here (AUC −38.9% vs CFEdge, the largest gain of any benchmark network).

### Output conventions
- `results_karate/`, `results_jazz/`, `results_football/`, `results_city/` — per-dataset plots and CSVs (classic 9 metrics).
- `results_<name>/` (from `run_network_suite.py`) — the wider conference-network set (dolphins, lesmis, baseball, transport, hermaphrodite, karate/football/jazz/tokyo re-derived).
- `results_comparison/` — cross-method benchmark outputs (both classic-method and KFC-family benchmarks).
- `results_tokyo_city/` exists separately from `results_city/` (older artifacts).
- Static plots: `result_<method>.png` or `plot_<method>.png`. Iterative: `result_<method>_iter.png` or `plot_iterative_<method>.png`.

## Conventions and Gotchas

- **Standardized DataFrame schema**: every `rank_*()` returns a DataFrame with `i, j` as the first two columns; metric columns follow. Drivers in `network_utils.py` infer metric columns as "everything except `i` and `j`".
- **CI must emit 4 variants** (`new_instructions.md` §2): skin/body × av/mul.
- **EI method is deprecated** — `new_instructions.md` §3 says drop it. Old `IE/` directories still exist; do not extend them.
- **LLBCe/LLBMEe1 must use `nx.edge_betweenness_centrality_subset`** with the first-order central domain `Γ(e) = {u, v} ∪ N(u) ∪ N(v)` as both sources and targets (Eq. 8 of MDPI Entropy 26(4) 315). Subgraph extraction is explicitly forbidden.
- **Reverse semantics**: `run_and_plot(..., reverse=True)` removes high-score edges first. Jaccard uses `reverse=False` (lower Jaccard = more critical bridge edge). KFC (all versions) dismantles `reverse=True`.
- **NumPy 2.x compat**: `np.trapz` → `np.trapezoid` is handled via `hasattr(np, "trapezoid")`.
- **Non-determinism in LLBCe**: `nx.edge_betweenness_centrality_subset` has tie-breaking dependent on dict iteration order; LLBCe values for tied-path edges are not reproducible across runs without sorting node iteration explicitly.
- **LLBMEe1 can be sign-mixed**: `LLBMEe1 = -LLBCe_raw·Σ log(LLBCe_raw)` is only uniformly negative if every neighbor raw subset-betweenness ≥ 1; with `normalized=False`, sub-1 values flip the log sign and a few edges can go positive. `validate_metrics.py` checks `LLBMEe1 ≤ 0` and reports sign-flip edges as info rather than failing.
- **Iterative metric values are not comparable across recompute steps** — `run_iterative_analysis.py` records the 1-based **removal order** per edge (`results_city/tokyo_iter_order_<method>.csv`) via `network_utils_iter.run_iterative_benchmark(..., order_csv_path=...)`, not the raw metric value. Static value CSVs remain directly sortable.
- **KFC v1 is superseded, not deleted** — kept for the diagnostic value of its failure modes (`kfc_v2_diagnosis.md`). Don't extend v1; extend v2 (`prototype_criticality_v2.py`) or its fast variant.
- **KFC GPU path is opt-in and silent-fallback** — `KFC_GPU=1` engages CUDA via `kfc_linalg.py` only above `KFC_GPU_MIN_N` (1500); any failure (no torch, no CUDA, OOM) falls back to NumPy without raising. Don't assume GPU is in use just because the env var is set — check `kfc_linalg`'s equivalence gate (`check_gpu_equivalence.py`) before trusting fp32 results.

## User constraints (standing, apply to all work in this repo)

- Do NOT modify data files (`japan_*.csv`, `tokyo_grid.gml`, `tokyo_grid_metrics.csv`). Code only.
- Grid work is framed as **resilience / infrastructure protection** (find critical components to harden / N-1 plan), not attack targeting.
- Use parallel agents (claude-flow / Agent tool with multiple parallel calls) to divide tasks where the work naturally splits (e.g. per-dataset benchmark runs).
- No generative/guess work for geographic data — every coordinate must be sourced.

## Resume point

Latest work: KFC v2 is now the standard edge-criticality method (superseding v1), with a fast/approximate variant (`rank_kfc_v2_fast`) for large networks and an optional GPU backend (`kfc_linalg.py`, opt-in via `KFC_GPU=1`). A design spec for an all-networks static-vs-iterative KFC v2 benchmark (GPU-ready) was added most recently at `.claude/plan/` — see that directory and `benchmark_kfc_allnets.py` for the implementation entry point when resuming that work. Older deferred items: renaming `CKS→LKS` in `{Karate,Jazz,Football}/CKS/CKS_Link_K_Shell.py`, and confirming CI 4-variant emission is complete everywhere it's supposed to be.
