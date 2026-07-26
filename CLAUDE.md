# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NetworkEntropy is a research codebase for **identifying critical edges in complex networks** using topological and entropy-based metrics. It applies the same suite of edge-ranking methods across four datasets: three benchmarks (Karate, Jazz, Football) and one authentic city-scale infrastructure (Tokyo Power Grid).

Source paradigm: each metric ranks edges, then edges are removed in rank order ("dismantling"). The Relative Giant Component (RGC) curve and its AUC measure how effectively a metric identifies structurally critical links — lower AUC = better.

Spec: see `new_instructions.md` (defines the canonical 9-column output schema and the CI 4-variant requirement). All metric DataFrames must have columns `i, j, <metric>` as the standard. Eleven metrics total: `LDC, Jaccard, LKS, LLBCe, LLBMEe1, CI_e_av_skin, CI_e_mul_skin, CI_e_av_body, CI_e_mul_body` (CI×4 + 5 others).

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

# Methodology benchmark: effectiveness (dismantling AUC) vs runtime, all methods, 4 networks
python benchmark_criticality_methods.py                 # full run → results_comparison/
python benchmark_criticality_methods.py --report-only   # rebuild report/plots from saved CSVs (no re-benchmark)

# Tests (resilience/criticality only — the rest of the repo has none)
.venv/Scripts/python.exe tests/test_resilience_utils.py                              # plain-script runner (no pytest needed)
.venv/Scripts/python.exe -m pytest tests/test_resilience_utils.py                    # or via pytest
.venv/Scripts/python.exe -m pytest tests/test_resilience_utils.py -k barbell         # a single test
```

Most of the codebase has **no linter or build step**, and the classic dismantling metrics are verified only via `validate_metrics.py` and visual inspection of `results_*/` plots. The **exception is `src/utils/resilience_utils.py` + `prototype_criticality.py`**, which have a real test suite (`tests/test_resilience_utils.py`, ~18 tests) pinning every headline metric to a closed-form/ground-truth value; it runs with **or without pytest** (bottom `if __name__ == "__main__"` runner). Add a test whenever you touch that module.

Python venv is at `.venv/` (Python 3.13, NetworkX, pandas, numpy, matplotlib, seaborn). Use `.venv/Scripts/python.exe` on Windows.

## Architecture

### Shared core: `src/utils/`
- `network_utils.py` — `NetworkDismantler` class (static dismantling), `run_and_plot()` (driver), `create_metric_df()` (standardizes scores dict → DataFrame with `i, j, metric` columns).
- `network_utils_iter.py` — `get_iterative_curve()` (recomputes ranks after each removal), `run_iterative_benchmark()`.
- `resilience_utils.py` — **flow/spectral criticality** for infrastructure protection / N-1 contingency (complement to the topological dismantling metrics). Edge `rank_*` funcs return the same `i, j, <metric>` frame (drop-in for `run_and_plot`): `EBC`, `CFEdge` (current-flow betweenness — fast-exact O(N³+E·N log N) via `current_flow_edge_betweenness`, the sorted pairwise-difference identity), `MinCutCrit` (min-cut membership via max-flow), `BridgeImpact` (N-1), `EffRes` (effective resistance). Plus `node_resilience_table()` and `global_resilience_summary()` (edge/node connectivity, Stoer–Wagner global + 2-core backbone min-cut, algebraic connectivity, Fiedler). **Pure NumPy + NetworkX — no scipy** (current-flow/spectral come from the Laplacian pseudoinverse / `eigh`).
- `prototype_criticality.py` (**branch `prototype` only**) — adaptive Kirchhoff Flow Criticality (`KFC`): fuses current-flow throughput with effective-resistance irreplaceability, exponent β auto-tuned by mean effective resistance; strict generalisation of `CFEdge` (β=0 recovers it). See `results_comparison/prototype_kfc_findings.md`.

Every per-dataset, per-method script imports from `src/utils/` via a `sys.path.insert(...)` hack. Path resolution is **cwd-relative** in most scripts (`run_iterative_analysis.py` is the exception — uses `__file__`-based paths).

### Resilience analysis & method benchmark
- `PowerGrid_City/run_resilience_analysis.py` → `results_city/tokyo_resilience_*.{csv,md}` + `plot_resilience_all.png` (joins line/substation names read-only from `datasets/japan_*.csv`; never writes `tokyo_grid_metrics.csv`).
- `benchmark_criticality_methods.py` → `results_comparison/` — compares every method on effectiveness (dismantling AUC) vs runtime across karate/tokyo/football/jazz. Finding: fast-exact `CFEdge` is the robust champion; the min-cut/subset-betweenness methods are ~10³–10⁴× slower for no gain.
- KFC reference docs (write-ups, not code): `results_comparison/kfc_algorithm_dfd.md` + `kfc_dfd_level{0,1}.svg` (data-flow diagrams), `results_comparison/prototype_kfc_findings.md` (benchmark verdict — KFC's mean gain over CFEdge is ~0.1%, honest), `docs/kfc_paper.tex` (research paper, compiles with `pdflatex`), `docs/kfc_slides.html` (self-contained slide deck). Note the docs' Sherman–Morrison motivation uses `‖d‖²/(1−R_eff)` but the code's `KFC` uses the `cf1` throughput term — it is Kirchhoff-*inspired*, not the exact sensitivity.

### Dataset directory pattern
```
{Karate,Jazz,Football}/{CI,CKS,IE,Jaccard,LDC,LLBC,ME}/
    <Method>_<descriptor>.py        # static scoring + RGC curve
    iterative_<method>.py           # iterative scoring (recompute after each removal)
```
Each method directory is independent — duplication across Karate/Jazz/Football is intentional (per-dataset reproducibility). Edits to one dataset's method must usually be replicated to the others.

### Tokyo Power Grid (`PowerGrid_City/`)
The newest and most-scrutinized pipeline. Departs from the per-method-folder pattern — all 11 metrics live in a single file.

```
download_and_filter_tokyo.py  →  datasets/tokyo_grid.gml (LCC)
                                       ↓
run_city_analysis.py          →  results_city/tokyo_grid_metrics.csv + plot_<method>.png
run_iterative_analysis.py     →  results_city/plot_iterative_<method>.png   (one method per CLI arg)
validate_metrics.py           →  results_city/validation_report.md
```

Data source: ComplexNetTSP/Power_grids Japan CSVs (raw OSM `power=*` extraction, NOT GridKit — GridKit is Europe + N. America only). Bbox `lon[138.3,141.0] lat[34.8,37.1]` → 156 nodes, 190 edges (LCC). All retained nodes are 50 Hz (TEPCO/J-Power), no Chubu (60 Hz) leak despite the loose bbox.

### Output conventions
- `results_karate/`, `results_jazz/`, `results_football/`, `results_city/` — per-dataset plots and CSVs.
- `results_tokyo_city/` exists separately from `results_city/` (older artifacts).
- Static plots: `result_<method>.png` or `plot_<method>.png`. Iterative: `result_<method>_iter.png` or `plot_iterative_<method>.png`.

## Conventions and Gotchas

- **Standardized DataFrame schema**: every `rank_*()` returns a DataFrame with `i, j` as the first two columns; metric columns follow. Drivers in `network_utils.py` infer metric columns as "everything except `i` and `j`".
- **CI must emit 4 variants** (`new_instructions.md` §2): skin/body × av/mul. Several scripts in the repo only emit `CI_e_mul_skin` — see "Known issues" below.
- **EI method is deprecated** — `new_instructions.md` §3 says drop it. Old `IE/` directories still exist; do not extend them.
- **LLBCe/LLBMEe1 must use `nx.edge_betweenness_centrality_subset`** with the first-order central domain `Γ(e) = {u, v} ∪ N(u) ∪ N(v)` as both sources and targets (Eq. 8 of MDPI Entropy 26(4) 315). Subgraph extraction is explicitly forbidden.
- **Reverse semantics**: `run_and_plot(..., reverse=True)` removes high-score edges first. Jaccard uses `reverse=False` (lower Jaccard = more critical bridge edge).
- **NumPy 2.x compat**: `np.trapz` → `np.trapezoid` is handled in both utils via `hasattr(np, "trapezoid")`.
- **Non-determinism in LLBCe**: `nx.edge_betweenness_centrality_subset` has tie-breaking dependent on dict iteration order; LLBCe values for tied-path edges are not reproducible across runs without sorting node iteration explicitly.

## Known Issues (status as of 2026-04-27)

Originally CRITICAL/HIGH findings from a code review; most are now resolved in-tree on `iterative-method`.

1. **RESOLVED — Reproducibility for Tokyo CSV**: `PowerGrid_City/run_city_analysis.py:144–174` regenerates the CSV with the full extended schema; columns match `validate_metrics.py:15`.
2. **RESOLVED — CI 4 variants**: `PowerGrid_City/run_city_analysis.py:35–71` emits all 4 variants (`CI_e_av_skin, CI_e_mul_skin, CI_e_av_body, CI_e_mul_body`). Verified 2026-05-29: `Football/CI/collective_influence.py:37–43` and `aggregate_results.py:135–146` also emit all 4 variants now. Fully resolved.
3. **RESOLVED — LLBCe normalization**: `PowerGrid_City/run_city_analysis.py:101–102` now divides by `|fc|·(|fc|−1)`, making values comparable across edges.
4. **RESOLVED — Iterative truncation**: `src/utils/network_utils_iter.py:28–39` now warns and pads the RGC curve to full dismantled length instead of silently truncating on empty DataFrame.
5. **RESOLVED — `CKS` vs `LKS` naming inconsistency**: verified 2026-05-29 that `Karate/CKS/CKS_Link_K_Shell.py`, `Jazz/CKS/CKS_Link_K_Shell.py`, and `Football/CKS/CKS_Link_K_Shell.py` all define `rank_lks` emitting the `LKS` column. (Plot filenames are still `result_cks.png` — cosmetic only.)
6. **RESOLVED — validator rejected negative LLBMEe1**: `PowerGrid_City/validate_metrics.py:40` previously asserted `LLBMEe1 ≥ 0`, but LLBMEe1 is inverted-criticality (≤ 0, smallest = most critical). Fixed 2026-05-29 — LLBMEe1 removed from the non-negative set and given a dedicated `≤ 0` check that reports any sign-flip edges as info.

## Current Investigation State (resume point)

**Branch**: `iterative-method`. **Date snapshot**: 2026-04-27.

The user is mid-investigation on the Tokyo Power Grid. Three parallel agents have completed:
- **Code review** of `PowerGrid_City/*.py` — produced the issue list above.
- **Authoritative source audit** — confirmed ComplexNetTSP Japan is OSM-derived, not GridKit. No publicly-digitized authoritative TEPCO graph exists; closest proxy is Toyoda et al. IEEJ Trans. FMS 128(3) on J-STAGE. TEPCO illustrated facility plan: tepco.co.jp/en/corpinfo/illustrated/electricity-supply/network-facility-plan-e.html
- **Verification of node coordinates** — 10/10 named substations spot-checked real (Shin-Tokorozawa, Shin-Tama, Shin-Keiyo, Shin-Hadano, Shin-Okabe, Boso, Shin-Haruna, Shin-Sawara, Katsunan, Shin-Fuji). All 156 LCC nodes are 50 Hz. Top-degree hubs are real 500/275 kV trunks. Caveat: 57/156 nodes are unnamed OSM line-junction vertices; LCC has 56 bridges, 54 articulation points, diameter 23 — line-segment topology overrepresented vs substations.

**Completed (2026-04-27)**:

Track A — Restored reproducibility for the Tokyo CSV. `PowerGrid_City/run_city_analysis.py` extended to emit the full canonical 9-column schema: `rank_ci` now emits all 4 variants (skin/body × av/mul), `LLBC→LLBCe`, `LLBME→LLBMEe1`, `CKS→LKS` aligned with `validate_metrics.py:15`. CSV regeneration path: lines 144–174.

Track B — Fixed LLBCe normalization in `PowerGrid_City/run_city_analysis.py:101–102` (divide by `|fc|·(|fc|−1)`), making LLBCe values comparable across edges with different neighborhood sizes.

Track C — Repository cleanup and CLAUDE.md refresh. Deleted the five `_*.txt` verification scratch files from repo root; refreshed Known Issues section to reflect the post-fix state; replaced the obsolete "Pending plan" subsection with this Completed log.

Also resolved: `src/utils/network_utils_iter.py:28–39` empty-DF guard (warn + pad instead of silent truncation).

Out of scope (deferred): renaming `CKS→LKS` in `{Karate,Jazz,Football}/CKS/CKS_Link_K_Shell.py` (Issue #5), and the CI 4-variant emission in `Football/CI/collective_influence.py` and `aggregate_results.py` (Issue #2 partial).

**Completed (2026-05-29)** — conference results refresh:

- **Validator fix (Issue #6)**: `PowerGrid_City/validate_metrics.py` no longer rejects negative `LLBMEe1`; it checks `LLBMEe1 ≤ 0` and reports sign-flip edges as info.
- **Iterative records ORDER, not value**: `network_utils_iter.run_iterative_benchmark` already supports `order_csv_path`; wired it into `PowerGrid_City/run_iterative_analysis.py` (→ `results_city/tokyo_iter_order_<method>.csv`). For static, value CSVs remain correct (sortable). For iterative, metric values are not comparable across recompute steps, so the 1-based removal order per edge is the record.
- **Unified multi-network suite**: `run_network_suite.py` runs the full 9-metric static + iterative dismantling on any `i,j` edge-list CSV (LCC + self-loop removal), writing `results_<name>/`: `<name>_metrics.csv` (static values), `<name>_iter_order.csv` (removal order), `<name>_iter_<metric>_rgc.csv`, plots, and `aucs_<name>.csv`. Ran the 7 conference networks from `Networks to check/` (karate, dolphins, lesmis, football, baseball, transport, hermaphrodite) plus Jazz and Tokyo (via derived `_derived_*.csv` edge lists).
- **Dashboard + comparison**: `build_dashboard_and_comparison.py` rebuilds `dashboard_app/dashboard_data.{js,json}` from suite outputs with the **canonical** schema (replacing the stale `CKS`/single-`CI`/`LLBC`/`LLBME` keys), makes the dashboard dataset dropdown dynamic, and writes `static_vs_iterative_comparison.xlsx` + `results_comparison/` figures (static-vs-iterative AUC scatter + per-metric bars).
- **LLBMEe sign caveat**: `LLBMEe1 = -LLBCe_raw·Σ log(LLBCe_raw)` is only uniformly negative if every neighbor raw subset-betweenness ≥ 1; with `normalized=False`, sub-1 values flip the log sign and a few edges can go positive. The validator now surfaces this rather than failing. If many positives appear, revisit raw-vs-normalized semantics with the user.

**Completed (2026-07-14)** — resilience / critical-component analysis (defensive framing):

- **New module `src/utils/resilience_utils.py`**: flow/spectral criticality complement to the topological dismantling suite, for identifying components to **protect / harden** (N-1 / N-k contingency). Network-agnostic, pure NumPy + NetworkX (**no scipy** — current-flow and Fiedler metrics are computed directly from the Laplacian pseudoinverse / `eigh`). Edge `rank_*` fns (`EBC`, `CFEdge` current-flow, `MinCutCrit` min-cut membership via max-flow, `BridgeImpact` N-1 single-failure) return the standard `i,j,<metric>` frame and are drop-in for `run_and_plot`. Also `node_resilience_table` and `global_resilience_summary` (edge/node connectivity, Stoer–Wagner global + 2-core backbone min-cut, algebraic connectivity, Fiedler bisection).
- **Driver `PowerGrid_City/run_resilience_analysis.py`**: writes NEW files only (`tokyo_resilience_metrics.csv`, `tokyo_node_resilience.csv`, `tokyo_resilience_report.md`, `plot_resilience_all.png`); joins line/substation names read-only from `japan_*.csv`. Never touches `tokyo_grid_metrics.csv`.
- **Key semantics**: the user's "min-cut/max-flow to find critical points" is served by min-cut *membership* (bottleneck edges), NOT flow *load* (which peaks at hub-incident edges). Raw global min-cut = 1 is a pendant-stub artifact; the report leads with the **2-core backbone** cut + bridge/N-1 rankings. Tokyo top findings: load corridors = Tadami/Shin-Koga 500 kV trunks; top bottleneck = Shin-Niigata 500 kV; critical substations = Shin-Koga/Shin-Tokorozawa/Shin-Tama (match prior coordinate audit).
- **Tests `tests/test_resilience_utils.py`** (14, all pass; runs with or without pytest): every headline metric pinned to a closed-form/ground-truth value (barbell, cycle, path, tree), plus sampling-branch determinism, dirty/disconnected-graph handling, and a `run_and_plot` integration smoke test. Hardened after an adversarial verification workflow found the numerics correct but under-asserted.

**User constraints**:
- Do NOT modify data files (`japan_*.csv`, `tokyo_grid.gml`, `tokyo_grid_metrics.csv`). Code only.
- Grid work is framed as **resilience / infrastructure protection** (find critical components to harden / N-1 plan), not attack targeting.
- Use parallel agents (claude-flow / Agent tool with multiple parallel calls) to divide tasks.
- No generative/guess work for geographic data — every coordinate must be sourced.

**Cleanup pending**: ~~temp files in repo root (`_check_out.txt`, `_neigh.txt`, `_ops.txt`, `_picks.txt`, `_topdeg.txt`)~~ — DELETED 2026-04-27 (Track C).

When resuming: re-read `report.md`, `new_instructions.md`, `PowerGrid_City/README.md`, then this section. The "Completed (2026-07-14)" resilience subsection above is the latest resume point (run `python PowerGrid_City/run_resilience_analysis.py`; tests via `.venv/Scripts/python.exe tests/test_resilience_utils.py`). Older remaining work is the per-dataset CKS/CI rename (Issue #5 and the deferred portion of Issue #2).
