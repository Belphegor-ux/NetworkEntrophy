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

# Per-method scripts inside dataset dirs
python Football/CI/collective_influence.py            # static
python Football/CI/iterative_ci.py                    # iterative
# (same pattern for CKS, IE, Jaccard, LDC, LLBC, ME)

# Aggregate everything (cross-dataset summary)
python aggregate_results.py
python get_tokyo_aucs.py        # quick AUC table for Tokyo
python print_aucs.py            # print AUC summary
```

There is **no test suite, linter, or build step** — this is a research codebase. Verification is via the validation report and visual inspection of dismantling plots in `results_*/`.

Python venv is at `.venv/` (Python 3.13, NetworkX, pandas, numpy, matplotlib, seaborn). Use `.venv/Scripts/python.exe` on Windows.

## Architecture

### Shared core: `src/utils/`
- `network_utils.py` — `NetworkDismantler` class (static dismantling), `run_and_plot()` (driver), `create_metric_df()` (standardizes scores dict → DataFrame with `i, j, metric` columns).
- `network_utils_iter.py` — `get_iterative_curve()` (recomputes ranks after each removal), `run_iterative_benchmark()`.

Every per-dataset, per-method script imports from `src/utils/` via a `sys.path.insert(...)` hack. Path resolution is **cwd-relative** in most scripts (`run_iterative_analysis.py` is the exception — uses `__file__`-based paths).

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
2. **RESOLVED — CI 4 variants**: `PowerGrid_City/run_city_analysis.py:35–71` now emits all 4 variants (`CI_e_av_skin, CI_e_mul_skin, CI_e_av_body, CI_e_mul_body`). Note: `Football/CI/collective_influence.py` and `aggregate_results.py` may still drop variants — out of scope for this session, tracked separately.
3. **RESOLVED — LLBCe normalization**: `PowerGrid_City/run_city_analysis.py:101–102` now divides by `|fc|·(|fc|−1)`, making values comparable across edges.
4. **RESOLVED — Iterative truncation**: `src/utils/network_utils_iter.py:28–39` now warns and pads the RGC curve to full dismantled length instead of silently truncating on empty DataFrame.
5. **OPEN — `CKS` vs `LKS` naming inconsistency** (MEDIUM): `rank_cks` in `Karate/CKS/CKS_Link_K_Shell.py:16`, `Jazz/CKS/CKS_Link_K_Shell.py:16`, and `Football/CKS/CKS_Link_K_Shell.py:16` still emits the column `CKS`; CSV/validator expect `LKS`. Not yet addressed (per-dataset benchmark scripts were out of scope for the Tokyo-focused session).

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

**User constraints**:
- Do NOT modify data files (`japan_*.csv`, `tokyo_grid.gml`, `tokyo_grid_metrics.csv`). Code only.
- Use parallel agents (claude-flow / Agent tool with multiple parallel calls) to divide tasks.
- No generative/guess work for geographic data — every coordinate must be sourced.

**Cleanup pending**: ~~temp files in repo root (`_check_out.txt`, `_neigh.txt`, `_ops.txt`, `_picks.txt`, `_topdeg.txt`)~~ — DELETED 2026-04-27 (Track C).

When resuming: re-read `report.md`, `new_instructions.md`, `PowerGrid_City/README.md`, then this section. The "Completed (2026-04-27)" subsection above is the resume point; remaining work is the per-dataset CKS/CI rename (Issue #5 and the deferred portion of Issue #2).
