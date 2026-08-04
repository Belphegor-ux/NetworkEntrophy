# KFC v2 (fast) — all-networks static vs iterative benchmark, GPU-ready

**Date:** 2026-08-05 · **Branch:** `prototype` · **Status:** approved (user, this session)

## Goal

Run the KFC v2 family's production variant — `rank_kfc_v2_fast` from
`src/utils/prototype_criticality_v2.py` — as **the** KFC v2 on the real edge
lists in `All the networks/` (22 networks after LCC + self-loop removal).
Static dismantling on all feasible networks; **iterative** (full re-rank after
every single edge removal, `step_size=1`) on the 16 networks where it is
computationally feasible. Compare static vs iterative RGC curves and AUCs.
All curves come from dismantling the actual loaded graphs — no synthetic or
placeholder data anywhere.

## Scope decisions (measured, this session)

Per-rank cost scales with node count (dense grounded-Laplacian inverse);
iterative cost ≈ 0.4 · E · static-time. Measured on this machine (RTX 3050
4 GB, 16 GB RAM, CUDA 12.7):

- **Iterative + static (16):** karate, dolphins, lesmis, transport, football,
  office, baseball, crime, haggle, celegans, jazz, manufacturing, email,
  wikipedia, power, restaurant. Estimated total ≈ 16 h worst case → overnight,
  parallel per-network processes.
- **Static-only tonight (best-effort):** arxiv (27k E), anybeat (49k E),
  yeastnet (362k E, N=5.8k), astro (120k E), internet (48k E, N=23k —
  borderline RAM). Guarded per-network `try/except`; failures recorded, not
  fatal.
- **Deferred ("keep ready"):** condmat (N=36k → ~21 GB dense inverse, exceeds
  16 GB RAM) and iterative on all six big networks. The benchmark ships an
  `--estimate` mode printing per-network projected cost so these can be
  launched deliberately later (GPU path below applies to them unchanged).

## Components

### 1. `src/utils/kfc_linalg.py` (new)

Thin linear-algebra backend. `inv(a)` (and `pinv(a)`) route to
`torch.linalg` on CUDA when `KFC_GPU=1`, torch is importable, CUDA is
available, and `n >= threshold` (default 1500; below that CPU wins); NumPy
float64 otherwise. Silent, logged fallback to NumPy on any GPU error.
Precision policy decided empirically before the run: fp64 on GPU if faster
than CPU, else fp32 **only if** the resulting KFC_v2_fast scores match NumPy
to ≤1e-6 max abs deviation and identical AUC on karate/dolphins/power —
otherwise GPU is not used and that is reported honestly.

### 2. Hot-spot routing (edit, minimal)

`prototype_criticality_fast._grounded_inverse` calls `kfc_linalg.inv` for its
(n−1)×(n−1) block; `resilience_utils._laplacian_pinv` untouched (exact path,
small graphs only). Default behavior (no `KFC_GPU`) is byte-identical NumPy —
existing pinned tests (`tests/test_kfc_fast.py`, `tests/test_kfc_v2.py`,
`tests/test_resilience_utils.py`) must pass unchanged.

### 3. Memory chunking (edit, exact)

`kfc_v2_fast_scores` builds `D` (E×R) in one shot — 8.4 GB on yeastnet. Add
edge-axis chunking (auto chunk size from a ~1 GB budget): build `D` and
accumulate `_weighted_pairwise_absdiff` per chunk. Bitwise-equal results
(pure reordering of independent row computations); covered by a new test
asserting chunked == unchunked on jazz.

### 4. `benchmark_kfc_allnets.py` (new)

Modeled on `benchmark_kfc_static_vs_iterative.py` (parts → aggregate), single
method `KFC_v2` computed via `rank_kfc_v2_fast` (fast variant IS the main
KFC v2 here, stated in the report). CLI: `--network <name>` (one process per
network for parallel overnight run), `--aggregate`, `--estimate`. Outputs
(NEW files only) in `results_comparison/`:

- `kfc_allnets_parts/<name>_{summary,curves}.csv`
- `kfc_v2_allnets.csv` — AUC + runtime summary
- `kfc_v2_allnets_bars.png` — static vs iterative AUC bars (16 nets) +
  static-only panel (big nets that completed)
- `kfc_v2_allnets_curves.png` — 4×4 RGC grid, static dashed vs iterative solid
- `kfc_v2_allnets.md` — report incl. deferred-network cost estimates

### 5. Overnight orchestration

Launcher runs the 16 iterative networks as parallel background processes
(small ones batched, power/restaurant/email/wikipedia each solo), then the
big statics sequentially (RAM), then auto-aggregates. Progress logged per
part; aggregation tolerates missing big-net parts.

## Error handling

Per-network isolation (one crash never kills the run); NaN AUC + reason
recorded for failures; loader validates CSV has ≥2 columns and a non-empty
LCC; GPU backend never raises to the caller (falls back).

## Testing

Existing three test files stay green (no-GPU default identical). New:
chunking equivalence test; GPU-vs-NumPy equivalence check script run before
launch (gates the GPU flag).
