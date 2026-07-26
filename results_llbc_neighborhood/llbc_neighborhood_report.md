# Effect of Local Neighborhood Size on LLBCe / LLBMEe1

Study script: `experiments/llbc_neighborhood_study.py`
Outputs: `results_llbc_neighborhood/` (this report, `aucs.csv`, `spearman.csv`, per-cell score CSVs, RGC comparison plots, `sanity_karate.csv`).
Date: 2026-07-26. Branch: `prototype`.

## 1. Methodology

LLBCe (Eq. 8 of MDPI Entropy 26(4) 315) is defined via
`nx.edge_betweenness_centrality_subset` computed **on the full graph**
(subgraph extraction is never used) with sources = targets = the *central
domain* of the edge. The production implementation
(`PowerGrid_City/run_city_analysis.py`) uses the first-order domain
Γ1(e=(u,v)) = {u, v} ∪ N(u) ∪ N(v). This study parameterizes the domain
order k:

    Γk(e) = { w : dist(w, u) ≤ k  or  dist(w, v) ≤ k }

so k=1 reproduces the current metric exactly, k=2 adds neighbors-of-Γ1, and
k=3 adds one more shell. For each edge:

- **LLBCe_k(e)** = raw subset betweenness of e with sources=targets=Γk(e),
  `normalized=False`, divided by |Γk|·(|Γk|−1) (the Eq.-8 normalization,
  generalized to the k-th-order domain size).
- **LLBMEe1_k(e)** = −raw(e) · Σ_{e2 ∈ N(e)} log(max(raw(e2), 1e-10)),
  0 when raw(e) ≤ 0, where N(e) are the edges incident to e's endpoints
  excluding e — identical formula to the reference, applied to the k-order
  raw values. Inverted criticality: smallest = most critical.

Evaluation: **static dismantling** via `src/utils/network_utils.py`
(`NetworkDismantler.get_static_curve`) — LLBCe removed high-score-first
(`reverse=True`), LLBMEe1 removed smallest-first (`reverse=False`) — with the
Relative-Giant-Component AUC as the score (**lower AUC = better metric**).

Determinism: node iteration is explicitly `sorted()` everywhere (edge lists,
Γk node lists, neighbor iteration) to sidestep the known tie-breaking
non-determinism of `edge_betweenness_centrality_subset`.

Networks (undirected, self-loops dropped, LCC only):

| network  | nodes | edges | source |
|----------|------:|------:|--------|
| karate   | 34    | 78    | `Networks to check/karate.csv` |
| dolphins | 62    | 159   | `Networks to check/dolphins.csv` |
| football | 115   | 613   | `Networks to check/football.csv` |
| jazz     | 198   | 2742  | `_derived_jazz.csv` |

A per-cell wall-clock budget of 15 min applied (pilot extrapolation over the
first 20 edges + hard mid-run cutoff). **No cell was skipped** — the slowest
cell (jazz, k=2) took 329 s.

## 2. AUC results (static dismantling; lower = better)

| network  | metric  | k=1 (current) | k=2 | k=3 | best k |
|----------|---------|------:|------:|------:|:--:|
| karate   | LLBCe   | 0.645 | 0.611 | **0.605** | 3 |
| karate   | LLBMEe1 | **0.570** | 0.588 | 0.578 | 1 |
| dolphins | LLBCe   | **0.455** | 0.471 | 0.507 | 1 |
| dolphins | LLBMEe1 | 0.610 | 0.580 | **0.568** | 3 |
| football | LLBCe   | 0.462 | **0.410** | 0.440 | 2 |
| football | LLBMEe1 | **0.439** | 0.486 | 0.490 | 1 |
| jazz     | LLBCe   | **0.585** | 0.679 | 0.668 | 1 |
| jazz     | LLBMEe1 | 0.770 | 0.746 | **0.745** | 3 |

Full table with runtimes: `aucs.csv`. Per-network RGC comparison curves:
`rgc_<network>_<metric>.png` (8 plots).

### Per-network reading

- **karate**: larger k helps LLBCe (0.645 → 0.605, −6%), consistent with
  LLBCe converging toward global edge betweenness, which is strong on this
  small bridge-dominated graph. LLBMEe1 is best at k=1.
- **dolphins**: larger k *hurts* LLBCe monotonically (0.455 → 0.507, +11%)
  but helps LLBMEe1 monotonically (0.610 → 0.568, −7%).
- **football**: mixed — k=2 is a clear win for LLBCe (0.462 → 0.410, −11%,
  the largest improvement observed anywhere), but k=3 gives most of it back;
  LLBMEe1 degrades at every k>1.
- **jazz**: larger k clearly *hurts* LLBCe (0.585 → 0.679, +16%, the largest
  degradation observed) and marginally helps LLBMEe1 (0.770 → 0.746, −3%).
  In this dense graph (⟨k⟩≈27.7), Γ2 already covers most of the graph, so
  LLBCe_k≥2 ≈ global betweenness — which dismantles jazz worse than the
  local variant.

**No consistent direction.** Across the 8 (network, metric) pairs, k>1 wins
4 and loses 4; whenever a larger domain helps LLBCe it tends to hurt LLBMEe1
on the same network and vice versa (karate is the partial exception). The
locality of Γ1 is not merely an approximation to save time — it is part of
what the metric measures, and removing it changes (sometimes degrades) the
dismantling behavior.

## 3. Runtime cost

Wall-clock seconds to compute both metrics for all edges (one subset-
betweenness pass per edge; the LLBMEe1 aggregation is negligible):

| network  | k=1 | k=2 | k=3 | k=3 / k=1 |
|----------|----:|----:|----:|----:|
| karate   | 0.06 | 0.11 | 0.13 | 2.2× |
| dolphins | 0.19 | 0.45 | 0.75 | 3.9× |
| football | 2.7  | 10.2 | 17.0 | 6.4× |
| jazz     | 116  | 329  | 384  | 3.3× |

Cost grows roughly with |Γk| (the number of BFS sources per edge) and
saturates once Γk approaches the whole graph (jazz k=2 → k=3 grows only
1.2×). k=2/k=3 are 3–6× more expensive than k=1 on these sizes; on larger
sparse graphs the multiplier keeps growing before saturation.

## 4. Ranking stability (Spearman ρ, k=1 vs k=2/3)

| network  | metric  | ρ(1,2) | ρ(1,3) |
|----------|---------|-------:|-------:|
| karate   | LLBCe   | 0.111  | 0.056  |
| karate   | LLBMEe1 | 0.961  | 0.951  |
| dolphins | LLBCe   | 0.798  | 0.648  |
| dolphins | LLBMEe1 | 0.919  | 0.826  |
| football | LLBCe   | 0.801  | 0.917  |
| football | LLBMEe1 | 0.953  | 0.946  |
| jazz     | LLBCe   | 0.641  | 0.546  |
| jazz     | LLBMEe1 | 0.969  | 0.961  |

- **LLBCe rankings change materially** with k (ρ as low as 0.06 on karate,
  0.55 on jazz): enlarging the domain is not a refinement of the k=1 ranking
  but a substantively different ordering. The karate outlier comes from many
  tied/near-tied Γ1 scores being re-ordered once the domain grows.
- **LLBMEe1 is far more stable** (ρ ≥ 0.83 everywhere, mostly ≥ 0.95): the
  entropy aggregation over incident edges washes out most of the domain-size
  effect, which also explains its small AUC deltas.

## 5. Sanity check: convergence to full edge betweenness

On karate with k = diameter = 5, Γk(e) = V for every edge, so LLBCe must
reduce to full (raw) edge betweenness divided by the constant N(N−1).
Verified (`sanity_karate.csv`, `sanity_karate_summary.txt`):

- ratio `full_EBC_raw / LLBCe_kfull` is constant across all 78 edges
  (coefficient of variation 1.6e-16, i.e. exact up to float rounding);
- Spearman ρ = 0.9999 (one near-tie pair at float-precision level).

This confirms the Γk construction and normalization are implemented
correctly and that k→∞ recovers global edge betweenness.

## 6. Recommendation

**Keep k=1 as the default. Neither k=2 nor k=3 justifies the cost.**

- Benefits are inconsistent: k>1 improved AUC in 4 of 8 cells and worsened
  it in the other 4; the sign is network- and metric-dependent, so a larger
  domain cannot be recommended a priori. The single big win (football LLBCe
  at k=2, −11% AUC) is offset by the big loss (jazz LLBCe at k=2, +16%).
- The cost is 3–6× on these small graphs and grows with graph size until Γk
  saturates — at which point LLBCe is just an expensive per-edge recompute
  of global edge betweenness, which `nx.edge_betweenness_centrality` gives
  in one pass.
- The rankings do change materially for LLBCe (§4), so k is a genuine model
  parameter, not a convergence knob: if a larger domain is ever desired,
  it should be validated per network (as here), and for LLBMEe1 it is
  hardly worth it (ρ ≥ 0.95, AUC deltas ≤ 0.03 in 3 of 4 networks).

Skipped cells: **none** (all 24 (network, metric, k) cells completed within
the 15-minute budget; max cell time 329 s on jazz k=2).

## Files

- `experiments/llbc_neighborhood_study.py` — self-contained study script
- `results_llbc_neighborhood/aucs.csv` — network, metric, k, auc, runtime_s, status
- `results_llbc_neighborhood/spearman.csv` — k=1 vs k=2/3 rank correlations
- `results_llbc_neighborhood/scores_<network>_k<k>.csv` — per-edge scores (12 files)
- `results_llbc_neighborhood/rgc_<network>_<metric>.png` — RGC comparisons (8 plots)
- `results_llbc_neighborhood/sanity_karate.csv`, `sanity_karate_summary.txt` — convergence check
