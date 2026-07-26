# KFC_fast and KFC_v2 — Design, Mathematics, and Benchmarks

*Branch `prototype`. Companion to `docs/kfc_paper.tex` (v1),
`results_comparison/kfc_v2_diagnosis.md`, `kfc_fast_notes.md`, and
`kfc_benchmark_report.md`. Status: 2026-07-26.*

---

## 1. Background: the KFC family

Kirchhoff Flow Criticality (KFC) treats an undirected network as an electrical
resistor circuit (every edge = a 1 Ω resistor) and ranks edges by how much they
matter to the circuit's ability to move current. The family has three members:

| variant | module | one-line summary |
|---|---|---|
| `rank_kfc` (v1) | `src/utils/prototype_criticality.py` | current-flow throughput × effective-resistance amplifier, adaptive β |
| `rank_kfc_fast` | `src/utils/prototype_criticality_fast.py` | v1 approximated over community-representative source/target pairs |
| `rank_kfc_v2` | `src/utils/prototype_criticality_v2.py` | current-flow throughput + community-bridging tier (replaces the amplifier) |

All three return the repo-standard DataFrame `i, j, <metric>` (one row per LCC
edge) and are drop-in for `run_and_plot(..., reverse=True)` in
`src/utils/network_utils.py`. All are pure NumPy + NetworkX (no scipy), never
mutate the input graph, and operate on the largest connected component.

### 1.1 The shared engine: current-flow edge betweenness (cf1)

For a unit current injected at source *s* and extracted at sink *t*, the current
through edge (u, v) is the potential difference `|p_u − p_v|`, where potentials
come from the graph Laplacian: `p = L⁺ (e_s − e_t)` (`L⁺` = Moore–Penrose
pseudoinverse of the Laplacian). Current-flow edge betweenness averages this
over **all** C(N,2) source–sink pairs:

```
cf1(u,v) = (1 / C(N,2)) · Σ_{s<t} | (L⁺_us − L⁺_ut) − (L⁺_vs − L⁺_vt) |
```

The repo computes this with the **sorted pairwise-difference identity**
(`resilience_utils.current_flow_edge_betweenness`): for the vector
`d = L⁺[u,:] − L⁺[v,:]`, the sum of |d_s − d_t| over all pairs equals
`Σ_k (2k − (N−1)) · sort(d)_k`. That turns an O(N²)-pairs sum into one
O(N log N) sort per edge — the "fast-exact" trick that made CFEdge the previous
benchmark champion. Total cost: one Laplacian pseudoinverse O(N³) + O(E·N log N).

### 1.2 Why v1 needed replacing (diagnosis summary)

v1 multiplied cf1 by an *irreplaceability* amplifier
`(1/(1 − R_eff))^β`, where `R_eff(u,v) = L⁺_uu + L⁺_vv − 2L⁺_uv` is the
effective resistance (for an edge inside a cycle R_eff < 1; for a bridge
R_eff = 1 exactly). The full diagnosis is in
`results_comparison/kfc_v2_diagnosis.md`; the three failure modes:

1. **Bridge saturation.** At a bridge, `1 − R_eff = 0`, so the amplifier hits
   the 1e-12 denominator floor and explodes to ~1e12. On tree-like graphs
   (Tokyo has 56 bridges) *every* bridge — including worthless leaf edges —
   jumps to the top of the ranking at any β > 0 (Tokyo AUC 0.179 → 0.324 at
   β = 0.25).
2. **Auto-β collapse.** To avoid (1), the adaptive rule
   `β = clip(1 − mean(R_eff)/0.15, 0, 1)` was tuned so conservatively that
   β = 0 on five of the six benchmark networks. **v1 is numerically identical
   to CFEdge almost everywhere by construction** — its reported "advantage on
   circuits" was really "the amplifier never engaged".
3. **Scale degeneracy.** On dense graphs all R_eff are tiny (jazz:
   0.04–0.10 → amplifier spans only ~[1.04, 1.11]^β) and on quasi-regular
   graphs R_eff is nearly constant (football: cv = 0.09) — no ranking signal
   either way. Bounded rescalings (z-score, rank-quantile) were swept and fix
   the blow-ups but yield only small, inconsistent gains. Conclusion:
   **effective resistance is the wrong irreplaceability signal at ranking
   granularity.**

---

## 2. KFC_fast — cluster-accelerated KFC

**File:** `src/utils/prototype_criticality_fast.py` ·
**API:** `rank_kfc_fast(G, beta=None, method='louvain', seed=42,
rep_fraction=0.5, min_centers=8)` → DataFrame `i, j, KFC_fast` ·
**Tests:** `tests/test_kfc_fast.py` (9)

### 2.1 Idea

Exact KFC accumulates current over **all** C(N,2) source–sink pairs. KFC_fast
replaces the full pair set with a community-stratified set of weighted
**representative** nodes, so the accumulation touches R ≪ N columns of the
inverse instead of all N.

### 2.2 Algorithm, step by step

**Step 1 — Community detection** (`detect_communities`).
`method='louvain'` uses `nx.community.louvain_communities(seed=seed)` directly.
`method='leiden'` is a **documented fallback**, not true Leiden:
`leidenalg`/`igraph` are not installed in the venv and networkx 3.6's
`leiden_communities` is a dispatch-only stub (raises `NotImplementedError`
without a GPU backend). Per project constraints no heavy dependency was added;
the fallback applies the two defining Leiden repairs on top of Louvain:

- *(a) well-connectedness*: split any internally-disconnected community into
  its connected components;
- *(b) refinement*: one deterministic greedy local-move sweep (nodes visited
  in sorted order, standard unweighted modularity delta
  `ΔQ = (k_{v,B} − k_{v,A∖v})/m − k_v(Σ_B − (Σ_A − k_v))/(2m²)`, ties keep the
  earliest community), then re-split.

Community order is canonicalized (sorted by min node label) so the whole
pipeline is bit-identical across runs (unit-tested).

**Step 2 — Representatives** (`community_representatives`).
Within each community C, nodes are ordered by closeness centrality of the
**community subgraph** (descending, ties by sorted label). The top node is the
community **center**. Center-only pairs were tried first and rejected — they
systematically starve peripheral edges of current (Spearman vs exact dropped to
~0.5). Instead, `k_C = min(|C|, max(min_centers, ⌈rep_fraction·|C|⌉))`
representatives are taken at **evenly spaced quantiles of the closeness
ordering** (quantile 0 = the center, so it is always included), each carrying
weight `w = |C| / k_C` — it "stands in" for that many community members.

**Step 3 — Weighted pair accumulation.**
Source/target pairs are all unordered pairs of distinct representatives
(both cross- and intra-community), pair (s,t) weighted `w_s·w_t`. The key
observation: the weighted sum `Σ_{a<b} w_a w_b |d_a − d_b|` collapses to a
**weighted version of the same sorted prefix-sum identity** used by exact
CFEdge, restricted to the R representative columns
(`_weighted_pairwise_absdiff`, fully vectorized across edges):
per edge O(R log R) instead of O(N log N).
With `rep_fraction=1.0` every node is a representative with weight 1 and the
result equals exact CFEdge **exactly** (pinned to 1e-9 in the tests) — the
approximation is a strict generalization of the exact computation.

**Step 4 — Fusion.** Identical to exact v1:
`KFC_fast(e) = cf_approx(e) · (1/(1 − R_eff(e)))^β`, with **exact** per-edge
effective resistances (they cost only O(E) once the inverse exists) and the
same adaptive β rule — so β matches exact KFC's choice on the same graph.

**Linear algebra note.** The dense solve is *not* avoided: R_eff needs (a
column basis of) the Laplacian inverse anyway. The module grounds one node and
uses `np.linalg.inv` on the (N−1)×(N−1) grounded Laplacian instead of
`np.linalg.pinv`'s SVD — same O(N³) class, several-fold smaller constant; the
grounding constant cancels in both the R_eff and the potential-difference
expressions (`_grounded_inverse` docstring).

### 2.3 Complexity and the honest accounting

| stage | exact KFC | KFC_fast |
|---|---|---|
| inverse | O(N³) (pinv/SVD) | O(N³) (grounded inv, smaller constant) |
| accumulation | O(E·N log N), per-edge Python loop | O(E·R log R), vectorized |
| preprocessing | — | Louvain/Leiden + per-community closeness (≈O(E), Python-constant-heavy) |

The genuine asymptotic saving is only the accumulation term. On the repo's
small networks (N ≤ 200) exact KFC already runs in milliseconds and the
preprocessing overhead **exceeds** the saving — KFC_fast is *slower*
end-to-end there. This is documented, not hidden.

### 2.4 Measured results (`results_comparison/kfc_fast_validation.csv`)

| network | N/E | Spearman vs exact | exact | fast (end-to-end) | speedup e2e | core-only |
|---|---|---|---|---|---|---|
| karate | 34/78 | 0.727 | 1.6 ms | 2.5 ms | 0.66× | 1.91× |
| dolphins | 62/159 | 0.825 | 3.2 ms | 5.3 ms | 0.61× | 2.20× |
| football | 115/613 | 0.822 | 14.8 ms | 18.9 ms | 0.79× | 1.24× |
| jazz | 198/2742 | 0.714 | 31.8 ms | 135 ms | 0.24× | 0.57× |
| tokyo | 156/190 | 0.968 | 14.7 ms | 27.6 ms | 0.53× | 0.82× |
| grid 40×40 (probe) | 1600/3120 | 0.989 | 1299 ms | 993 ms | **1.31×** | 2.48× |

Observations:

- **Crossover ≈ N ~ 10³**; the advantage grows with N (the O(E·R log R) vs
  O(E·N log N) gap widens and the vectorization constant matters more).
- **Approximation quality tracks community strength**: Tokyo (strong regional
  structure) ρ = 0.968; jazz (weak, overlapping communities) ρ = 0.714 and
  needs rep_fraction ≈ 0.5 of all nodes just to clear ρ > 0.7 — weak
  communities compress poorly.
- In the dismantling benchmark KFC_fast ranks #4–#9 per network (mean AUC
  0.4299 vs exact v1's 0.3993) — at these sizes it is an effectiveness
  downgrade with no speed benefit. **Its only valid niche is N ≳ 10³.**
- Louvain vs the Leiden fallback differ by ≲ 0.05 Spearman — the partition
  method is not the bottleneck.

### 2.5 Edge cases

- < 2 representatives (e.g. a tiny graph): falls back to exact `kfc_scores`
  and flags `info["fallback"] = True`.
- Empty/near-empty LCC: returns an empty score dict.
- Disconnected or self-looped input: handled via the LCC copy; input untouched.

---

## 3. KFC_v2 — the community-bridging generalization

**File:** `src/utils/prototype_criticality_v2.py` ·
**API:** `rank_kfc_v2(G, community_weight=None, method='louvain', seed=42)` →
DataFrame `i, j, KFC_v2` · **Tests:** `tests/test_kfc_v2.py` (9)

### 3.1 The generalization

The empirical sweep behind the diagnosis showed that what actually marks a
critical edge on modular networks is whether it **bridges communities** — and,
crucially, the same holds on the Tokyo grid: its 500 kV inter-region trunk
lines *are* its inter-community edges (Tokyo shows the largest v2 gain, not the
smallest — this is not a social-network hack). v2 therefore fuses current-flow
throughput with community bridging instead of raw effective resistance:

```
KFC_v2(e) = cf1(e) + max(cf1) · [e is inter-community]        (default)
```

This is a **two-tier score**: all inter-community edges rank strictly above all
intra-community edges, and each tier is internally ordered by exact cf1. The
tier shift `w = max(cf1)` guarantees the strict separation.

**Parameter-freeness.** The tiered form is the γ → ∞ limit of the
multiplicative blend `cf1 · (1 + γ·inter)`. The finite-γ sweep saturates to the
limit by γ ≈ 4 on **every** benchmark network, so the limit is the safe,
parameter-free default (`community_weight=None`). Passing a finite
`community_weight = g` gives the blend `cf1 · (1 + g·inter)`, and

- `community_weight = 0` **recovers exact CFEdge values** (pinned to 1e-9 in
  the tests — strict-generalization property preserved from v1, and the
  partition is skipped entirely in this branch);
- a single-community partition degrades gracefully to pure CFEdge.

Communities come from `prototype_criticality_fast.detect_communities`
(Louvain seed 42; `'leiden'` = the documented fallback of §2.2) — the two
modules share one partition implementation.

### 3.2 Benchmark results

Static dismantling AUC (lower = better), from
`results_comparison/kfc_v2_validation.csv`:

| network | CFEdge | KFC v1 | KFC_v2 | v2 vs CFEdge |
|---|---|---|---|---|
| karate | 0.5439 | 0.5439 | **0.4244** | −22.0% |
| dolphins | 0.4243 | 0.4243 | **0.3374** | −20.5% |
| lesmis | 0.3367 | 0.3367 | **0.2852** | −15.3% |
| football | 0.3449 | 0.3449 | **0.3326** | −3.6% |
| jazz | 0.5690 | 0.5670 | **0.4104** | −27.9% |
| tokyo | 0.1788 | 0.1788 | **0.1093** | −38.9% |
| **mean** | 0.3996 | 0.3993 | **0.3166** | **−21.4%** |

In the full 17-method comparison (`results_comparison/kfc_benchmark_report.md`)
**KFC_v2 ranks #1 on all six networks** — no other method beats it on any
network — at essentially CFEdge runtime (mean 0.018 s vs 0.015 s per ranking;
the only extra work over CFEdge is one Louvain call and an O(E) tier shift).
Seed-robustness: max AUC spread across Louvain seeds {0, 1, 7, 42} ≈ 0.015.

(Note: the benchmark harness reports slightly different absolute AUCs than the
validation table above — e.g. karate 0.4847 vs 0.4244 — due to
dismantling-curve harness details; each table is internally consistent and the
rankings agree.)

### 3.3 Cluster-accelerated v2 and the min-cut tier (`rank_kfc_v2_fast`)

`rank_kfc_v2_fast(G, ..., rep_fraction=0.5, mincut=False)` combines the two
ideas: the v2 community tier applied on top of the representative-pair cf
approximation of §2 (one shared community-detection pass). `rep_fraction=1.0`
recovers exact v2 to 1e-9 (unit-tested); `community_weight` semantics match
exact v2.

`mincut=True` adds a **third tier** answering "why not use max-flow/min-cut
here too": for every pair of *adjacent* communities, the minimum edge cut
between their centers is computed by max-flow (a handful of calls — versus
`MinCutCrit`'s thousands of sampled pairs), and cut-member edges outrank all
other inter-community edges (ordering: intra < inter < inter-min-cut).

Measured on the six benchmark networks
(`results_comparison/kfc_v2_fast_validation.csv`, same harness as the
17-method benchmark):

| variant | dolphins | football | jazz | karate | lesmis | tokyo | mean |
|---|---|---|---|---|---|---|---|
| KFC_v2 (exact) | **0.3374** | **0.3341** | 0.4104 | 0.4847 | 0.2852 | **0.1093** | 0.3268 |
| KFC_v2 + mincut | 0.3414 | 0.3379 | 0.4114 | **0.4410** | **0.2803** | 0.1241 | **0.3227** |
| KFC_v2_fast | 0.3730 | 0.3481 | **0.3966** | 0.4738 | 0.2848 | 0.1124 | 0.3315 |
| KFC_v2_fast + mincut | 0.3767 | 0.3510 | 0.3980 | 0.4210 | 0.2823 | 0.1271 | 0.3260 |

Honest read:

- **v2_fast** costs a little effectiveness (mean 0.3315 vs 0.3268) and, as
  with v1-fast, saves no time at these sizes — its niche remains N ≳ 10³,
  where it now carries the champion scoring rule rather than v1's.
- **The min-cut tier is a wash, not a win.** It helps clearly on karate
  (−0.044) and slightly on lesmis, but *hurts* on Tokyo (+0.015), dolphins,
  football, and jazz; the mean difference vs plain v2 is ~0.004. Mechanism:
  the center-to-center min-cut promotes the *narrowest* inter-community
  corridor, which is not always the highest-impact one — on Tokyo it elevates
  thin peripheral cuts above the high-current 500 kV trunks that current flow
  already ranks correctly. It stays available as an option (default **off**);
  plain two-tier v2 remains the default.

### 3.4 Known limitations

1. **Weak community structure ⇒ margin shrinks toward CFEdge.** Football
   (conference structure, dense inter-links) is the smallest win (−3.6%). On
   graphs with no community structure v2 degrades gracefully to CFEdge — a
   floor, not a failure, but it bounds the advantage.
2. **Louvain seed dependence.** Rankings vary mildly with the community seed
   (AUC spread ≈ 0.015). Reproducibility requires pinning the seed (default
   42); adversarially seed-sensitive graphs cannot be ruled out.
3. **Ranking-only semantics.** The tiered value is no longer a physical
   Kirchhoff current. When flow *values* are needed (load studies, capacity
   arguments), use `community_weight=0` and keep KFC_v2 for prioritization.
4. **Same O(N³) dense-algebra scaling as CFEdge** — for graphs too large for
   the Laplacian pseudoinverse, LDC (O(E)) remains the pragmatic screen.

---

## 4. Recommendations

- **Rank critical edges → `rank_kfc_v2`** (parameter-free default). New
  recommended default of the KFC family; supersedes "CFEdge is champion"
  (`prototype_kfc_findings.md` addendum).
- **Need physical current values → `community_weight=0`** (= exact CFEdge).
- **N ≳ 10³ → `rank_kfc_v2_fast`** — the cluster-accelerated form of the
  champion scoring rule (§3.3). `rank_kfc_fast` (v1-based) is kept for
  comparison only.
- **Very large sparse graphs → `LDC`** as an O(E) pre-screen.

## 5. Reproduction

```bash
# All KFC-family tests (run with or without pytest)
.venv/Scripts/python.exe tests/test_kfc_fast.py     # 9 tests
.venv/Scripts/python.exe tests/test_kfc_v2.py       # 9 tests
.venv/Scripts/python.exe tests/test_resilience_utils.py  # 18 tests

# KFC_fast speed/quality validation (rebuilds kfc_fast_validation.csv/notes)
.venv/Scripts/python.exe benchmark_kfc_fast.py

# Full 17-method × 6-network benchmark (rebuilds kfc_benchmark_*.{csv,md,png})
.venv/Scripts/python.exe benchmark_kfc_comparison.py
```

Key artifacts: `results_comparison/kfc_v2_diagnosis.md` (β-sweep + topology
signatures), `kfc_v2_validation.csv`, `kfc_fast_validation.csv`,
`kfc_fast_notes.md`, `kfc_benchmark_report.md` (§d: where KFC lacks),
`kfc_benchmark_results.csv`, and the figures
`kfc_benchmark_scatter.png` / `kfc_benchmark_bars.png`.
