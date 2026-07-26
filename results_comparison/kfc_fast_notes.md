# Cluster-accelerated KFC (`rank_kfc_fast`) — validation notes

**Branch:** `prototype` · Module: `src/utils/prototype_criticality_fast.py` ·
Tests: `tests/test_kfc_fast.py` · Raw numbers: `kfc_fast_validation.csv`.

## Design

1. **Communities**: `nx.community.louvain_communities(seed=42)`. `leidenalg`/
   `igraph` are not installed and networkx 3.6's `leiden_communities` is a
   dispatch-only stub, so `method='leiden'` is a documented **fallback**:
   Louvain + Leiden-style repairs (split internally-disconnected communities,
   one deterministic greedy local-move modularity sweep, re-split). Not the
   full Leiden algorithm.
2. **Centers**: per community, nodes are ordered by closeness centrality of the
   community subgraph (ties broken by sorted label). The top node is the
   center; representatives are taken at evenly spaced quantiles of that
   ordering — center-only pairs starve peripheral edges of current, so the
   periphery must be sampled. `k_C = min(|C|, max(8, ceil(0.5*|C|)))`, each
   representative weighted `|C|/k_C`.
3. **Pairs**: all unordered representative pairs, weight `w_s*w_t`. The
   weighted sum of absolute pairwise potential differences collapses to the
   same sorted prefix-sum identity as exact CFEdge, restricted to R
   representative columns and vectorised: **O(E·R log R)** vs exact's
   per-edge-Python-loop O(E·N log N). With `rep_fraction=1` the result equals
   exact KFC to 1e-9 (unit-tested).
4. **Fusion**: identical to exact KFC — `cf_approx * (1/(1-R_eff))^beta`,
   beta auto-tuned from the graph's mean effective resistance. R_eff is exact
   (O(E) from the inverse), so beta matches exact KFC exactly.

## Honest accounting — where time is (and is not) saved

* The dense solve is **not avoided**: the R_eff fusion term needs the
  (grounded) Laplacian inverse anyway. We use `np.linalg.inv` on the grounded
  Laplacian instead of `pinv`'s SVD — same O(N^3) class, smaller constant.
* The accumulation drops from O(E·N log N) (Python loop) to a vectorised
  O(E·R log R).
* The **new cost** is Louvain/Leiden + per-community closeness (pure-Python,
  constant-heavy). On the five benchmark networks (N <= 200) this overhead
  **exceeds the saving** — exact KFC already runs in 2-32 ms there. The
  `speedup_core_only` column (fast runtime minus partitioning)
  shows the linear-algebra + accumulation core is faster; the end-to-end
  crossover appears on the synthetic 40x40 grid (N=1600).

## Results

Louvain rows; the leiden fallback is within ~0.05 Spearman of louvain on
every network — full numbers in the CSV.

| network | N | E | reps | Spearman vs exact | exact (ms) | fast (ms) | end-to-end speedup | core-only speedup |
|---|---|---|---|---|---|---|---|---|
| karate | 34 | 78 | 26 | 0.727 | 1.6 | 2.5 | 0.66x | 1.91x |
| dolphins | 62 | 159 | 40 | 0.825 | 3.2 | 5.3 | 0.61x | 2.20x |
| football | 115 | 613 | 80 | 0.822 | 14.8 | 18.9 | 0.79x | 1.24x |
| jazz | 198 | 2742 | 101 | 0.714 | 31.8 | 135.1 | 0.24x | 0.57x |
| tokyo | 156 | 190 | 94 | 0.968 | 14.7 | 27.6 | 0.53x | 0.82x |
| grid40x40_synthetic | 1600 | 3120 | 805 | 0.989 | 1298.7 | 992.5 | 1.31x | 2.48x |

## Verdict

* Rank agreement with exact KFC is good on modular/sparse graphs (tokyo ~0.97)
  and adequate on dense low-modularity graphs (jazz ~0.7 needs half the nodes
  as representatives — community compression is weak when communities are).
* For the networks in this repo, **exact `rank_kfc` remains the right tool**
  (milliseconds). `rank_kfc_fast` is the scaling path: its advantage grows
  with N (see the synthetic grid row) because R stays ~rep_fraction*N with a
  vectorised accumulation while partitioning stays near-linear.
