# KFC generalization — diagnosis and KFC_v2 design

**Branch:** `prototype` · Module: `src/utils/prototype_criticality_v2.py` ·
Tests: `tests/test_kfc_v2.py` · Numbers: `kfc_v2_validation.csv`.

**Problem:** v1 KFC (`rank_kfc`) only beats CFEdge on circuit-like networks;
mean gain over CFEdge across the benchmark was ~0.1% (`prototype_kfc_findings.md`).
Goal: make the method competitive across all network types.

## Diagnosis — why v1 fails off-circuit

Static dismantling AUC (lower = better), β-sweep of
`KFC(e) = cf1(e) · (1/(1−R_eff(e)))^β`. `b0 = β=0 = CFEdge` exactly.

| network | auto β | b0 (CFEdge) | b0.25 | b0.5 | b1 | b2 | b4 | auto |
|---|---|---|---|---|---|---|---|---|
| karate | 0.00 | 0.5439 | 0.5428 | 0.5677 | 0.5541 | 0.5383 | 0.5304 | 0.5439 |
| dolphins | 0.00 | 0.4243 | 0.4093 | 0.4071 | 0.4133 | 0.4288 | 0.4327 | 0.4243 |
| lesmis | 0.00 | 0.3367 | 0.3360 | 0.3360 | 0.3345 | 0.3417 | 0.3465 | 0.3367 |
| football | 0.00 | 0.3449 | 0.3468 | 0.3482 | 0.3491 | 0.3512 | 0.3631 | 0.3449 |
| jazz | 0.52 | 0.5690 | 0.5680 | 0.5670 | 0.5653 | 0.5629 | 0.5586 | 0.5670 |
| tokyo | 0.00 | 0.1788 | 0.3237 | 0.3252 | 0.3340 | 0.3448 | 0.3476 | 0.1788 |

Topology signatures (from the same Laplacian pseudoinverse):

| network | density | mean R_eff | cv(R_eff) | bridges | clustering |
|---|---|---|---|---|---|
| karate | 0.139 | 0.423 | 0.32 | 1 | 0.571 |
| dolphins | 0.084 | 0.384 | 0.49 | 9 | 0.259 |
| lesmis | 0.087 | 0.299 | 0.76 | 18 | 0.573 |
| football | 0.094 | 0.186 | 0.09 | 0 | 0.403 |
| jazz | 0.141 | 0.072 | 0.87 | 5 | 0.617 |
| tokyo | 0.016 | 0.816 | 0.19 | 56 | 0.039 |

Three compounding root causes:

1. **Bridge saturation of the irreplaceability term.** At a bridge
   `R_eff = 1` exactly, so `1/(1−R_eff)` hits the denominator floor (→ 1e12).
   Tokyo has 56 bridges (many of them structurally worthless leaf/pendant
   edges): at any β > 0 *every* bridge outranks every non-bridge, and the
   dismantling curve collapses (0.179 → 0.324 at β=0.25, +81%). This single
   failure mode forced the auto-tuning to be ultra-conservative.
2. **Auto-β collapses to 0 almost everywhere.** With `β = clip(1 −
   meanR_eff/0.15, 0, 1)`, five of six networks (all with mean R_eff ≥ 0.15)
   get β = 0 — i.e. **v1 IS CFEdge by construction on 5/6 networks**. The
   ~0.1% mean gain came entirely from jazz, the only network dense enough to
   clear the threshold. The mean-R_eff heuristic is not "mis-calibrated" so
   much as forced into a corner by failure mode (1).
3. **Scale degeneracy / no signal on dense or quasi-regular graphs.** On jazz
   all R_eff ∈ [0.04, 0.10]: the amplifier spans only ~[1.04, 1.11]^β and can
   barely reorder anything (hence the tiny jazz gain even at β=4). On football
   R_eff is nearly constant (cv = 0.09) — the term carries no information and
   any amplification is noise (AUC worsens monotonically with β).

Bounded rescalings of the same term (z-score `exp(β·z(R_eff))`, rank-quantile
`exp(β·(q−½))`) were swept too: they fix the blow-ups but the gains stay small
and inconsistent (best case ~−0.02 AUC on dolphins, still harmful on football,
mildly harmful on tokyo at large β). **Effective resistance is simply not the
right irreplaceability signal at ranking granularity.**

## What generalizes: community bridging

Sweeping a community-aware factor `cf1 · (1 + γ·[e inter-community])`
(Louvain, seed 42, via `prototype_criticality_fast.detect_communities`)
dominated every R_eff variant **on every network, including Tokyo** — the
grid's 500 kV inter-region trunks are exactly its inter-community edges, so
the term is not a social-network hack:

| network | γ=0 (CFEdge) | γ=1 | γ=2 | γ=4 | γ→∞ (tiered) |
|---|---|---|---|---|---|
| karate | 0.5439 | 0.4244 | 0.4244 | 0.4244 | 0.4244 |
| dolphins | 0.4243 | 0.3577 | 0.3399 | 0.3374 | 0.3374 |
| lesmis | 0.3367 | 0.2891 | 0.2874 | 0.2852 | 0.2852 |
| football | 0.3449 | 0.3326 | 0.3326 | 0.3326 | 0.3326 |
| jazz | 0.5690 | 0.4703 | 0.4221 | 0.4122 | 0.4104 |
| tokyo | 0.1788 | 0.1239 | 0.1115 | 0.1093 | 0.1093 |

The sweep saturates by γ ≈ 4 to the **lexicographic (two-tier) limit**: all
inter-community edges first (ordered by current flow), then all intra edges
(ordered by current flow). That limit is best or tied-best everywhere, needs
**no tuned constant at all** (no per-dataset parameters, no threshold), and is
seed-robust (Louvain seeds 0/1/7/42: max AUC spread ~0.015, ordering of
methods never changes).

## KFC_v2 (implemented)

```
KFC_v2(e) = cf1(e) + max(cf1) · [e is inter-community]   # default (γ→∞ tier)
KFC_v2(e; γ) = cf1(e) · (1 + γ·[inter])                  # finite blend
KFC_v2(e; γ=0) = CFEdge(e)                               # exact recovery
```

`cf1` is the same fast-exact current-flow edge betweenness (sorted
pairwise-difference identity) used by v1 and `resilience_utils`. Strict
generalisation is kept: `community_weight=0` reproduces CFEdge values to
1e-9 (unit-tested), and a single-community partition (e.g. K6) degrades to
pure CFEdge automatically. Pure NumPy + NetworkX; input never mutated;
standard `i, j, KFC_v2` frame.

## Validation (full table in `kfc_v2_validation.csv`)

| network | CFEdge | KFC v1 | **KFC_v2** | v2 vs CFEdge | v2 vs v1 |
|---|---|---|---|---|---|
| karate | 0.5439 | 0.5439 | **0.4244** | −22.0% | −22.0% |
| dolphins | 0.4243 | 0.4243 | **0.3374** | −20.5% | −20.5% |
| lesmis | 0.3367 | 0.3367 | **0.2852** | −15.3% | −15.3% |
| football | 0.3449 | 0.3449 | **0.3326** | −3.6% | −3.6% |
| jazz | 0.5690 | 0.5670 | **0.4104** | −27.9% | −27.6% |
| tokyo | 0.1788 | 0.1788 | **0.1093** | −38.9% | −38.9% |
| **mean** | 0.3996 | 0.3993 | **0.3166** | **−21.4%** | −21.3% |

Success criteria met: KFC_v2 ≤ CFEdge on **every** network (strictly better on
all six), strictly better than v1 on all social networks, and the circuit
advantage on Tokyo is not lost — it grows (largest single gain, −38.9%).

Caveats (honest): all gains are static-dismantling AUC with the repo's
standard tie-break; the community partition adds a Louvain pass (near-linear,
negligible vs the O(N³) pseudoinverse) and a seed dependence (bounded, see
above). The two-tier score is a ranking device — the raw value is no longer a
pure physical current, so for flow interpretation read the `γ=0` column.

Tests: `tests/test_kfc_v2.py` — 9/9 pass; `tests/test_resilience_utils.py`
18/18; `tests/test_kfc_fast.py` 9/9 (plain-script runners, no pytest needed).
