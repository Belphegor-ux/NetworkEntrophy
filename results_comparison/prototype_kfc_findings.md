# Prototype: a faster, better critical-edge method — findings

**Branch:** `prototype` · **Goal:** design a new edge-criticality algorithm that is
faster and at least as effective as the existing suite across most networks, for
resilience / protection prioritisation of the Tokyo grid and general graphs.

**Testbed:** Karate (34n/78e), Tokyo grid (156n/190e), Football (115n/613e),
Jazz (198n/2742e) — sparse-to-dense, tree-like-to-modular-to-dense.
Effectiveness = static dismantling AUC (lower = ranks critical edges better);
efficiency = ranking runtime.

---

## Result 1 (biggest practical win): fast-exact current-flow betweenness

Across all four networks the single most robust method is **current-flow (electrical)
edge betweenness, `CFEdge`** — best or tied-best everywhere (best mean AUC 0.409).
Its textbook cost is O(N²·E) (a current computed for every source–sink pair). But the
edge score is a sum of pairwise differences of one vector, which collapses to a
**sorted-vector identity**:

```
CFEdge(e) = Σ_{s<t} |d[s] − d[t]| / n_pairs = Σ_i (2i − (n−1))·sort(d)_i / n_pairs,
d = L⁺[u,:] − L⁺[v,:]
```

giving **O(N³ + E·N log N)** — one Laplacian pseudoinverse plus a sort per edge, with
*identical values* (unit-tested). Measured speed-up over the naive accumulation:

| network | naive CFEdge | fast CFEdge | speed-up |
|---|---|---|---|
| Tokyo | 94 ms | 15 ms | 6× |
| Jazz | 395 ms | 29 ms | 13× |

And versus the expensive existing methods on Jazz: fast CFEdge (29 ms) is **~1,000×**
faster than min-cut membership (30 s) and **~3,900×** faster than LLBCe (114 s), at
better AUC. *This is the recommended default method.*

## Result 2 (the new algorithm): Adaptive Kirchhoff Flow Criticality (`KFC`)

Can we beat CFEdge? CFEdge captures *throughput*; effective resistance `EffRes`
captures *irreplaceability* and wins on dense graphs but ties all bridges and fails
on Tokyo. Both come from the same L⁺, so they can be fused via the Sherman–Morrison
**Kirchhoff-index sensitivity** (how much removing an edge raises the network's total
resistance-distance):

```
KFC(e) = CFEdge(e) · ( 1 / (1 − R_eff(e)) )^β ,   R_eff(e) = L⁺[u,u]+L⁺[v,v]−2L⁺[u,v]
β = clip( 1 − mean(R_eff)/0.15 , 0, 1 )      # adaptive: 0 for tree-like, →1 for dense
```

The exponent β adapts to the graph's **mean effective resistance** (tree-likeness),
which is free from the same L⁺. Sweeps showed a *fixed* amount of amplification helps
dense graphs but wrecks Tokyo (β=1 → AUC 0.334, +87%), so β must default to 0 and rise
only when the graph is clearly dense.

**KFC is a strict generalisation of CFEdge (β=0 recovers it exactly) and never regresses:**

| network | mean R_eff | adaptive β | CFEdge AUC | KFC AUC | Δ |
|---|---|---|---|---|---|
| Karate | 0.423 | 0.00 | 0.5439 | 0.5439 | 0.0000 |
| Tokyo | 0.816 | 0.00 | 0.1788 | 0.1788 | 0.0000 |
| Football | 0.186 | 0.00 | 0.3449 | 0.3449 | 0.0000 |
| Jazz | 0.072 | 0.52 | 0.5690 | **0.5670** | **−0.0019** |
| **mean** | | | 0.4092 | **0.4087** | −0.0005 |

## Honest verdict

- **CFEdge (fast-exact) is the robust champion** and the recommended method for the
  Tokyo grid and general use: best mean effectiveness, now 6–13× faster than naive and
  ~10³–10⁴× faster than the expensive existing methods.
- **KFC is a safe, principled generalisation** that auto-detects when irreplaceability
  weighting helps. On this four-network testbed the mean-AUC gain is small (~0.1%, from
  the densest network only) — a *negative-leaning* result: no fixed or adaptive fusion
  we found decisively beats plain current-flow betweenness. KFC's guarantee is that it
  *never does worse* and improves on dense graphs, decided automatically.
- **Where cheap-and-local is required**, `LDC` (degree product, O(E)) is the fastest
  near-optimal method on sparse grids (Tokyo AUC 0.229 in 0.4 ms) but is network-
  dependent (worst on dense Jazz/Football).

**Recommendation:** adopt fast-exact `CFEdge` as the production method; keep `KFC` as an
optional drop-in for dense networks. A larger, more diverse network sample would be
needed to establish whether adaptive amplification pays off more decisively.

---

## Addendum (2026-07-26) — verdict superseded by KFC_v2

The recommendation above predates `rank_kfc_v2` (`src/utils/prototype_criticality_v2.py`).
Diagnosis (`kfc_v2_diagnosis.md`) showed v1's effective-resistance amplifier was the wrong
irreplaceability signal: bridges saturate it (R_eff = 1) and dense graphs give it no spread,
so auto-β collapsed to 0 on 5/6 networks — v1 was numerically ≈ CFEdge, and "CFEdge is
champion" was really "v1 never engaged".

KFC_v2 replaces the resistance term with a parameter-free community-bridging tier
(Louvain inter-community edges first, current-flow order within tiers;
`community_weight=0` recovers CFEdge exactly). On the six-network benchmark
(`kfc_benchmark_report.md`) it ranks **#1 on every network** — mean dismantling AUC
0.3268 vs CFEdge 0.3996 — at CFEdge-level runtime. **KFC_v2 is the new recommended
default**, with caveats documented in the benchmark report §(d): pinned Louvain seed,
reduced margin on weak-community graphs, ranking-only (not physical-current) semantics.
