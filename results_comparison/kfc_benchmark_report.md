# KFC-family benchmark — full method comparison and gap analysis

Branch `prototype`. Seeds fixed (Louvain seed 42). Per-cell budget 900 s. Effectiveness = static dismantling AUC via `NetworkDismantler.get_static_curve` (lower = better); runtime = one wall-clock ranking computation.

Networks: karate (34n/78e), dolphins (62n/159e), lesmis (77n/254e), football (115n/613e), jazz (198n/2742e), tokyo (156n/190e).

## (a) Dismantling AUC — methods × networks (lower = better)

| metric | karate | dolphins | lesmis | football | jazz | tokyo | mean |
|---|---|---|---|---|---|---|---|
| KFC_v2 ⭐ | 0.4847 (#1) | 0.3374 (#1) | 0.2852 (#1) | 0.3341 (#1) | 0.4104 (#1) | 0.1093 (#1) | 0.3268 |
| KFC ⭐ | 0.5439 (#3) | 0.4243 (#3) | 0.3367 (#3) | 0.3449 (#2) | 0.5670 (#3) | 0.1788 (#2) | 0.3993 |
| CFEdge | 0.5439 | 0.4243 | 0.3367 | 0.3449 | 0.5690 | 0.1788 | 0.3996 |
| KFC_fast ⭐ | 0.5782 (#9) | 0.4756 (#8) | 0.3601 (#5) | 0.3904 (#5) | 0.5914 (#7) | 0.1834 (#4) | 0.4299 |
| Jaccard | 0.5601 | 0.4542 | 0.3330 | 0.3693 | 0.6351 | 0.3679 | 0.4533 |
| EffRes | 0.5156 | 0.4422 | 0.4090 | 0.4531 | 0.5650 | 0.3870 | 0.4620 |
| EBC | 0.6092 | 0.5140 | 0.4005 | 0.4452 | 0.6662 | 0.2081 | 0.4739 |
| LLBCe | 0.6454 | 0.4549 | 0.4030 | 0.4619 | 0.5850 | 0.3062 | 0.4761 |
| LLBMEe1 | 0.5696 | 0.6097 | 0.3805 | 0.4393 | 0.7701 | 0.1933 | 0.4937 |
| MinCutCrit | 0.5537 | 0.3896 | 0.4384 | 0.7472 | 0.5747 | 0.2852 | 0.4982 |
| CI_e_av_body | 0.6054 | 0.6350 | 0.5198 | 0.7788 | 0.8790 | 0.2701 | 0.6147 |
| CI_e_av_skin | 0.6797 | 0.6072 | 0.6902 | 0.8288 | 0.6905 | 0.2749 | 0.6286 |
| CI_e_mul_body | 0.5767 | 0.6278 | 0.6291 | 0.7839 | 0.9149 | 0.2513 | 0.6306 |
| LDC | 0.5824 | 0.6721 | 0.6086 | 0.8077 | 0.9138 | 0.2291 | 0.6356 |
| BridgeImpact | 0.6559 | 0.5731 | 0.6048 | 0.8401 | 0.7570 | 0.4261 | 0.6428 |
| CI_e_mul_skin | 0.6582 | 0.6082 | 0.7696 | 0.8347 | 0.7660 | 0.2557 | 0.6487 |
| LKS | 0.5982 | 0.6847 | 0.6792 | 0.8458 | 0.8677 | 0.3415 | 0.6695 |

Per-network rank of each KFC variant (out of 17/17/17/17/17/17 metrics):

| variant | karate | dolphins | lesmis | football | jazz | tokyo |
|---|---|---|---|---|---|---|
| KFC | #3 | #3 | #3 | #2 | #3 | #2 |
| KFC_fast | #9 | #8 | #5 | #5 | #7 | #4 |
| KFC_v2 | #1 | #1 | #1 | #1 | #1 | #1 |

## (b) Runtime (seconds) — one ranking computation

| metric | karate | dolphins | lesmis | football | jazz | tokyo | mean |
|---|---|---|---|---|---|---|---|
| KFC_v2 | 0.002 | 0.006 | 0.009 | 0.018 | 0.049 | 0.028 | 0.018 |
| KFC | 0.001 | 0.004 | 0.004 | 0.014 | 0.032 | 0.025 | 0.013 |
| CFEdge | 0.001 | 0.004 | 0.005 | 0.014 | 0.033 | 0.032 | 0.015 |
| KFC_fast | 0.004 | 0.006 | 0.007 | 0.033 | 0.150 | 0.044 | 0.041 |
| Jaccard | 0.000 | 0.001 | 0.001 | 0.002 | 0.015 | 0.001 | 0.003 |
| EffRes | 0.001 | 0.003 | 0.004 | 0.011 | 0.021 | 0.025 | 0.011 |
| EBC | 0.003 | 0.006 | 0.009 | 0.028 | 0.128 | 0.058 | 0.039 |
| LLBCe | 0.053 | 0.177 | 0.602 | 2.824 | 112.212 | 0.375 | 19.374 |
| LLBMEe1 | 0.053 | 0.177 | 0.602 | 2.824 | 112.212 | 0.375 | 19.374 |
| MinCutCrit | 0.302 | 1.886 | 2.826 | 7.177 | 41.203 | 4.668 | 9.677 |
| CI_e_av_body | 0.001 | 0.002 | 0.004 | 0.011 | 0.040 | 0.003 | 0.010 |
| CI_e_av_skin | 0.001 | 0.002 | 0.004 | 0.011 | 0.040 | 0.003 | 0.010 |
| CI_e_mul_body | 0.001 | 0.002 | 0.004 | 0.011 | 0.040 | 0.003 | 0.010 |
| LDC | 0.000 | 0.000 | 0.001 | 0.001 | 0.004 | 0.001 | 0.001 |
| BridgeImpact | 0.001 | 0.004 | 0.009 | 0.003 | 0.066 | 0.054 | 0.023 |
| CI_e_mul_skin | 0.001 | 0.002 | 0.004 | 0.011 | 0.040 | 0.003 | 0.010 |
| LKS | 0.001 | 0.000 | 0.001 | 0.001 | 0.004 | 0.001 | 0.001 |

Note: rows sharing one ranking function (the four CI variants; LLBCe/LLBMEe1) show the runtime of that single shared computation.

## (c) Figures

- `kfc_benchmark_scatter.png` — mean AUC vs mean runtime (log-x), KFC family highlighted.
- `kfc_benchmark_bars.png` — per-network AUC bars, all metrics.

## Skipped / over-budget cells

None. Every (method, network) cell completed within the 900 s budget (MinCutCrit is sample-bounded at max_pairs=2000, which keeps it under budget even on jazz).

## (d) Where KFC lacks

### Cells where another method beats KFC_v2

| network | KFC_v2 AUC (rank) | better method | its AUC | margin (ΔAUC) | hypothesis |
|---|---|---|---|---|---|
| — | — | — | — | — | KFC_v2 is #1 on every network |

No (network, method) cell exists where another method beats KFC_v2 — it holds rank #1 on
all six networks. The gaps that remain are therefore about *margins*, *variants*, and
*practical caveats*, not head-to-head losses:

1. **Weak-community topologies compress the margin.** Football is KFC_v2's smallest win
   (0.3341 vs CFEdge 0.3449, ΔAUC ≈ 0.011, −3%). Football's Louvain communities are the
   athletic conferences — internally dense but heavily inter-linked, so the
   inter-community tier contains many edges and the two-tier device adds little
   information beyond current flow. Where community structure is weak or absent
   (near-regular / near-complete graphs), expect KFC_v2 → CFEdge. This is graceful
   degradation, not failure, but it bounds the method's advantage.

2. **KFC_fast is an effectiveness downgrade at these sizes.** It ranks #4–#9 per network
   (mean AUC 0.4299 vs exact KFC 0.3993) *and* is slower than exact below N≈10³ (the
   Louvain/closeness preprocessing dominates; jazz: 0.150 s vs 0.032 s exact). Its only
   valid niche is large graphs (crossover ~N≈10³, synthetic 40×40 grid: 1.31× end-to-end,
   Spearman 0.989). On repo-sized networks, use exact KFC/KFC_v2. Note KFC_fast
   approximates *v1*, not v2 — a cluster-accelerated v2 does not exist yet.

3. **Louvain seed dependence.** KFC_v2 rankings vary with the community seed (measured
   AUC spread ≈ 0.015 across seeds 0/1/7/42, seed fixed at 42 throughout). Bounded but
   real: reproducibility requires pinning the seed, and adversarially seed-sensitive
   graphs cannot be ruled out.

4. **Loss of physical interpretability.** The two-tier score is a ranking device: the
   raw value is no longer a pure Kirchhoff current. Where a physical flow value is
   needed (load studies, capacity arguments), use `community_weight=0` (= CFEdge values)
   and keep KFC_v2 for prioritisation only.

5. **When raw speed dominates.** LDC/LKS remain ~20–50× faster (sub-ms). On very large
   sparse grids where O(N³) dense linear algebra is infeasible, LDC is still the
   pragmatic screen (Tokyo AUC 0.229 in 0.4 ms) despite much worse effectiveness.

Absolute AUCs here differ slightly from `kfc_v2_validation.csv` (e.g. karate KFC_v2
0.4847 vs 0.4244) because the two runs used different dismantling-curve harness details;
each table is internally consistent and the rankings agree.

## (e) Verdict — update to `prototype_kfc_findings.md`

The previous verdict ("CFEdge is the robust champion; KFC's gain ~0.1%") is **superseded**.
The diagnosis showed v1's resistance amplifier was the wrong signal (auto-β collapsed to 0
on 5/6 networks); replacing it with the community-bridging tier changes the outcome:

- **KFC_v2 is the new recommended default** for edge-criticality ranking: #1 on all six
  networks, mean AUC 0.3268 vs CFEdge 0.3996 (−18%), at essentially CFEdge runtime
  (mean 0.018 s vs 0.015 s).
- **Conditions**: pin the Louvain seed; use `community_weight=0` when physical current
  values are required; expect the margin to shrink toward CFEdge on graphs without
  community structure.
- **CFEdge** remains the fallback when interpretability or seed-free determinism is
  mandatory; **LDC** remains the O(E) screen for very large graphs; **KFC_fast** is
  reserved for N≳10³ (and should be re-based on v2 before production use).
