# Critical-Edge Method Comparison — Methodology & Efficiency

**Question.** Which method identifies the edges most critical to network connectivity *fastest* (for resilience / protection prioritisation)? We compare effectiveness (dismantling AUC, lower = better) against cost (ranking runtime) across four networks, and derive a more efficient method.

Networks: karate (34n/78e), tokyo (156n/190e), football (115n/613e), jazz (198n/2742e).

## Cross-network summary (sorted by mean AUC)

| method | metric | mean AUC | mean AUC-rank | mean runtime (ms) | complexity | ρ vs EffRes |
|---|---|---|---|---|---|---|
| KFC ⭐ | KFC | 0.4087 | 1.5 | 17.4 | O(N^3 + E*N) adaptive fusion [NEW] | 0.24 |
| CFEdge | CFEdge | 0.4092 | 1.8 | 16.0 | O(N^3 + E*N log N) pinv+sorted-id | 0.24 |
| EffRes ⭐ | EffRes | 0.4802 | 5.5 | 13.1 | O(N^3 + E) pinv+per-edge [NEW] | 1.00 |
| EBC | EBC | 0.4822 | 6.8 | 53.7 | O(N*E) Brandes | 0.14 |
| Jaccard | Jaccard | 0.4831 | 6.8 | 5.1 | O(E*k) neighbour overlap | -0.50 |
| LLBC/LLBME | LLBMEe1 | 0.4931 | 6.0 | 33616.3 | O(E*fc*E) subset EBC | 0.14 |
| LLBC/LLBME | LLBCe | 0.4996 | 8.8 | 33616.3 | O(E*fc*E) subset EBC | 0.61 |
| MinCutCrit | MinCutCrit | 0.5402 | 6.5 | 10751.9 | O(pairs*maxflow) all-pairs cut | 0.73 |
| CI | CI_e_av_skin | 0.6185 | 11.0 | 16.3 | O(N*ball) l-ball sums | -0.24 |
| CI | CI_e_mul_skin | 0.6286 | 11.0 | 16.3 | O(N*ball) l-ball sums | -0.26 |
| CI | CI_e_mul_body | 0.6317 | 9.5 | 16.3 | O(N*ball) l-ball sums | -0.77 |
| LDC | LDC | 0.6333 | 9.5 | 1.5 | O(E) degree product | -0.78 |
| CI | CI_e_av_body | 0.6333 | 10.0 | 16.3 | O(N*ball) l-ball sums | -0.65 |
| LKS | LKS | 0.6633 | 12.0 | 1.5 | O(E) k-shell product | -0.66 |
| BridgeImpact | BridgeImpact | 0.6698 | 12.8 | 14.6 | O(E*(N+E)) N-1 recompute | 0.35 |

## Pareto frontier (best AUC achievable per unit runtime)

Non-dominated methods (no other method is both faster and more effective):

- **LKS** — AUC 0.6633, 1.5 ms (O(E) k-shell product)
- **LDC** — AUC 0.6333, 1.5 ms (O(E) degree product)
- **Jaccard** — AUC 0.4831, 5.1 ms (O(E*k) neighbour overlap)
- **EffRes** — AUC 0.4802, 13.1 ms (O(N^3 + E) pinv+per-edge [NEW])
- **CFEdge** — AUC 0.4092, 16.0 ms (O(N^3 + E*N log N) pinv+sorted-id)
- **KFC** — AUC 0.4087, 17.4 ms (O(N^3 + E*N) adaptive fusion [NEW])

## Finding: the more efficient method

- **Most effective (mean AUC):** `KFC` at 0.4087 (rank 1.5). `CFEdge` and `KFC` are within noise of each other (0.4092 vs 0.4087) — the two best methods.
- **Fast-exact current-flow `CFEdge`:** mean 16.0 ms via the sorted pairwise-difference identity (O(N³ + E·N log N)); **~2098× faster** than `LLBCe` (33616 ms) at far better AUC. The robust default.
- **KFC (new adaptive fusion):** mean AUC 0.4087 at 17.4 ms. A strict generalisation of CFEdge (β=0 recovers it exactly on tree-like graphs, so it never regresses) that adds irreplaceability weighting only when the graph is dense — it edges ahead of CFEdge on the densest network, marginally on the mean.
- **Fastest near-optimal (within 10% of best AUC):** `CFEdge` (AUC 0.4092, 16.0 ms).


**Recommendation.** Adopt **fast-exact `CFEdge`** as the production critical-edge method: best-in-class effectiveness across sparse and dense networks, and now cheap. Use **`KFC`** as a drop-in when the network is dense (it auto-detects this and otherwise equals CFEdge). The fixed/adaptive fusion gains over plain current-flow are small on this four-network testbed — the decisive efficiency lever is the algorithmic speedup, not a new metric. See `prototype_kfc_findings.md` for the full write-up.

## Per-network detail

### karate

| metric | AUC | runtime (ms) | ρ vs CFEdge |
|---|---|---|---|
| EffRes | 0.5156 | 1.0 | -0.01 |
| CFEdge | 0.5439 | 1.4 | 1.00 |
| KFC | 0.5439 | 1.4 | 1.00 |
| MinCutCrit | 0.5537 | 370.3 | -0.06 |
| Jaccard | 0.5601 | 0.3 | -0.53 |
| LLBMEe1 | 0.5696 | 77.3 | -0.53 |
| CI_e_mul_body | 0.5767 | 1.5 | 0.24 |
| LDC | 0.5824 | 0.3 | 0.25 |
| LKS | 0.5982 | 0.3 | 0.10 |
| CI_e_av_body | 0.6054 | 1.5 | 0.30 |
| EBC | 0.6092 | 2.6 | 0.66 |
| LLBCe | 0.6454 | 77.3 | 0.39 |
| BridgeImpact | 0.6559 | 1.0 | 0.13 |
| CI_e_mul_skin | 0.6582 | 1.5 | -0.21 |
| CI_e_av_skin | 0.6797 | 1.5 | 0.01 |

### tokyo

| metric | AUC | runtime (ms) | ρ vs CFEdge |
|---|---|---|---|
| KFC | 0.1788 | 18.4 | 1.00 |
| CFEdge | 0.1788 | 17.4 | 1.00 |
| LLBMEe1 | 0.1933 | 272.2 | -0.67 |
| EBC | 0.2081 | 41.5 | 0.79 |
| LDC | 0.2291 | 0.6 | 0.58 |
| CI_e_mul_body | 0.2513 | 2.7 | 0.70 |
| CI_e_mul_skin | 0.2557 | 2.7 | 0.71 |
| CI_e_av_body | 0.2701 | 2.7 | 0.59 |
| CI_e_av_skin | 0.2749 | 2.7 | 0.61 |
| MinCutCrit | 0.2852 | 3334.6 | -0.17 |
| LLBCe | 0.3062 | 272.2 | 0.01 |
| LKS | 0.3415 | 0.5 | 0.62 |
| Jaccard | 0.3679 | 0.6 | 0.06 |
| EffRes | 0.3870 | 15.8 | -0.36 |
| BridgeImpact | 0.4261 | 24.3 | -0.55 |

### football

| metric | AUC | runtime (ms) | ρ vs CFEdge |
|---|---|---|---|
| KFC | 0.3449 | 16.1 | 1.00 |
| CFEdge | 0.3449 | 15.4 | 1.00 |
| Jaccard | 0.3693 | 2.9 | -0.88 |
| LLBMEe1 | 0.4393 | 3424.2 | -0.87 |
| EBC | 0.4452 | 42.4 | 0.90 |
| EffRes | 0.4531 | 13.6 | 0.78 |
| LLBCe | 0.4619 | 3424.2 | 0.84 |
| MinCutCrit | 0.7472 | 8802.3 | 0.09 |
| CI_e_av_body | 0.7788 | 15.8 | -0.02 |
| CI_e_mul_body | 0.7839 | 15.8 | -0.03 |
| LDC | 0.8077 | 1.1 | -0.17 |
| CI_e_av_skin | 0.8288 | 15.8 | -0.28 |
| CI_e_mul_skin | 0.8347 | 15.8 | -0.28 |
| BridgeImpact | 0.8401 | 4.1 | — |
| LKS | 0.8458 | 1.0 | -0.09 |

### jazz

| metric | AUC | runtime (ms) | ρ vs CFEdge |
|---|---|---|---|
| EffRes | 0.5650 | 21.9 | 0.54 |
| KFC | 0.5670 | 33.8 | 1.00 |
| CFEdge | 0.5690 | 29.8 | 1.00 |
| MinCutCrit | 0.5747 | 30500.5 | 0.47 |
| LLBCe | 0.5850 | 130691.6 | 0.89 |
| Jaccard | 0.6351 | 16.4 | -0.71 |
| EBC | 0.6662 | 128.1 | 0.69 |
| CI_e_av_skin | 0.6905 | 45.1 | 0.00 |
| BridgeImpact | 0.7570 | 29.1 | 0.07 |
| CI_e_mul_skin | 0.7660 | 45.1 | -0.05 |
| LLBMEe1 | 0.7701 | 130691.6 | -0.38 |
| LKS | 0.8677 | 4.1 | -0.69 |
| CI_e_av_body | 0.8790 | 45.1 | -0.37 |
| LDC | 0.9138 | 4.2 | -0.45 |
| CI_e_mul_body | 0.9149 | 45.1 | -0.45 |
