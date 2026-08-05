# KFC_v2 (fast) — static vs iterative on the full corpus

`rank_kfc_v2_fast` (cluster-accelerated KFC v2, Louvain seed 42) reported as `KFC_v2`. Iterative recomputes the full edge ranking after every removal (`step_size=1`). AUC of the RGC curve, lower = better. Real dismantling of the loaded graphs throughout — no synthetic data.

## Static vs iterative (16 networks)

| network | N | E | static AUC | iterative AUC | gain (s−i) | static s | iterative s |
|---|---|---|---|---|---|---|---|
| karate | 34 | 78 | 0.4097 | 0.3252 | +0.0845 | 0.01 | 0.37 |
| dolphins | 62 | 159 | 0.3730 | 0.2438 | +0.1292 | 0.02 | 0.72 |
| lesmis | 77 | 254 | 0.3047 | 0.2229 | +0.0819 | 0.03 | 1.15 |
| transport | 369 | 430 | 0.1193 | 0.0629 | +0.0565 | 0.55 | 15.39 |
| football | 115 | 613 | 0.3506 | 0.2127 | +0.1380 | 0.12 | 14.05 |
| office | 92 | 755 | 0.5031 | 0.3874 | +0.1156 | 0.10 | 13.88 |
| baseball | 45 | 947 | 0.7831 | 0.6665 | +0.1165 | 0.05 | 17.32 |
| crime | 1263 | 1377 | 0.0376 | 0.0153 | +0.0223 | 7.17 | 58.45 |
| haggle | 274 | 2124 | 0.6953 | 0.5053 | +0.1901 | 1.69 | 694.09 |
| celegans | 297 | 2148 | 0.5143 | 0.3638 | +0.1505 | 1.50 | 1021.67 |
| jazz | 198 | 2742 | 0.4127 | 0.3027 | +0.1100 | 3.49 | 419.14 |
| manufacturing | 167 | 3250 | 0.7117 | 0.5797 | +0.1320 | 2.79 | 901.78 |
| email | 1133 | 5451 | 0.3771 | 0.2454 | +0.1316 | 3.86 | 1228.23 |
| wikipedia | 677 | 6517 | 0.2691 | 0.1683 | +0.1008 | 5.26 | 3437.30 |
| power | 4941 | 6594 | 0.0413 | 0.0096 | +0.0317 | 112.81 | 1654.29 |
| restaurant | 4906 | 13457 | 0.5390 | 0.3589 | +0.1801 | 44.76 | 27079.29 |

## Static only (large networks)

| network | N | E | static AUC | runtime s |
|---|---|---|---|---|
| arxiv | 8798 | 27416 | 0.1326 | 164.30 |
| anybeat | 12645 | 49132 | 0.4315 | 496.57 |
| yeastnet | 5808 | 362421 | 0.5328 | 8307.45 |
| astro | 14845 | 119652 | 0.2372 | 1903.87 |
| internet | 22963 | 48436 | 0.2982 | 1145.88 |

## Deferred

- `condmat` (36,458 nodes): the dense grounded-Laplacian inverse needs ~21 GB — exceeds this machine's 16 GB RAM. Needs an out-of-core or sparse-solver path.
- Iterative on the six large networks (arxiv, internet, anybeat, astro, condmat, yeastnet): projected days-to-weeks each; run `--estimate` for current projections before attempting (GPU path via `KFC_GPU=1` applies unchanged).

## Figures

- `kfc_v2_allnets_bars.png` — AUC bars (hatched = static, solid = iterative; right panel static-only).
- `kfc_v2_allnets_curves.png` — 4x4 RGC grid (dashed = static, solid = iterative).