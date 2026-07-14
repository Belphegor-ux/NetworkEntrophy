# Tokyo Power Grid — Resilience & Critical-Component Report

**Purpose.** This report ranks the edges (transmission lines) and nodes (substations / junctions) of the Tokyo high-voltage backbone by how critical they are to global connectivity, so operators can prioritise them for **protection, redundancy, and N-1 / N-k contingency planning**. It is the flow/spectral complement to the topological dismantling suite in `tokyo_grid_metrics.csv`.

> **Data caveat.** The graph is an OpenStreetMap-derived extract (156 nodes, 190 edges, largest connected component). Many nodes are line-junction vertices rather than named substations, and the topology carries no per-line electrical parameters, so current-flow metrics use unit conductances (a topological electrical model, not a full AC power-flow). Treat rankings as structural indicators, not operational targeting.

## 1. Global robustness summary

- **Edge connectivity λ(G)** = 1 (minimum number of lines whose loss disconnects the network)
- **Node connectivity κ(G)** = 1
- **Global min-cut (Stoer–Wagner)** = 1, partition sizes [1, 155]
- **Backbone (2-core) min-cut** = 1, partition sizes [4, 97] (over 101 backbone nodes) — the meaningful structural bottleneck once dead-end stub lines are pruned
- **Algebraic connectivity (Fiedler value)** = 0.01199 (global robustness index; near 0 = close to fragmenting)
- **Fiedler bisection** splits the grid into [73, 83] nodes
- **Bridges** (single-line cut edges): 56
- **Articulation points** (single-node cut vertices): 54
- **Diameter**: 23 hops

Interpretation: λ = κ = 1 and a raw global min-cut of 1 reflect the many pendant stub lines in the extract (the cheapest cut merely isolates one leaf). The **backbone min-cut** and the **bridge/N-1 rankings below** are the actionable resilience signal.

## 2. Dismantling effectiveness (AUC — lower = ranks critical edges better)

| metric | dismantling AUC |
|---|---|
| CFEdge | 0.1787 |
| EBC | 0.2081 |
| MinCutCrit | 0.2531 |
| BridgeImpact | 0.4261 |

_Note: `BridgeImpact` is a single-failure (N-1) indicator — it is 0 for every non-bridge, so it is deliberately not a full dismantling order and its higher AUC here is expected. Use it as a protection flag (Section 3), not as a sequential ranking._

## 3. Top critical lines to protect (N-1 single-failure impact)

Edges ranked by the fraction of the network severed if that one line is lost (bridges only; non-bridges = 0).

| rank | edge (v_id) | line | BridgeImpact |
|---|---|---|---|
| 1 | 596–6057 | 江東線 | 0.0385 |
| 2 | 312–6069 | 西越谷線 (Nishi Koshigaya Sen) (275kV) | 0.0385 |
| 3 | 146–7911 | 新新潟幹線 (500kV) | 0.0321 |
| 4 | 5936–6069 | 西越谷線 (Nishi Koshigaya Sen) (275kV) | 0.0321 |
| 5 | 6057–6093 | 江東線 (275kV) | 0.0321 |
| 6 | 189–5748 | 東京西線;東京西線;東京西線;東京西線;東京西線; 愛宕線 | 0.0256 |
| 7 | 5499–6093 | 江東線 (275kV) | 0.0256 |
| 8 | 403–5936 | 西越谷線 (Nishi Koshigaya Sen) (275kV) | 0.0256 |
| 9 | 312–5793 | 春日部線 (Kasukabe Sen) (275kV) | 0.0256 |
| 10 | 403–5934 | 西越谷線 (Nishi Koshigaya Sen) | 0.0192 |

## 4. Top bottleneck lines (min-cut membership via max-flow)

Edges lying on the minimum source–sink cut for the largest share of node pairs.

| rank | edge (v_id) | line | MinCutCrit |
|---|---|---|---|
| 1 | 146–7911 | 新新潟幹線 (500kV) | 0.0392 |
| 2 | 5640–6060 | 5640–6060 (275kV) | 0.0235 |
| 3 | 189–5748 | 東京西線;東京西線;東京西線;東京西線;東京西線; 愛宕線 | 0.0231 |
| 4 | 536–6092 | 京浜線1,2号線 (275kV) | 0.0222 |
| 5 | 2082–8606 | 2082–8606 (275kV) | 0.0198 |
| 6 | 6044–6045 | 6044–6045 (275kV) | 0.0189 |
| 7 | 5886–6042 | 新京葉線 (275kV) | 0.0188 |
| 8 | 596–6057 | 江東線 | 0.0183 |
| 9 | 312–6069 | 西越谷線 (Nishi Koshigaya Sen) (275kV) | 0.0170 |
| 10 | 275–9174 | 佐久間東幹線;佐久間東幹線;佐久間東幹線;佐久間東幹線;佐久間東幹線;佐久間東幹線 | 0.0156 |

## 5. Top load-bearing corridors (current-flow betweenness)

Edges carrying the most electrical throughput under a unit-current resistor model.

| rank | edge (v_id) | line | CFEdge |
|---|---|---|---|
| 1 | 401–429 | 只見幹線;只見幹線 (Tadami Kansen);只見幹線 (Tadami Kansen) | 0.2100 |
| 2 | 189–401 | 只見幹線;只見幹線;只見幹線 | 0.2066 |
| 3 | 1006–420 | 新古河線 (Shin Koga Sen) (500kV) | 0.2053 |
| 4 | 1006–6063 | 新古河線 (Shin Koga Sen) (500kV) | 0.2045 |
| 5 | 415–6045 | 新佐原線 (500kV) | 0.1989 |
| 6 | 1605–981 | 1605–981 (500kV) | 0.1919 |
| 7 | 218–597 | 新茂木線 (Shin Motegi Sen);新茂木線 (Shin Motegi Sen) | 0.1890 |
| 8 | 1605–6775 | 1605–6775 | 0.1876 |
| 9 | 218–6775 | 218–6775 | 0.1855 |
| 10 | 479–497 | 新多摩線 (Shin Tama Sen);新多摩線 (Shin Tama Sen);新多摩線 (Shin Tama Sen) | 0.1819 |

## 6. Top critical substations / junctions to protect

Nodes ranked by current-flow betweenness; articulation points are single-node cut vertices whose loss disconnects the network.

| rank | node (v_id) | name | CFNode | AP? |
|---|---|---|---|---|
| 1 | 420 | 新古河変電所 | 0.3712 |  |
| 2 | 433 | 新所沢変電所 (Shin Tokorozawa Hendensho) | 0.3236 |  |
| 3 | 189 | 西東京変電所 | 0.2928 | yes |
| 4 | 312 | 北東京変電所 (Kita Tōkyō Hendensho) | 0.2918 | yes |
| 5 | 596 | 新京葉変電所 | 0.2738 | yes |
| 6 | 1605 | 新岡部変電所 | 0.2668 |  |
| 7 | 399 | 新坂戸変電所 | 0.2664 |  |
| 8 | 479 | 新多摩変電所 (Shin Tama Substation) | 0.2481 |  |
| 9 | 429 | 南狭山変電所 (Minamisayama Hendensho) | 0.2450 |  |
| 10 | 6045 | 新佐原線 | 0.2386 |  |
