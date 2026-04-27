# Tokyo Power Grid - Critical Link Analysis

## 1. Objective
Identify and quantify critical edges in the Tokyo power grid using a set of static network topology metrics. This analysis aims to detect vulnerabilities in city-scale infrastructure based on authentic, non-generative data.

## 2. Methodology & Data Acquisition
### Authentic Data Sourcing
To ensure results are representative of real infrastructure, the dataset was sourced from the **ComplexNetTSP/Power_grids** repository (GridKit extract of OpenStreetMap).
- **Dataset**: `highvoltage_vertices.csv` and `highvoltage_links.csv` for Japan.
- **Geographical Filtering**: Applied a bounding box for the **Tokyo/Eastern Japan** region:
  - Longitude: `[138.3, 141.0]`
  - Latitude: `[34.8, 37.1]`
- **Graph Construction**: Developed a NetworkX graph using the filtered nodes and links, specifically extracting the **Largest Connected Component** to ensure a contiguous network.
- **Network Stats**: 156 nodes, 190 edges (high-voltage backbone).

### Implementation of Network Metrics
Eleven static edge-ranking metrics were applied in parallel to the extracted graph:
1. **LDC (Link Degree Centrality)**: $k_i \cdot k_j$.
2. **Jaccard Index**: Neighborhood similarity.
3. **LKS (Link K-Shell)**: Core centrality product.
4. **CI (Collective Influence) Variants**: Calculated for $l=3$:
   - **CI_skin**: Nodes exactly at distance $l$.
   - **CI_body**: Nodes within the ball of radius $l$.
   - **Aggregations**: Average (`_e_av`) and Multiplication (`_e_mul`) for each node CI, resulting in 4 distinct columns.
5. **LLBCe & LLBMEe1**:
   - **LLBCe**: Link-Local Betweenness Centrality based on the first-order central domain.
   - **LLBMEe1**: Link-Local Betweenness Mapping Entropy.
   - Strictly used `nx.edge_betweenness_centrality_subset` as specified in `new_instructions.md`.

## 3. Findings
All metrics were aggregated into a unified Pandas DataFrame:
- **Result File**: `results_city/tokyo_grid_metrics.csv`
- **Visualization**: Dismantling curves for each metric (Relative Giant Component vs. Fraction of Edges Removed) were plotted in `results_city/`.
- **AUC Scores**: AUC values were computed for each metric to rank their effectiveness in identifying critical links for maintaining global connectivity.

---
*Prepared by Gemini CLI Hive Mind - April 2026*
