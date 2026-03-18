# Network Entropy Analysis Report (Updated March 10, 2026)

## 1. Project Overview
This project focuses on identifying critical links in networks using various topological metrics and entropy-based measures. The codebase has been standardized to ensure consistency across different analysis methods and to support both static and iterative network dismantling.

## 2. Methodology & Standardization
### Data Representation
To avoid disparities in graphics and analysis, all network metric recording objects have been standardized to **Pandas DataFrames**:
- **Columns "i" and "j"**: Represent the nodes of existing edges.
- **Metric Columns**: Each calculation method adds its specific scores as additional columns in the same DataFrame.
- This format ensures compatibility with common dataset storage (CSV/Excel) and consistent processing by the `NetworkDismantler`.

### File Organization
Scripts have been renamed to use underscores for better Python compatibility and easier importing into the iterative analysis framework:
- `LDC_link_degree_centrality.py`
- `jaccard_index.py`
- `CKS_Link_K_Shell.py`
- `collective_influence.py`
- `LLBC_contrast.py` (Now implements LLBCe and LLBMEe1 from MDPI Entropy 26(4), 315)
- `IE_informative_entropy.py`
- `ME_mapping_entropy.py` (Improved LLBMe)

---

## 3. Implemented Methods

| Method | Metric Column(s) | Description |
| :--- | :--- | :--- |
| **LDC** | `LDC` | Link Degree Centrality: $Degree(u) \times Degree(v)$. |
| **Jaccard** | `Jaccard` | Jaccard Index: $\frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}$. |
| **LKS** | `LKS` | Link K-Shell Index: $Core(u) \times Core(v)$. |
| **IE** | `IE` | Information Entropy based on neighborhood degree distribution. |
| **LLBME Improved** | `LLBMEe_improved` | Improved LLBMe with Jaccard weighting and epsilon smoothing. |
| **LLBCe** | `LLBCe` | Link-Local Betweenness Centrality (MDPI Entropy 26(4), 315, Eq. 8). |
| **LLBMEe1** | `LLBMEe1` | Link-Local Betweenness Mapping Entropy (MDPI Entropy 26(4), 315, Eq. 11). |
| **CI (Skin)** | `CI_e_av_skin`, `CI_e_mul_skin` | Collective Influence based on nodes exactly at distance $l=3$. |
| **CI (Body)** | `CI_e_av_body`, `CI_e_mul_body` | Collective Influence based on nodes within distance $l=3$. |

---

## 4. Key Updates

### Collective Influence (CI) Refactoring
Following the clarification of the CI definition ($l=3$), we now calculate both:
- **CI_skin**: Focuses on nodes at the $l$-th order boundary.
- **CI_body**: Includes all nodes within the ball of radius $l$.
Edge importance is derived using both the **average** and **multiplication** of node CI values.

### LLBCe & LLBMEe1 Implementation
Implemented the specific local betweenness metrics from the MDPI paper "Research on a Critical Link Discovery Method for Network Security Situational Awareness":
- Calculations use `nx.edge_betweenness_centrality_subset` on the full graph, focused on the first-order central domain of each link.

### Removal of EI Method
The EI (Explosive Immunization) method was removed due to flaws and ambiguity in its published description.

---

## 5. Verification Results (Karate Club)
All static methods were verified for successful execution. The `run_and_plot` utility now automatically generates comparative plots for multi-metric outputs (like CI and LLBC/LLBME).

| Metric | AUC (Static) |
| :--- | :--- |
| LDC | 0.612 |
| Jaccard | 0.855 |
| LKS | 0.729 |
| CI_e_av_skin | 0.609 |
| CI_e_mul_skin | 0.518 |
| CI_e_av_body | 0.627 |
| CI_e_mul_body | 0.523 |
| LLBCe | 0.592 |
| LLBMEe1 | 0.610 |
| IE | 0.621 |
| LLBMEe_improved | 0.742 |

---

## 6. Conclusion
The codebase is now fully standardized and aligned with the latest requirements. The addition of the "body" version of CI and the implementation of the specified LLBC/LLBME metrics provides a more comprehensive suite for network vulnerability and critical link analysis.
