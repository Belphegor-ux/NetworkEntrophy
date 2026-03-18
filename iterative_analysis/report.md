# Iterative Dismantling Analysis Report (Updated March 10, 2026)

## 1. Methodology
Iterative dismantling recalculates the edge importance scores after each removal. This captures the dynamic shifts in the network topology as the giant component is broken down. The process is more computationally expensive than static dismantling but provides a more accurate representation of the network's resilience.

## 2. Updated Results (Zachary's Karate Club)
The iterative process now supports multi-metric comparisons in a single execution.

| Method | Metric | AUC (Iterative) |
| :--- | :--- | :--- |
| **LDC** | `LDC` | 0.621 |
| **CI (Skin)** | `CI_e_av_skin` | 0.612 |
| | `CI_e_mul_skin` | 0.528 |
| **CI (Body)** | `CI_e_av_body` | 0.628 |
| | `CI_e_mul_body` | 0.528 |
| **IE** | `IE` | 0.621 |
| **ME Improved** | `LLBMEe_improved` | 0.751 |
| **LLBC/LLBME** | `LLBCe` | 0.589 |
| | `LLBMEe1` | 0.618 |
| **LKS** | `LKS` | 0.728 |
| **Jaccard** | `Jaccard` | 0.854 |

## 3. Comparative Observations
- **Mapping Entropy Performance**: The `LLBMEe_improved` method consistently shows strong performance (AUC ~0.75), indicating its effectiveness in identifying critical nodes in the Karate Club network.
- **CI Body vs. Skin**: The `CI_e_av_body` version shows a slightly higher AUC (0.628) than the skin version (0.612), suggesting that including the full volume of the ball of radius $l=3$ provides a more comprehensive importance score in this topology.
- **Aggregation Strategy**: In both CI versions, the **average** aggregation (`CI_e_av`) significantly outperformed the multiplication (`CI_e_mul`) in terms of AUC.
- **Jaccard Index**: Interestingly, the Jaccard Index (calculated iteratively) shows a high AUC (0.854), though it's important to note that Jaccard removal is based on *low* similarity scores (removing bridges first).

## 4. Technical Improvements
- **Iterative Framework**: `network_utils_iter.py` was refactored to support the standardized DataFrame format.
- **Canonical Edge Handling**: The iterative loop now uses canonical representations for edge lookups to ensure robustness across different graph states.
- **Modular Imports**: Each iterative script now imports the core ranking logic from the main project scripts, reducing code duplication and ensuring that static and iterative results are based on identical definitions.
