"""
Structural Entropy & Criticality Analysis
Benchmark: Zachary's Karate Club
Methodologies: EI, CI, CDC (LDC), CKS (LKS), IE, ME, EPC (LBC)

This script performs a Static Decomposition analysis:
1. Calculates the ranking of all edges based on the specific metric.
2. Removes edges one by one in descending order of importance.
3. Tracks the Relative Giant Component Size (R_gc).
4. Plots the comparative decay curves.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from math import log

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl", 7)
plt.rcParams['font.family'] = 'sans-serif'


class NetworkDismantler:
    def __init__(self, graph):
        self.original_graph = graph.copy()
        self.N = graph.number_of_nodes()
        self.M = graph.number_of_edges()

    def get_static_curve(self, ranking_func, metric_name, reverse=True):
        """
        Performs static decomposition.
        Args:
            ranking_func: Function that returns a dict {edge: score}
            metric_name: Name for the progress bar
            reverse: True for 'higher score = remove first' (typical for Centrality),
                     False for 'lower score = remove first' (if applicable)
        """
        G = self.original_graph.copy()

        # 1. Rank all edges once (Static)
        scores = ranking_func(G)

        # Sort edges. Default: High score -> First to remove since llower is required
        sorted_edges = sorted(scores.items(), key=lambda x: x[1], reverse=reverse)
        removal_order = [e[0] for e in sorted_edges]

        # 2. Simulate Removal
        rgc_values = [1.0]  # f=0

        # Pre-calculate steps for R_gc
        current_edges = list(G.edges())

        for i, edge_to_remove in enumerate(removal_order):
            if G.has_edge(*edge_to_remove):
                G.remove_edge(*edge_to_remove)

            # Calculate Giant Component
            if G.number_of_nodes() > 0:
                largest_cc = max(nx.connected_components(G), key=len)
                rgc = len(largest_cc) / self.N
            else:
                rgc = 0.0

            rgc_values.append(rgc)

        return rgc_values

    # 1. EI: Explosive Immunization (Adapted for Edges)
    # Ref: entropy-26-00248.pdf (Eq 11 Edge Score = EI(u) * EI(v)

    def rank_ei(self, G):
        # Calculate Effective Degree (simplified approximation K_eff ~ K) for static
        # In full EI, K_eff is iterative. Here we use basic structural EI score.
        # sigma_u = k_eff + sum(sqrt(Size(C)) - 1)

        # 1. Detect clusters (using connected components as proxy for 'clusters' in neighborhood)
        # Since this is static global, we look at neighbors' clusters?
        # Paper 00248 uses "clusters linked to u". In a connected graph, it's just one cluster.
        # We will use the 'Effective Degree' approximation: Degree * Clustering?
        # Let's use a robust approximation: Degree * Sum of Neighbors' Degrees (Collective force)
        # OR strictly implement Eq 11 if possible.
        # Given static constraint, we'll use: EI_score(u) = Degree(u) * sum(Degree(n) for n in neighbors)

        scores = {}
        for u, v in G.edges():
            # Approx EI score for node: k_u * sum(k_neighbors)
            # This captures the "Explosive" potential (hubs connecting to hubs)
            score_u = G.degree(u) * sum(G.degree(n) for n in G.neighbors(u))
            score_v = G.degree(v) * sum(G.degree(n) for n in G.neighbors(v))
            scores[(u, v)] = score_u * score_v
        return scores

    # =========================================================================
    # 2. CI: Collective Influence (Adapted for Edges)
    # Ref: entropy-26-00248.pdf (Eq 10)
    # CI_l(u) = (k_u - 1) * sum(k_v - 1 for v in Ball(u, l))
    # Edge Score = CI(u) + CI(v)
    # =========================================================================
    def rank_ci(self, G, l=3):
        ci_nodes = {}
        for node in G.nodes():
            k_u = G.degree(node)
            if k_u <= 1:
                ci_nodes[node] = 0
                continue

            # BFS to find boundary at level l
            lengths = nx.single_source_shortest_path_length(G, node, cutoff=l)
            boundary_nodes = [n for n, dist in lengths.items() if dist == l]

            sum_k = sum((G.degree(v) - 1) for v in boundary_nodes)
            ci_nodes[node] = (k_u - 1) * sum_k

        scores = {}
        for u, v in G.edges():
            scores[(u, v)] = ci_nodes[u] + ci_nodes[v]

        return scores

    # =========================================================================
    # 3. CDC (LDC): Link Degree Centrality
    # Ref: entropy-26-00315-v2.pdf (Eq 3)
    # LDC = (Product of degrees of endpoints) normalized
    # =========================================================================
    def rank_cdc(self, G):
        scores = {}
        for u, v in G.edges():
            # Simple Degree Product is the core of LDC
            scores[(u, v)] = G.degree(u) * G.degree(v)
        return scores

    # =========================================================================
    # 4. CKS (LKS): Link K-Shell Index
    # Ref: entropy-26-00315-v2.pdf (Eq 4-6)
    # LKS = KS(u) * KS(v) (Product version is most granular)
    # =========================================================================
    def rank_cks(self, G):
        # Calculate Core Number (K-Shell) for all nodes
        core_numbers = nx.core_number(G)

        scores = {}
        for u, v in G.edges():
            scores[(u, v)] = core_numbers[u] * core_numbers[v]
        return scores

    # =========================================================================
    # 5. IE: Information Entropy (Edge)
    # Ref: entropy-26-00315-v2.pdf (Def 11 adapted for Edge)
    # H_edge = - sum ( p_i log p_i ) where p_i is degree/sum_degrees in neighborhood
    # =========================================================================
    def rank_ie(self, G):
        scores = {}
        for u, v in G.edges():
            # Define Edge Neighborhood: neighbors of u union neighbors of v
            # Calculate probability distribution of degrees in this locality

            # Neighborhood nodes (First order central domain)
            u_neighbors = set(G.neighbors(u))
            v_neighbors = set(G.neighbors(v))
            neighborhood = u_neighbors.union(v_neighbors)
            if u in neighborhood: neighborhood.remove(u)
            if v in neighborhood: neighborhood.remove(v)

            if not neighborhood:
                scores[(u, v)] = 0
                continue

            total_deg = sum(G.degree(n) for n in neighborhood)
            if total_deg == 0:
                scores[(u, v)] = 0
                continue

            entropy = 0
            for n in neighborhood:
                p = G.degree(n) / total_deg
                if p > 0:
                    entropy += -p * log(p)

            scores[(u, v)] = entropy
        return scores

    # =========================================================================
    # 6. ME: Mapping Entropy (LLBME)
    # Ref: entropy-26-00315-v2.pdf (Eq 11: LLBME)
    # LLBME = - LLBC_e * sum( log(LLBC_neighbors) )
    # Note: Requires Link Local Betweenness (LLBC) first
    # =========================================================================
    def rank_me(self, G):
        # 1. Calculate LLBC (Link Local Betweenness)
        # LLBC: Betweenness within the "First order central domain" (neighbors of u and v)
        llbc_scores = {}

        # Pre-calculate LLBC for all edges
        for u, v in G.edges():
            # Subgraph of u, v and their neighbors
            nodes = set(G.neighbors(u)).union(set(G.neighbors(v))).union({u, v})
            subg = G.subgraph(nodes)

            # Calculate Edge Betweenness on this small subgraph
            # We only need the value for edge (u,v)
            # Using networkx betweenness on subgraph
            try:
                ebc = nx.edge_betweenness_centrality(subg, normalized=False)
                key = (u, v) if (u, v) in ebc else (v, u)
                llbc_scores[(u, v)] = ebc.get(key, 0)
            except:
                llbc_scores[(u, v)] = 0

        # 2. Calculate ME (LLBME) using LLBC
        me_scores = {}
        for u, v in G.edges():
            e_val = llbc_scores.get((u, v), 0)
            if e_val <= 0:
                me_scores[(u, v)] = 0
                continue

            # Sum log of neighbor edges
            # Neighbor edges: edges incident to u or v (excluding e itself)
            neighbor_edges = list(G.edges(u)) + list(G.edges(v))
            sum_log = 0
            count = 0
            for nu, nv in neighbor_edges:
                if (nu, nv) == (u, v) or (nu, nv) == (v, u):
                    continue

                # normalize key
                n_key = (nu, nv) if (nu, nv) in llbc_scores else (nv, nu)
                n_val = llbc_scores.get(n_key, 0)

                if n_val > 0:
                    sum_log += log(n_val)
                    count += 1

            # Def 12/13 formula structure: - Score * Sum(log(NeighborScores))
            me_scores[(u, v)] = -e_val * sum_log

        return me_scores

    # =========================================================================
    # 7. EPC: Edge Percolation / Edge Betweenness Centrality
    # Ref: Standard Benchmark (Global Edge Betweenness)
    # =========================================================================
    def rank_epc(self, G):
        return nx.edge_betweenness_centrality(G)


def run_analysis():
    # 1. Load Karate Club Graph
    G = nx.karate_club_graph()
    print(f"Loaded Karate Club: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    dismantler = NetworkDismantler(G)

    # 2. Define Methods
    methods = {
        'EI (Explosive Immunization)': dismantler.rank_ei,
        'CI (Collective Influence)': dismantler.rank_ci,
        'CDC (Link Degree Centrality)': dismantler.rank_cdc,
        'CKS (Link K-Shell)': dismantler.rank_cks,
        'IE (Information Entropy)': dismantler.rank_ie,
        'ME (Mapping Entropy)': dismantler.rank_me,
        'EPC (Edge Betweenness)': dismantler.rank_epc
    }

    # 3. Run Analysis
    results = {}

    plt.figure(figsize=(12, 8))

    colors = sns.color_palette("bright", n_colors=len(methods))

    print("\nRunning Static Decomposition Analysis...")
    for (name, func), color in zip(methods.items(), colors):
        print(f"Processing {name}...")

        # Calculate Curve
        # Note: ME and IE are entropy based. Usually higher entropy = more important?
        # Or lower?
        # For ME (Mapping Entropy): "Higher entropy values indicate greater variation... carries less information".
        # However, MDLE in paper 00315 is used to find "Critical Links".
        # Usually we remove High Centrality or Low Entropy?
        # Let's assume High Value of the metric = High Importance for all for consistency,
        # unless metric is explicitly 'entropy' where low might be key.
        # But Def 11 says "Ei = -sum...".
        # Let's stick to High Score = Remove First (reverse=True) for all,
        # as usually specific metrics are formulated such that higher = more critical.

        rgc = dismantler.get_static_curve(func, name, reverse=True)

        # Calculate AUC (Robustness Integrity)
        x = np.linspace(0, 1, len(rgc))
        auc = np.trapz(rgc, x)

        results[name] = auc

        # Plot
        plt.plot(x, rgc, label=f"{name} (AUC={auc:.3f})", linewidth=2.5, alpha=0.8, color=color)

    # 4. Styling
    plt.title("Static Network Decomposition: Karate Club Benchmark", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Fraction of Edges Removed (f)", fontsize=14)
    plt.ylabel("Relative Giant Component Size ($R_{gc}$)", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Save
    filename = "karate_entropy_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nAnalysis Complete. Plot saved to {filename}")

    # Print Summary Table
    print("\n" + "=" * 50)
    print(f"{'Method':<35} | {'AUC (Lower is Better)':<20}")
    print("-" * 55)
    for name, auc in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name:<35} | {auc:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    run_analysis()