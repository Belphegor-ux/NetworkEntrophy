"""
Shared utilities for Network Dismantling Analysis.
Contains the core dismantling logic and plotting functions.
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Global Style Settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'

class NetworkDismantler:
    def __init__(self, graph):
        self.original_graph = graph.copy()
        self.N = graph.number_of_nodes()
        self.M = graph.number_of_edges()

    def get_static_curve(self, ranking_func, reverse=True):
        """
        Performs static decomposition based on a ranking function.
        """
        G = self.original_graph.copy()

        # Rank all edges once
        scores = ranking_func(G)

        # Sort edges (Reverse=True means Higher Score = Remove First)
        sorted_edges = sorted(scores.items(), key=lambda x: x[1], reverse=reverse)
        removal_order = [e[0] for e in sorted_edges]

        # Simulate Removal
        rgc_values = [1.0] # f=0

        for i, edge_to_remove in enumerate(removal_order):
            if G.has_edge(*edge_to_remove):
                G.remove_edge(*edge_to_remove)

            if G.number_of_nodes() > 0:
                largest_cc = max(nx.connected_components(G), key=len)
                rgc = len(largest_cc) / self.N
            else:
                rgc = 0.0

            rgc_values.append(rgc)

        return rgc_values

def run_and_plot(graph, method_name, rank_func, filename, reverse=True):
    """
    Helper to run analysis and save plot for a single method.
    """
    print(f"Running analysis for: {method_name}...")
    dismantler = NetworkDismantler(graph)
    rgc = dismantler.get_static_curve(rank_func, reverse=reverse)

    # Calculate AUC
    x = np.linspace(0, 1, len(rgc))

    # Handle NumPy 2.0+ change (np.trapz -> np.trapezoid)
    if hasattr(np, "trapezoid"):
        auc = np.trapezoid(rgc, x)
    else:
        auc = np.trapz(rgc, x)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(x, rgc, label=f"{method_name} (AUC={auc:.3f})", linewidth=2.5, color='#e74c3c')

    plt.title(f"Dismantling: {method_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Fraction of Edges Removed", fontsize=12)
    plt.ylabel("Relative Giant Component Size", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Completed. AUC: {auc:.4f}. Plot saved to {filename}")

if __name__ == "__main__":
    print("NOTICE: This is a shared utility module, not a standalone script.")
    print("It contains common functions used by the other analysis scripts.")
    print("To generate graphs and results, please run the specific algorithm files:")
    print("  - run_ei.py")
    print("  - run_ci.py")
    print("  - run_cdc.py")
    print("  - etc.")
    print("\nMake sure all files are in the same directory.")