"""
Shared utilities for Network Dismantling Analysis.
Contains the core dismantling logic and plotting functions.
"""
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reproducibility: deterministic tie-breaks in EBC and any RNG-driven path.
# Matches Silam's reference scripts (np.random.seed(42)).
np.random.seed(42)

# Global Style Settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'

def create_metric_df(G, scores_dict, metric_name):
    """
    Standardize the network metrics recording object (Pandas dataframe).
    The first two columns should be named "i" and "j".
    """
    data = []
    for (u, v), val in scores_dict.items():
        data.append({'i': u, 'j': v, metric_name: val})
    return pd.DataFrame(data)

class NetworkDismantler:
    def __init__(self, graph):
        self.original_graph = graph.copy()
        self.N = graph.number_of_nodes()
        self.M = graph.number_of_edges()

    def get_static_curve(self, df_scores, metric_column, reverse=True, return_order=False):
        """
        Performs static decomposition based on a DataFrame of scores.

        When return_order is True, returns (rgc_values, removal_order) where
        removal_order is the list of (i, j) edges in the order they are removed.
        """
        G = self.original_graph.copy()

        # Sort edges (Reverse=True means Higher Score = Remove First).
        # Tie-break: edge tuple (i, j) descending — matches Silam's universal rule
        # `max((score, edge_tuple))`, which always prefers the larger edge tuple
        # regardless of whether the score itself is sorted ascending or descending.
        sorted_df = df_scores.sort_values(
            by=[metric_column, 'i', 'j'],
            ascending=[not reverse, False, False],
            kind='mergesort',
        )
        removal_order = list(zip(sorted_df['i'], sorted_df['j']))

        # Simulate Removal
        rgc_values = [1.0] # f=0

        for edge_to_remove in removal_order:
            if G.has_edge(*edge_to_remove):
                G.remove_edge(*edge_to_remove)

            if G.number_of_nodes() > 0:
                # Find connected components and get the size of the largest one
                ccs = list(nx.connected_components(G))
                if ccs:
                    largest_cc = max(ccs, key=len)
                    rgc = len(largest_cc) / self.N
                else:
                    rgc = 0.0
            else:
                rgc = 0.0

            rgc_values.append(rgc)

        if return_order:
            return rgc_values, removal_order
        return rgc_values

def _resolve_reverse(reverse, metric_col):
    """Resolve a bool-or-dict reverse spec for a given metric column.

    A dict lets different metric columns be dismantled in opposite directions
    (e.g. LLBCe high-first vs LLBMEe1 smallest-first); missing keys default True.
    """
    if isinstance(reverse, dict):
        return reverse.get(metric_col, True)
    return reverse

def run_and_plot(graph, method_name, rank_func, filename, reverse=True):
    """
    Helper to run analysis and save plot for a single method or multiple metrics in a DataFrame.

    reverse may be a bool (applied to every metric column) or a dict mapping
    column name -> bool, so columns returned by one rank_func can be removed in
    opposite directions.
    """
    print(f"Running analysis for: {method_name}...")
    dismantler = NetworkDismantler(graph)
    
    # rank_func should now return a DataFrame with columns 'i', 'j', and metric name(s)
    df_scores = rank_func(graph)
    
    # Metric columns are all columns except 'i' and 'j'
    metric_cols = [col for col in df_scores.columns if col not in ['i', 'j']]
    
    plt.figure(figsize=(10, 6))
    
    results = {}
    for metric_col in metric_cols:
        print(f"  - Processing metric: {metric_col}")
        col_reverse = _resolve_reverse(reverse, metric_col)
        rgc = dismantler.get_static_curve(df_scores, metric_col, reverse=col_reverse)

        # Calculate AUC
        x = np.linspace(0, 1, len(rgc))

        # Handle NumPy 2.0+ change (np.trapz -> np.trapezoid)
        if hasattr(np, "trapezoid"):
            auc = np.trapezoid(rgc, x)
        else:
            auc = np.trapz(rgc, x)

        results[metric_col] = {
            "x": x.tolist(),
            "y": rgc,
            "auc": float(auc)
        }

        # Plotting
        plt.plot(x, rgc, label=f"{metric_col} (AUC={auc:.3f})", linewidth=2.5)

    plt.title(f"Dismantling: {method_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Fraction of Edges Removed", fontsize=12)
    plt.ylabel("Relative Giant Component Size", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Completed. Plot saved to {filename}")
    return results

if __name__ == "__main__":
    print("NOTICE: This is a shared utility module, not a standalone script.")
    print("It contains common functions used by the other analysis scripts.")
    print("To generate graphs and results, please run the specific algorithm files:")
    print("  - run_ei.py")
    print("  - run_ci.py")
    print("  - run_cdc.py")
    print("  - etc.")
    print("\nMake sure all files are in the same directory.")