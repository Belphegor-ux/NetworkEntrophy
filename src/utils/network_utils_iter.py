import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_iterative_curve(graph, rank_func, metric_column, reverse=True):
    """
    Performs iterative decomposition. Recalculates ranks after EACH removal.
    """
    G = graph.copy()
    N = G.number_of_nodes()
    rgc_values = [1.0] # f=0

    # Total edges to remove
    original_m = G.number_of_edges()
    
    for i in range(original_m):
        if G.number_of_edges() == 0:
            break
            
        # RECALCULATE ranks for current state of G
        df_scores = rank_func(G)
        
        # Sort and pick the TOP edge for the specified metric
        sorted_df = df_scores.sort_values(by=metric_column, ascending=not reverse)
        if sorted_df.empty:
            break
            
        edge_to_remove = (sorted_df.iloc[0]['i'], sorted_df.iloc[0]['j'])
        
        G.remove_edge(*edge_to_remove)

        # Calculate RGC
        if G.number_of_nodes() > 0:
            comps = list(nx.connected_components(G))
            if comps:
                largest_cc = max(comps, key=len)
                rgc = len(largest_cc) / N
            else:
                rgc = 0.0
        else:
            rgc = 0.0

        rgc_values.append(rgc)

    return rgc_values

def run_iterative_benchmark(graph, method_name, rank_func, filename, reverse=True):
    print(f"Running ITERATIVE analysis for: {method_name}...")
    
    # Get all metric columns from a single call to rank_func
    df_sample = rank_func(graph)
    metric_cols = [col for col in df_sample.columns if col not in ['i', 'j']]
    
    plt.figure(figsize=(10, 6))
    
    for metric_col in metric_cols:
        print(f"  - Iterative processing metric: {metric_col}")
        rgc = get_iterative_curve(graph, rank_func, metric_col, reverse=reverse)
        
        x = np.linspace(0, 1, len(rgc))
        if hasattr(np, "trapezoid"):
            auc = np.trapezoid(rgc, x)
        else:
            auc = np.trapz(rgc, x)

        plt.plot(x, rgc, label=f"Iterative {metric_col} (AUC={auc:.3f})", linewidth=2.5)

    plt.title(f"Iterative Dismantling: {method_name}")
    plt.xlabel("Fraction of Edges Removed")
    plt.ylabel("Relative Giant Component Size")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Completed {method_name} iterative plots.")

