import warnings

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reproducibility — match Silam's reference scripts.
np.random.seed(42)

def _resolve_reverse(reverse, metric_col):
    """Resolve a bool-or-dict reverse spec for a given metric column.

    A dict lets different metric columns be dismantled in opposite directions
    (e.g. LLBCe high-first vs LLBMEe1 smallest-first); missing keys default True.
    """
    if isinstance(reverse, dict):
        return reverse.get(metric_col, True)
    return reverse

def get_iterative_curve(graph, rank_func, metric_column, reverse=True, step_size=1,
                        return_order=False):
    """
    Performs iterative decomposition. Recalculates ranks after 'step_size' removals.
    Returns RGC samples of length original_m + 1 (one per edge removal plus f=0),
    padding with the last observed RGC if the rank function ever returns empty
    while edges remain.

    When return_order is True, returns (rgc_values, removal_order) where
    removal_order is the list of (i, j) edges in the order they were removed.
    """
    G = graph.copy()
    N = G.number_of_nodes()
    rgc_values = [1.0] # f=0
    removal_order = []

    # Total edges to remove
    original_m = G.number_of_edges()

    while G.number_of_edges() > 0:
        # RECALCULATE ranks for current state of G
        df_scores = rank_func(G)

        # Sort and pick the TOP edges for the specified metric.
        # Tie-break: edge tuple (i, j) descending — matches Silam's universal rule
        # `max((score, edge_tuple))`, which always prefers the larger edge tuple
        # regardless of whether the score is sorted ascending or descending.
        sorted_df = df_scores.sort_values(
            by=[metric_column, 'i', 'j'],
            ascending=[not reverse, False, False],
            kind='mergesort',
        )
        if sorted_df.empty:
            warnings.warn(
                f"rank_func returned an empty DataFrame with "
                f"{G.number_of_edges()} edges still present "
                f"(metric={metric_column}); padding RGC curve with last value.",
                RuntimeWarning,
                stacklevel=2,
            )
            last = rgc_values[-1] if rgc_values else 0.0
            while len(rgc_values) < original_m + 1:
                rgc_values.append(last)
            break
            
        # Remove up to 'step_size' edges
        edges_to_remove = sorted_df.head(step_size)
        
        for _, row in edges_to_remove.iterrows():
            edge = (row['i'], row['j'])
            if G.has_edge(*edge):
                G.remove_edge(*edge)
            removal_order.append(edge)

            # Calculate RGC after each removal for a smooth curve
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
            
            if G.number_of_edges() == 0:
                break

    if return_order:
        return rgc_values, removal_order
    return rgc_values

def run_iterative_benchmark(graph, method_name, rank_func, filename=None, reverse=True,
                            step_size=1, order_csv_path=None):
    print(f"Running ITERATIVE analysis for: {method_name} (step_size={step_size})...")

    # Get all metric columns from a single call to rank_func
    df_sample = rank_func(graph)
    metric_cols = [col for col in df_sample.columns if col not in ['i', 'j']]

    if filename:
        plt.figure(figsize=(10, 6))

    results = {}
    order_records = {}  # metric_col -> {edge(sorted tuple): removal_order_int}

    for metric_col in metric_cols:
        print(f"  - Iterative processing metric: {metric_col}")
        col_reverse = _resolve_reverse(reverse, metric_col)
        rgc, removal_order = get_iterative_curve(
            graph, rank_func, metric_col, reverse=col_reverse,
            step_size=step_size, return_order=True,
        )

        # Record the removal ORDER (1-based) per edge. Iterative metric values are
        # not comparable across steps, so order is the meaningful record.
        order_records[metric_col] = {
            tuple(sorted(edge)): pos
            for pos, edge in enumerate(removal_order, start=1)
        }

        x = np.linspace(0, 1, len(rgc)).tolist()
        if hasattr(np, "trapezoid"):
            auc = np.trapezoid(rgc, x)
        else:
            auc = np.trapz(rgc, x)

        if filename:
            plt.plot(x, rgc, label=f"Iterative {metric_col} (AUC={auc:.3f})", linewidth=2.5)

        results[metric_col] = {
            'x': x,
            'y': rgc,
            'auc': float(auc)
        }

    if order_csv_path and order_records:
        all_edges = [tuple(sorted(e)) for e in graph.edges()]
        rows = []
        for (i, j) in all_edges:
            row = {'i': i, 'j': j}
            for metric_col in metric_cols:
                row[f'order_{metric_col}'] = order_records[metric_col].get((i, j))
            rows.append(row)
        pd.DataFrame(rows).to_csv(order_csv_path, index=False)
        print(f"  Saved iterative removal-order CSV to {order_csv_path}")

    if filename:
        plt.title(f"Iterative Dismantling: {method_name}")
        plt.xlabel("Fraction of Edges Removed")
        plt.ylabel("Relative Giant Component Size")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Completed {method_name} iterative plots.")
        
    return results

