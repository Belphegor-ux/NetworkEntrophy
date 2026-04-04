import networkx as nx
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
from math import log
import numpy as np
from network_utils import run_and_plot, create_metric_df

def rank_me_improved(G):
    """
    Improved Link-Local Mapping Betweenness Entropy (LLBME).
    Addresses the 'mesh subgraph issue' with Jaccard weighting and epsilon smoothing.
    """
    # 1. Calculating LLBC with local subgraphs
    llbc_scores = {}
    for u, v in G.edges():
        nodes = set(G.neighbors(u)).union(set(G.neighbors(v))).union({u, v})
        subg = G.subgraph(nodes)
        try:
            ebc = nx.edge_betweenness_centrality(subg, normalized=False)
            key = (u, v) if (u, v) in ebc else (v, u)
            llbc_scores[(u, v)] = ebc.get(key, 0)
        except:
            llbc_scores[(u, v)] = 0

    # 2. Calculate Jaccard similarity for topological weighting
    jaccard_scores = {}
    for u, v in G.edges():
        u_neighbors = set(G.neighbors(u))
        v_neighbors = set(G.neighbors(v))
        intersection = len(u_neighbors.intersection(v_neighbors))
        union = len(u_neighbors.union(v_neighbors))
        jaccard_scores[(u, v)] = intersection / union if union > 0 else 0

    # 3. Calculate Improved Mapping Entropy
    me_scores = {}
    epsilon = 1e-6 # Smoothing factor to prevent zero values in log
    for u, v in G.edges():
        e_val = llbc_scores.get((u, v), 0)
        if e_val <= 0:
            me_scores[(u, v)] = 0
            continue

        neighbor_edges = list(G.edges(u)) + list(G.edges(v))
        sum_log = 0
        for nu, nv in neighbor_edges:
            if (nu, nv) == (u, v) or (nu, nv) == (v, u):
                continue

            n_key = (nu, nv) if (nu, nv) in llbc_scores else (nv, nu)
            n_val = llbc_scores.get(n_key, 0)

            # Apply epsilon smoothing for neighbors
            sum_log += log(max(n_val, epsilon))

        # Weight by (1 - Jaccard) to penalize mesh edges (high overlap)
        # and favor bridge edges (low overlap)
        me_scores[(u, v)] = -e_val * sum_log * (1 - jaccard_scores[(u, v)])

    return create_metric_df(G, me_scores, "LLBMEe_improved")

if __name__ == "__main__":

    os.makedirs('results', exist_ok=True)
    G = nx.karate_club_graph()
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    out_name = "results/result_me.png"
    run_and_plot(G, "Improved LLBMe", rank_me_improved, out_name)
