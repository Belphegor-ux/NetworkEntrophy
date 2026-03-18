import networkx as nx
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
    # Test on Karate Club
    print("Testing Improved LLBMe on Karate Club...")
    G_karate = nx.karate_club_graph()
    run_and_plot(G_karate, "Improved LLBMe", rank_me_improved, "result_me_improved_karate.png")

    # Test on Football Club (if available)
    try:
        # Assuming football.gml might be in the root or graph folder
        import os
        path = "football.gml"
        if not os.path.exists(path):
            path = os.path.join("graph", "football.gml")
        
        if os.path.exists(path):
            print(f"Testing Improved LLBMe on Football Club ({path})...")
            G_foot = nx.read_gml(path)
            run_and_plot(G_foot, "Improved LLBMe (Football)", rank_me_improved, "result_me_improved_football.png")
        else:
            print("football.gml not found. Skipping football test.")
    except Exception as e:
        print(f"Error loading football dataset: {e}")
