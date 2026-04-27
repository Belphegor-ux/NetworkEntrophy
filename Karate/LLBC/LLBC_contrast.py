import networkx as nx
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
import numpy as np
import pandas as pd
from network_utils import run_and_plot

def rank_llbc_me(G):
    """
    Optimized implementation of LLBCe and LLBMEe1 using parallel processing.
    """
    import collections
    from concurrent.futures import ThreadPoolExecutor
    
    edges = [tuple(sorted(e)) for e in G.edges()]
    node_to_neighbors = {n: set(G.neighbors(n)) for n in G.nodes()}
    
    edge_to_fc = {}
    for u, v in edges:
        fc = {u, v} | node_to_neighbors[u] | node_to_neighbors[v]
        edge_to_fc[(u, v)] = list(fc)

    def get_score(e):
        u, v = e
        fc_nodes = edge_to_fc[e]
        ebc_subset = nx.edge_betweenness_centrality_subset(G, sources=fc_nodes, targets=fc_nodes, normalized=False)
        return e, ebc_subset.get((u, v), ebc_subset.get((v, u), 0))

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(get_score, edges))
    
    llbc_e = dict(results)

    llbme_e1 = {}
    for u, v in edges:
        e1 = (u, v)
        neighbors_u = G.edges(u)
        neighbors_v = G.edges(v)
        gamma_e1_edges = set()
        for e in list(neighbors_u) + list(neighbors_v):
            gamma_e1_edges.add(tuple(sorted(e)))
        
        local_llbc_vals = [llbc_e.get(e, 0) for e in gamma_e1_edges]
        sum_llbc = sum(local_llbc_vals)
        
        if sum_llbc == 0:
            llbme_e1[e1] = 0
        else:
            entropy = 0
            for val in local_llbc_vals:
                if val > 0:
                    p = val / sum_llbc
                    entropy -= p * np.log(p)
            llbme_e1[e1] = entropy

    data = []
    for u, v in edges:
        e_can = (u, v)
        data.append({
            'i': u,
            'j': v,
            'LLBCe': llbc_e[e_can],
            'LLBMEe1': llbme_e1[e_can]
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":

    os.makedirs('results', exist_ok=True)
    G = nx.karate_club_graph()
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    out_name = "results/result_llbc.png"
    run_and_plot(G, "LLBCe and LLBMEe1", rank_llbc_me, out_name)
