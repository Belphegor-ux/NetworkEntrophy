import networkx as nx
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
import numpy as np
import pandas as pd
from network_utils import run_and_plot

def rank_llbc_me(G):
    """
    Implements LLBCe (Equation 8) and LLBMEe1 (Equation 11) from MDPI Entropy 26(4), 315.
    
    LLBCe: Link-Local Betweenness Centrality using subset betweenness on the full graph.
    LLBMEe1: Link-Local Betweenness Mapping Entropy.
    """
    nodes = list(G.nodes())
    edges = list(G.edges())
    
    llbc_e = {}
    
    # 1. Calculate LLBCe for each edge
    for u, v in edges:
        # Canonical key
        en = tuple(sorted((u, v)))
        
        # FC_G[e] = first-order central domain (nodes of e and their neighbors)
        fc_nodes = set([u, v]).union(G.neighbors(u)).union(G.neighbors(v))
        fc_nodes = list(fc_nodes)
        
        # Use subset betweenness on the FULL graph as instructed
        # sources=fc_nodes, targets=fc_nodes
        ebc_subset = nx.edge_betweenness_centrality_subset(
            G, sources=fc_nodes, targets=fc_nodes, normalized=False
        )
        
        # Get score for the current edge e
        score = ebc_subset.get((u, v), ebc_subset.get((v, u), 0))
        llbc_e[en] = score

    # 2. Calculate LLBMEe1 for each edge
    llbme_e1 = {}
    for u, v in edges:
        e1 = tuple(sorted((u, v)))
        # Gamma(e1) = neighboring links (sharing a node with e1)
        # Including e1 itself
        neighbors_u = [(u, n) for n in G.neighbors(u)]
        neighbors_v = [(v, n) for n in G.neighbors(v)]
        gamma_e1_edges = set()
        for e in neighbors_u + neighbors_v:
            # Canonical edge representation
            en = tuple(sorted(e))
            gamma_e1_edges.add(en)
        
        # Get LLBC values for these edges
        local_llbc_vals = []
        for e in gamma_e1_edges:
            val = llbc_e.get(e, 0)
            local_llbc_vals.append(val)
            
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

    # 3. Create DataFrame
    data = []
    for u, v in edges:
        e_can = tuple(sorted((u, v)))
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
    run_and_plot(G, "LLBC and LLBME", rank_llbc_me, out_name)
