import networkx as nx
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
from math import log
import numpy as np
from network_utils import run_and_plot, create_metric_df

def rank_me_improved(G):
    """
    Link-Local Mapping Betweenness Entropy (LLBMEe1) per Eq. 11 of
    MDPI Entropy 26(4) 315.

    LLBCe(e=(u,v)) is computed via nx.edge_betweenness_centrality_subset
    over the first-order central domain fc = {u,v} U N(u) U N(v) as both
    sources and targets, then normalized by |fc| * (|fc| - 1) (Eq. 8).
    Subgraph extraction is explicitly forbidden by the spec.

    LLBMEe1(e1) = -sum_p p log p over LLBCe values of edges incident to
    the endpoints of e1, where p = LLBCe(e_k) / sum(LLBCe in Gamma(e1)).
    """
    edges = [tuple(sorted(e)) for e in G.edges()]
    node_to_neighbors = {n: set(G.neighbors(n)) for n in G.nodes()}

    # 1. LLBCe via edge_betweenness_centrality_subset over fc = {u,v} U N(u) U N(v)
    llbc_e = {}
    for u, v in edges:
        fc = {u, v} | node_to_neighbors[u] | node_to_neighbors[v]
        fc_nodes = list(fc)
        ebc_subset = nx.edge_betweenness_centrality_subset(
            G, sources=fc_nodes, targets=fc_nodes, normalized=False
        )
        raw = ebc_subset.get((u, v), ebc_subset.get((v, u), 0))
        denom = len(fc_nodes) * (len(fc_nodes) - 1)
        llbc_e[(u, v)] = raw / denom if denom > 0 else 0.0

    # 2. LLBMEe1: entropy over LLBCe of edges incident to endpoints of e1
    llbme_e1 = {}
    for u, v in edges:
        gamma_e1_edges = set()
        for e in list(G.edges(u)) + list(G.edges(v)):
            gamma_e1_edges.add(tuple(sorted(e)))

        local_vals = [llbc_e.get(e, 0) for e in gamma_e1_edges]
        sum_llbc = sum(local_vals)

        if sum_llbc == 0:
            llbme_e1[(u, v)] = 0.0
        else:
            entropy = 0.0
            for val in local_vals:
                if val > 0:
                    p = val / sum_llbc
                    entropy -= p * np.log(p)
            llbme_e1[(u, v)] = entropy

    return create_metric_df(G, llbme_e1, "LLBMEe1")

if __name__ == "__main__":

    os.makedirs('results', exist_ok=True)
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/football/football.gml'))
    G = nx.read_gml(dataset_path, label='id')
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    out_name = "results/result_me.png"
    run_and_plot(G, "Improved LLBMe", rank_me_improved, out_name)
