import networkx as nx
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
from math import log
import numpy as np
from network_utils import run_and_plot, create_metric_df

def rank_me_improved(G):
    """
    Link-Local Mapping Betweenness Entropy (LLBMEe1).

    LLBCe(e=(u,v)) is the RAW edge betweenness from
    nx.edge_betweenness_centrality_subset over the first-order central
    domain fc = {u,v} U N(u) U N(v) as both sources and targets
    (normalized=False). It is NOT divided by |fc| * (|fc| - 1).
    Subgraph extraction is explicitly forbidden by the spec.

    LLBMEe(e1) = -LLBCe(e1) * sum_{e2 in neighbor edges of e1} log(LLBCe(e2))

    where the neighbor edges of e1=(u,v) are the edges incident to u
    (excluding the edge to v) and incident to v (excluding the edge to u);
    e1 itself is excluded. log uses np.log(max(value, 1e-10)). The score is
    0 when LLBCe(e1) <= 0. Values are NEGATIVE, smallest = most critical, so
    dismantling must be smallest-first (reverse=False).
    """
    edges = [tuple(sorted(e)) for e in G.edges()]
    node_to_neighbors = {n: set(G.neighbors(n)) for n in G.nodes()}

    # 1. RAW LLBCe via edge_betweenness_centrality_subset over fc = {u,v} U N(u) U N(v)
    llbc_e = {}
    for u, v in edges:
        fc = {u, v} | node_to_neighbors[u] | node_to_neighbors[v]
        fc_nodes = list(fc)
        ebc_subset = nx.edge_betweenness_centrality_subset(
            G, sources=fc_nodes, targets=fc_nodes, normalized=False
        )
        llbc_e[(u, v)] = ebc_subset.get((u, v), ebc_subset.get((v, u), 0))

    # 2. LLBMEe1: mapping entropy over RAW LLBCe of the neighbor edges of e1
    llbme_e1 = {}
    for u, v in edges:
        neigh_edges = []
        for n in G.neighbors(u):
            if n != v:
                neigh_edges.append((min(u, n), max(u, n)))
        for n in G.neighbors(v):
            if n != u:
                neigh_edges.append((min(v, n), max(v, n)))

        sum_log = sum(
            np.log(max(llbc_e.get(e, 1e-10), 1e-10)) for e in neigh_edges
        )
        val = llbc_e.get((u, v), 0)
        llbme_e1[(u, v)] = (-val * sum_log) if val > 0 else 0

    return create_metric_df(G, llbme_e1, "LLBMEe1")

if __name__ == "__main__":

    os.makedirs('results', exist_ok=True)
    G = nx.karate_club_graph()
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    out_name = "results/result_me.png"
    run_and_plot(G, "Improved LLBMe", rank_me_improved, out_name, reverse=False)
