import networkx as nx
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
from math import log
from network_utils import run_and_plot, create_metric_df


def rank_ie(G):
    """
    Information Entropy (IE) for Edges.
    H_edge = - sum ( p_i log p_i ) based on neighborhood degree distribution.
    """
    scores = {}
    for u, v in G.edges():
        # Neighborhood: neighbors of u union neighbors of v (excluding u, v)
        u_neighbors = set(G.neighbors(u))
        v_neighbors = set(G.neighbors(v))
        neighborhood = u_neighbors.union(v_neighbors)
        if u in neighborhood: neighborhood.remove(u)
        if v in neighborhood: neighborhood.remove(v)

        if not neighborhood:
            scores[(u, v)] = 0
            continue

        total_deg = sum(G.degree(n) for n in neighborhood)
        if total_deg == 0:
            scores[(u, v)] = 0
            continue

        entropy = 0
        for n in neighborhood:
            p = G.degree(n) / total_deg
            if p > 0:
                entropy += -p * log(p)

        scores[(u, v)] = entropy
    return create_metric_df(G, scores, "IE")


if __name__ == "__main__":

    os.makedirs('results', exist_ok=True)
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/jazz/arenas-jazz/out.arenas-jazz'))
    G = nx.read_edgelist(dataset_path, comments='%')
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    out_name = "results/result_ie.png"
    run_and_plot(G, "Information Entropy (IE)", rank_ie, out_name)
