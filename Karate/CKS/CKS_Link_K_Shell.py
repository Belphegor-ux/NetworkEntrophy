import networkx as nx
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
from network_utils import run_and_plot, create_metric_df

def rank_cks(G):
    """
    Link K-Shell Index (LKS).
    Score = Core(u) * Core(v)
    """
    core_numbers = nx.core_number(G)
    scores = {}
    for u, v in G.edges():
        scores[(u, v)] = core_numbers[u] * core_numbers[v]
    return create_metric_df(G, scores, "LKS")

if __name__ == "__main__":

    os.makedirs('results', exist_ok=True)
    G = nx.karate_club_graph()
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    out_name = "results/result_cks.png"
    run_and_plot(G, "Link K-Shell (CKS)", rank_cks, out_name)
