import networkx as nx
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
from network_utils import run_and_plot, create_metric_df

def rank_cdc(G):
    """
    Link Degree Centrality (LDC).
    Score = Degree(u) * Degree(v)
    """
    scores = {}
    for u, v in G.edges():
        scores[(u, v)] = G.degree(u) * G.degree(v)
    return create_metric_df(G, scores, "LDC")

if __name__ == "__main__":

    os.makedirs('results', exist_ok=True)
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/football/football.gml'))
    G = nx.read_gml(dataset_path, label='id')
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    out_name = "results/result_ldc.png"
    run_and_plot(G, "Link Degree Centrality (CDC)", rank_cdc, out_name)
