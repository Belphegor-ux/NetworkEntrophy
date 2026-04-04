import networkx as nx
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
import pandas as pd
from network_utils import run_and_plot

def rank_ci(G, l=3):
    """
    Collective Influence (Node version) calculated in two ways:
    1. CI_skin: nodes exactly at distance l.
    2. CI_body: nodes within distance l (inclusive).
    
    Then aggregated to edges in two ways:
    - Average (CI_e_av)
    - Multiplication (CI_e_mul)
    """
    ci_skin_nodes = {}
    ci_body_nodes = {}
    
    for node in G.nodes():
        k_u = G.degree(node)
        if k_u <= 1:
            ci_skin_nodes[node] = 0
            ci_body_nodes[node] = 0
            continue

        # Find nodes at different distances
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=l)
        
        # CI_skin: only nodes at dist == l
        skin_nodes = [n for n, dist in lengths.items() if dist == l]
        sum_k_skin = sum((G.degree(v) - 1) for v in skin_nodes)
        ci_skin_nodes[node] = (k_u - 1) * sum_k_skin
        
        # CI_body: all nodes in the ball (dist <= l), excluding the node itself (dist 0)
        body_nodes = [n for n, dist in lengths.items() if 0 < dist <= l]
        sum_k_body = sum((G.degree(v) - 1) for v in body_nodes)
        ci_body_nodes[node] = (k_u - 1) * sum_k_body

    data = []
    for u, v in G.edges():
        row = {'i': u, 'j': v}
        # Skin aggregation
        row['CI_e_av_skin'] = (ci_skin_nodes[u] + ci_skin_nodes[v]) / 2
        row['CI_e_mul_skin'] = ci_skin_nodes[u] * ci_skin_nodes[v]
        # Body aggregation
        row['CI_e_av_body'] = (ci_body_nodes[u] + ci_body_nodes[v]) / 2
        row['CI_e_mul_body'] = ci_body_nodes[u] * ci_body_nodes[v]
        data.append(row)
        
    return pd.DataFrame(data)

if __name__ == "__main__":

    os.makedirs('results', exist_ok=True)
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/jazz/arenas-jazz/out.arenas-jazz'))
    G = nx.read_edgelist(dataset_path, comments='%')
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    out_name = "results/result_ci.png"
    run_and_plot(G, "Collective Influence (CI)", rank_ci, out_name)
