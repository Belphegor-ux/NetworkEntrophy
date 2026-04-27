import networkx as nx
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
import pandas as pd
from network_utils import run_and_plot

def rank_ci(G: nx.Graph, l: int = 3) -> pd.DataFrame:
    """
    Collective Influence — emits 4 edge-level variants:
        CI_e_av_skin, CI_e_mul_skin, CI_e_av_body, CI_e_mul_body
    Per the spec (new_instructions.md §2):
      - skin: nodes exactly at distance l (Ball boundary)
      - body: all nodes within 0 < d <= l (Ball volume)
    Per-node CI(u) = (k_u - 1) * sum_{v in set} (k_v - 1).
    Per-edge: average and multiplication of the two endpoint scores.
    """
    ci_skin_nodes: dict = {}
    ci_body_nodes: dict = {}
    for node in G.nodes():
        k_u = G.degree(node)
        if k_u <= 1:
            ci_skin_nodes[node] = 0
            ci_body_nodes[node] = 0
            continue
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=l)
        skin_nodes = [n for n, dist in lengths.items() if dist == l]
        sum_k_skin = sum((G.degree(v) - 1) for v in skin_nodes)
        ci_skin_nodes[node] = (k_u - 1) * sum_k_skin
        body_nodes = [n for n, dist in lengths.items() if 0 < dist <= l]
        sum_k_body = sum((G.degree(v) - 1) for v in body_nodes)
        ci_body_nodes[node] = (k_u - 1) * sum_k_body
    data = []
    for u, v in G.edges():
        s_u, s_v = ci_skin_nodes[u], ci_skin_nodes[v]
        b_u, b_v = ci_body_nodes[u], ci_body_nodes[v]
        data.append({
            'i': u, 'j': v,
            'CI_e_av_skin': (s_u + s_v) / 2,
            'CI_e_mul_skin': s_u * s_v,
            'CI_e_av_body': (b_u + b_v) / 2,
            'CI_e_mul_body': b_u * b_v,
        })
    return pd.DataFrame(data)

if __name__ == "__main__":

    os.makedirs('results', exist_ok=True)
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/football/football.gml'))
    G = nx.read_gml(dataset_path, label='id')
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    out_name = "results/result_ci.png"
    run_and_plot(G, "Collective Influence (CI)", rank_ci, out_name)
