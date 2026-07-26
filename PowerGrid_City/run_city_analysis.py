import os
import networkx as nx
import pandas as pd
import numpy as np
import sys

# Add src/utils to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/utils')))
from network_utils import run_and_plot

def rank_ldc(G: nx.Graph) -> pd.DataFrame:
    data = []
    for u, v in G.edges():
        data.append({'i': u, 'j': v, 'LDC': G.degree(u) * G.degree(v)})
    return pd.DataFrame(data)

def rank_jaccard(G: nx.Graph) -> pd.DataFrame:
    data = []
    for u, v in G.edges():
        u_nb = set(G.neighbors(u))
        v_nb = set(G.neighbors(v))
        inter = len(u_nb.intersection(v_nb))
        union = len(u_nb.union(v_nb))
        data.append({'i': u, 'j': v, 'Jaccard': inter / union if union > 0 else 0})
    return pd.DataFrame(data)

def rank_lks(G: nx.Graph) -> pd.DataFrame:
    """Link K-Shell: product of node core numbers for each edge."""
    core_nums = nx.core_number(G)
    data = []
    for u, v in G.edges():
        data.append({'i': u, 'j': v, 'LKS': core_nums[u] * core_nums[v]})
    return pd.DataFrame(data)

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

def rank_llbc_me(G: nx.Graph) -> pd.DataFrame:
    """
    LLBCe and LLBMEe1 (MDPI Entropy 26(4) 315, Eq. 8 + 11).

    LLBCe(e=(u,v)) uses nx.edge_betweenness_centrality_subset over the
    first-order central domain fc = {u,v} ∪ N(u) ∪ N(v) as both sources
    and targets, then normalizes by |fc| * (|fc| - 1) per Eq. 8.

    LLBMEe1(e1) is a mapping entropy on the RAW (un-normalized) LLBCe
    betweenness values:

        LLBMEe1(e1) = -LLBCe_raw(e1) * sum_{e2 in N(e1)} log(LLBCe_raw(e2))

    where N(e1) is the set of edges incident to the endpoints of e1
    EXCLUDING e1 itself, log uses np.log(max(val, 1e-10)), and the
    score is 0 when LLBCe_raw(e1) <= 0. This yields NEGATIVE values:
    the smallest is the most critical edge.
    """
    edges = [tuple(sorted(e)) for e in G.edges()]
    node_to_neighbors = {n: set(G.neighbors(n)) for n in G.nodes()}

    edge_to_fc: dict = {}
    for u, v in edges:
        fc = {u, v} | node_to_neighbors[u] | node_to_neighbors[v]
        edge_to_fc[(u, v)] = list(fc)

    def get_score(e):
        u, v = e
        fc_nodes = edge_to_fc[e]
        ebc_subset = nx.edge_betweenness_centrality_subset(
            G, sources=fc_nodes, targets=fc_nodes, normalized=False
        )
        raw = ebc_subset.get((u, v), ebc_subset.get((v, u), 0))
        denom = len(fc_nodes) * (len(fc_nodes) - 1)
        normalized = raw / denom if denom > 0 else 0.0
        return e, raw, normalized

    llbc_raw: dict = {}
    llbc_e: dict = {}
    for e in edges:
        _, raw, normalized = get_score(e)
        llbc_raw[e] = raw
        llbc_e[e] = normalized

    # LLBMEe1: mapping entropy on RAW betweenness (reference formula).
    llbme_e1: dict = {}
    for u, v in edges:
        e1 = (u, v)
        neigh_edges = []
        for n in G.neighbors(u):
            if n != v:
                neigh_edges.append((min(u, n), max(u, n)))
        for n in G.neighbors(v):
            if n != u:
                neigh_edges.append((min(v, n), max(v, n)))

        sum_log = sum(
            np.log(max(llbc_raw.get(e2, 1e-10), 1e-10))
            for e2 in neigh_edges
        )
        val = llbc_raw.get(e1, 0)
        llbme_e1[e1] = (-val * sum_log) if val > 0 else 0

    data = []
    for u, v in edges:
        e_can = (u, v)
        data.append({
            'i': u,
            'j': v,
            'LLBCe': llbc_e[e_can],
            'LLBMEe1': llbme_e1[e_can],
        })

    return pd.DataFrame(data)

def main() -> None:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "tokyo_grid.gml")
    if not os.path.exists(gml_path):
        print(f"Error: {gml_path} not found. Run download script first.")
        return

    G = nx.read_gml(gml_path)
    G = nx.convert_node_labels_to_integers(G)
    print(f"Loaded Tokyo Grid: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # Create results directory
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_city")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Compute and combine all metrics
    print("Computing metrics...")
    df_ldc = rank_ldc(G)
    df_jaccard = rank_jaccard(G)
    df_lks = rank_lks(G)
    df_ci = rank_ci(G)
    df_llbc = rank_llbc_me(G)

    # Merge all
    df_combined = df_ldc.merge(df_jaccard, on=['i', 'j'])
    df_combined = df_combined.merge(df_lks, on=['i', 'j'])
    df_combined = df_combined.merge(df_ci, on=['i', 'j'])
    df_combined = df_combined.merge(df_llbc, on=['i', 'j'])

    # Save to CSV
    csv_out = os.path.join(RESULTS_DIR, "tokyo_grid_metrics.csv")
    df_combined.to_csv(csv_out, index=False)
    print(f"Saved combined metrics to {csv_out}")

    # 2. Run dismantling analysis and plots
    print("Running dismantling analysis...")
    run_and_plot(G, "LDC", rank_ldc, os.path.join(RESULTS_DIR, "plot_ldc.png"))
    run_and_plot(G, "Jaccard", rank_jaccard, os.path.join(RESULTS_DIR, "plot_jaccard.png"), reverse=False)
    run_and_plot(G, "LKS", rank_lks, os.path.join(RESULTS_DIR, "plot_lks.png"))
    run_and_plot(G, "Collective Influence", rank_ci, os.path.join(RESULTS_DIR, "plot_ci.png"))
    run_and_plot(G, "LLBCe and LLBMEe1", rank_llbc_me, os.path.join(RESULTS_DIR, "plot_llbc.png"), reverse={"LLBCe": True, "LLBMEe1": False})

    print("All analyses completed. Check 'results_city' directory.")

if __name__ == "__main__":
    main()
