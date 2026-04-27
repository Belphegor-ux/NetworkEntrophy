import networkx as nx
import sys
import os
import json
import numpy as np
import pandas as pd
from math import log
import concurrent.futures
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend for multiprocessing safety

# Ensure src/utils is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/utils')))
from network_utils import run_and_plot, create_metric_df
from network_utils_iter import run_iterative_benchmark

# --- Entropy Methods ---

def rank_ldc(G):
    scores = {}
    for u, v in G.edges():
        scores[(u, v)] = G.degree(u) * G.degree(v)
    return create_metric_df(G, scores, "LDC")

def rank_jaccard(G):
    scores = {}
    for u, v in G.edges():
        u_neighbors = set(G.neighbors(u))
        v_neighbors = set(G.neighbors(v))
        intersection = len(u_neighbors.intersection(v_neighbors))
        union = len(u_neighbors.union(v_neighbors))
        scores[(u, v)] = intersection / union if union > 0 else 0
    return create_metric_df(G, scores, "Jaccard")

def rank_lks(G):
    core_numbers = nx.core_number(G)
    scores = {}
    for u, v in G.edges():
        scores[(u, v)] = core_numbers[u] * core_numbers[v]
    return create_metric_df(G, scores, "LKS")

def rank_ie(G):
    scores = {}
    for u, v in G.edges():
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
            entropy = sum(- (val/sum_llbc) * np.log(val/sum_llbc) for val in local_llbc_vals if val > 0)
            llbme_e1[e1] = entropy
    
    data = []
    for u, v in edges:
        e_can = (u, v)
        data.append({'i': u, 'j': v, 'LLBCe': llbc_e[e_can], 'LLBMEe1': llbme_e1[e_can]})
    return pd.DataFrame(data)

def rank_ci(G, l=3):
    """
    Collective Influence — emits 4 edge-level variants:
        CI_e_av_skin, CI_e_mul_skin, CI_e_av_body, CI_e_mul_body
    skin: nodes exactly at distance l. body: nodes within 0 < d <= l.
    Per-node CI(u) = (k_u - 1) * sum_{v in set} (k_v - 1).
    Per-edge: average and multiplication of endpoint scores.
    Legacy CI/CI_e_mul_skin readers degrade gracefully via _ci_columns_present().
    """
    ci_skin_nodes = {}
    ci_body_nodes = {}
    for node in G.nodes():
        k_u = G.degree(node)
        if k_u <= 1:
            ci_skin_nodes[node] = ci_body_nodes[node] = 0
            continue
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=l)
        sum_k_skin = sum((G.degree(v) - 1) for v, dist in lengths.items() if dist == l)
        ci_skin_nodes[node] = (k_u - 1) * sum_k_skin
        sum_k_body = sum((G.degree(v) - 1) for v, dist in lengths.items() if 0 < dist <= l)
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


# Backward-compat helper: list of CI variant column names that may appear in a
# metrics DataFrame loaded from older CSVs. Use to read whichever subset exists.
CI_VARIANT_COLUMNS = (
    "CI_e_av_skin",
    "CI_e_mul_skin",
    "CI_e_av_body",
    "CI_e_mul_body",
)


def _ci_columns_present(df) -> list:
    """Return the CI variant columns that actually exist in df.

    Falls back to legacy single-variant ('CI') if no new-style columns are
    present, allowing older cached metrics CSVs to keep working.
    """
    present = [c for c in CI_VARIANT_COLUMNS if c in df.columns]
    if present:
        return present
    if "CI" in df.columns:
        return ["CI"]
    return []

# --- Tasks & Multiprocessing ---

def run_metric_task(ds_name, G, m_name, m_func, reverse, is_iterative):
    try:
        print(f"[{'Iterative' if is_iterative else 'Static'}] Starting {ds_name} - {m_name}...")
        # LLBC/LLBME are O(E^2 * subsets)
        edge_limit = 3000 if m_name in ["LLBC", "LLBME"] else 5000
        if is_iterative and G.number_of_edges() > edge_limit:
            print(f"Skipping {ds_name} {m_name} Iterative (Size: {G.number_of_edges()} edges > {edge_limit})")
            return ds_name, "Iterative", {}

        res_dir = f"results_{ds_name.lower().replace(' ', '_')}"
        os.makedirs(res_dir, exist_ok=True)
        mode_suffix = "iter" if is_iterative else "static"
        plot_file = os.path.join(res_dir, f"result_{m_name.lower()}_{mode_suffix}.png")

        if is_iterative:
            # Dynamically set step_size to finish ASAP (recalculate ~50 times total)
            step_size = max(1, G.number_of_edges() // 50)
            res = run_iterative_benchmark(G, f"{ds_name} - {m_name}", m_func, plot_file, reverse=reverse, step_size=step_size)
        else:
            res = run_and_plot(G, f"{ds_name} - {m_name}", m_func, plot_file, reverse=reverse)
            
        print(f"[{'Iterative' if is_iterative else 'Static'}] Finished {ds_name} - {m_name}")
        return ds_name, "Iterative" if is_iterative else "Static", res
    except Exception as e:
        print(f"Error evaluating {m_name} on {ds_name} ({'Iterative' if is_iterative else 'Static'}): {e}")
        return ds_name, "Iterative" if is_iterative else "Static", {}

def rank_llbc(G):
    df = rank_llbc_me(G)
    return df[['i', 'j', 'LLBCe']].rename(columns={'LLBCe': 'LLBC'})

def rank_llbme(G):
    df = rank_llbc_me(G)
    return df[['i', 'j', 'LLBMEe1']].rename(columns={'LLBMEe1': 'LLBME'})

def main():
    datasets = {}
    
    # 1. Karate Club
    datasets["Karate"] = nx.karate_club_graph()
    
    # 2. Football
    football_path = "datasets/football/football.gml"
    if os.path.exists(football_path):
        datasets["Football"] = nx.read_gml(football_path)
    
    # 3. Jazz
    jazz_path = "datasets/jazz/arenas-jazz/out.arenas-jazz"
    if os.path.exists(jazz_path):
        G_jazz = nx.Graph()
        with open(jazz_path, "r") as f:
            for line in f:
                if line.startswith("%") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    G_jazz.add_edge(u, v)
        datasets["Jazz"] = G_jazz

    # 4. Tokyo City
    tokyo_path = "PowerGrid_City/datasets/tokyo_grid.gml"
    if os.path.exists(tokyo_path):
        datasets["Tokyo City"] = nx.read_gml(tokyo_path)

    methods = [
        ("LDC", rank_ldc, True),
        ("Jaccard", rank_jaccard, False),
        ("LKS", rank_lks, True),
        # DEPRECATED per new_instructions.md §3 — not registered for runs
        # ("IE", rank_ie, True),
        ("LLBC", rank_llbc, True),
        ("LLBME", rank_llbme, True),
        ("CI", rank_ci, True)
    ]
    
    tasks = []
    for ds_name, G in datasets.items():
        G = nx.convert_node_labels_to_integers(G)
        
        # Prepare params to multiprocessing (static & iterative)
        for m_name, m_func, reverse in methods:
            tasks.append((ds_name, G, m_name, m_func, reverse, False)) # Static
            tasks.append((ds_name, G, m_name, m_func, reverse, True))  # Iterative

    all_results = {ds: {"Static": {}, "Iterative": {}} for ds in datasets.keys()}
    
    print(f"Submitting {len(tasks)} tasks for parallel processing...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_metric_task, *args) for args in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            ds_name, mode, res = future.result()
            all_results[ds_name][mode].update(res)

    print("Saving aggregated data to dashboard_app/...")
    with open("dashboard_app/dashboard_data.js", "w") as f:
        f.write("const chartData = ")
        json.dump(all_results, f)
        f.write(";")
    with open("dashboard_app/dashboard_data.json", "w") as f:
        json.dump(all_results, f)
    print("Aggregate data saved to dashboard_app/dashboard_data.js and dashboard_app/dashboard_data.json")

if __name__ == "__main__":
    main()
