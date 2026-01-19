import networkx as nx
from math import log
from network_utils import run_and_plot


def rank_me(G):
    """
    Mapping Entropy (ME) / Link-Local Mapping Betweenness Entropy (LLBME).
    1. Calculate Link Local Betweenness (LLBC) for all edges.
    2. Calculate Entropy based on neighbors' LLBC.
    """
    # 1. Calculate LLBC
    llbc_scores = {}
    for u, v in G.edges():
        # Subgraph of u, v and their neighbors
        nodes = set(G.neighbors(u)).union(set(G.neighbors(v))).union({u, v})
        subg = G.subgraph(nodes)
        try:
            ebc = nx.edge_betweenness_centrality(subg, normalized=False)
            key = (u, v) if (u, v) in ebc else (v, u)
            llbc_scores[(u, v)] = ebc.get(key, 0)
        except:
            llbc_scores[(u, v)] = 0

    # 2. Calculate ME
    me_scores = {}
    for u, v in G.edges():
        e_val = llbc_scores.get((u, v), 0)
        if e_val <= 0:
            me_scores[(u, v)] = 0
            continue

        neighbor_edges = list(G.edges(u)) + list(G.edges(v))
        sum_log = 0
        for nu, nv in neighbor_edges:
            if (nu, nv) == (u, v) or (nu, nv) == (v, u):
                continue

            n_key = (nu, nv) if (nu, nv) in llbc_scores else (nv, nu)
            n_val = llbc_scores.get(n_key, 0)

            if n_val > 0:
                sum_log += log(n_val)

        me_scores[(u, v)] = -e_val * sum_log

    return me_scores


if __name__ == "__main__":
    G = nx.karate_club_graph()
    run_and_plot(G, "Mapping Entropy (ME)", rank_me, "result_me.png")