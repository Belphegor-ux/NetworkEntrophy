import networkx as nx
from network_utils import run_and_plot

def rank_ci(G, l=3):
    """
    Collective Influence (Edge Version).
    CI_l(u) = (k_u - 1) * sum(k_v - 1 for v in Ball(u, l))
    """
    ci_nodes = {}
    for node in G.nodes():
        k_u = G.degree(node)
        if k_u <= 1:
            ci_nodes[node] = 0
            continue

        # BFS to find boundary at level l
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=l)
        boundary_nodes = [n for n, dist in lengths.items() if dist == l]

        sum_k = sum((G.degree(v) - 1) for v in boundary_nodes)
        ci_nodes[node] = (k_u - 1) * sum_k

    scores = {}
    for u, v in G.edges():
        scores[(u, v)] = ci_nodes[u] + ci_nodes[v]
    return scores

if __name__ == "__main__":
    G = nx.karate_club_graph()
    run_and_plot(G, "Collective Influence (CI)", rank_ci, "result_ci.png")