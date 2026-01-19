import networkx as nx
from math import log
from network_utils import run_and_plot


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
    return scores


if __name__ == "__main__":
    G = nx.karate_club_graph()
    run_and_plot(G, "Information Entropy (IE)", rank_ie, "result_ie.png")