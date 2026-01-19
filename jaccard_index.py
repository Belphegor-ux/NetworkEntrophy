import networkx as nx
from network_utils import run_and_plot


def rank_jaccard(G):
    """
    Jaccard Index (Jaccard).
    Score = |Neighbors(u) INTERSECT Neighbors(v)| / |Neighbors(u) UNION Neighbors(v)|

    Paper Definition: A smaller value indicates higher importance (bridge-like).
    Therefore, we will sort Ascending (reverse=False) for removal.
    """
    scores = {}
    for u, v in G.edges():
        u_neighbors = set(G.neighbors(u))
        v_neighbors = set(G.neighbors(v))

        # Calculate Intersection and Union of neighborhoods
        intersection = len(u_neighbors.intersection(v_neighbors))
        union = len(u_neighbors.union(v_neighbors))

        if union == 0:
            scores[(u, v)] = 0
        else:
            scores[(u, v)] = intersection / union

    return scores


if __name__ == "__main__":
    G = nx.karate_club_graph()
    # Note: reverse=False because Lower Score = Higher Importance for Jaccard
    run_and_plot(G, "Jaccard Index", rank_jaccard, "result_jaccard.png", reverse=False)