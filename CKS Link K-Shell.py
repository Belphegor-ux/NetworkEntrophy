import networkx as nx
from network_utils import run_and_plot

def rank_cks(G):
    """
    Link K-Shell Index (LKS).
    Score = Core(u) * Core(v)
    """
    core_numbers = nx.core_number(G)
    scores = {}
    for u, v in G.edges():
        scores[(u, v)] = core_numbers[u] * core_numbers[v]
    return scores

if __name__ == "__main__":
    G = nx.karate_club_graph()
    run_and_plot(G, "Link K-Shell (CKS)", rank_cks, "result_cks.png")