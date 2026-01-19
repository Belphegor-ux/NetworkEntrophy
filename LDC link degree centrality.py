import networkx as nx
from network_utils import run_and_plot

def rank_cdc(G):
    """
    Link Degree Centrality (LDC).
    Score = Degree(u) * Degree(v)
    """
    scores = {}
    for u, v in G.edges():
        scores[(u, v)] = G.degree(u) * G.degree(v)
    return scores

if __name__ == "__main__":
    G = nx.karate_club_graph()
    run_and_plot(G, "Link Degree Centrality (CDC)", rank_cdc, "result_cdc.png")