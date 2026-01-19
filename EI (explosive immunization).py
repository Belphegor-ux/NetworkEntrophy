import networkx as nx
from network_utils import run_and_plot

def rank_ei(G):
    """
    Explosive Immunization (Edge Version Approximation).
    Score ~ Degree(u) * Sum(Neighbors' Degrees)
    """
    scores = {}
    for u, v in G.edges():
        # Captures the "Explosive" potential (hubs connecting to hubs)
        score_u = G.degree(u) * sum(G.degree(n) for n in G.neighbors(u))
        score_v = G.degree(v) * sum(G.degree(n) for n in G.neighbors(v))
        scores[(u, v)] = score_u * score_v
    return scores

if __name__ == "__main__":
    G = nx.karate_club_graph()
    run_and_plot(G, "Explosive Immunization (EI)", rank_ei, "result_ei.png")