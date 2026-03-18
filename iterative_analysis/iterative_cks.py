import networkx as nx
import sys
import os

# Add parent directory to path to import static ranking functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network_utils_iter import run_iterative_benchmark
from CKS_Link_K_Shell import rank_cks

if __name__ == "__main__":
    G = nx.karate_club_graph()
    run_iterative_benchmark(G, "Link K-Shell (CKS)", rank_cks, "result_cks_iter.png")
