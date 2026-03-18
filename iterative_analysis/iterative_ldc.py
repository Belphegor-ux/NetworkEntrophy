import networkx as nx
import sys
import os

# Add parent directory to path to import static ranking functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network_utils_iter import run_iterative_benchmark
from LDC_link_degree_centrality import rank_cdc

if __name__ == "__main__":
    G = nx.karate_club_graph()
    # Note: rank_cdc is imported and returns a DataFrame
    run_iterative_benchmark(G, "Link Degree Centrality (LDC)", rank_cdc, "result_ldc_iter.png")
