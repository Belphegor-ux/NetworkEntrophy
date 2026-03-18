import networkx as nx
import sys
import os

# Add parent directory to path to import static ranking functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network_utils_iter import run_iterative_benchmark
from IE_informative_entropy import rank_ie

if __name__ == "__main__":
    G = nx.karate_club_graph()
    run_iterative_benchmark(G, "Information Entropy (IE)", rank_ie, "result_ie_iter.png")
