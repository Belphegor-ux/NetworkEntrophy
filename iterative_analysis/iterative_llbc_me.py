import networkx as nx
import sys
import os

# Add parent directory to path to import static ranking functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from network_utils_iter import run_iterative_benchmark
from LLBC_contrast import rank_llbc_me

if __name__ == "__main__":
    G = nx.karate_club_graph()
    run_iterative_benchmark(G, "LLBC and LLBME", rank_llbc_me, "result_llbc_me_iter.png")
