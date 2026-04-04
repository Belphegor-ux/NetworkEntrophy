import networkx as nx
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import networkx as nx
import sys
import os

# Add both .. and ../static to the system path

from network_utils_iter import run_iterative_benchmark
from CKS_Link_K_Shell import rank_cks

if __name__ == "__main__":

    os.makedirs('results', exist_ok=True)
    G = nx.karate_club_graph()
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    out_name = "results/result_cks_iter.png"
    run_iterative_benchmark(G, "Link K-Shell (CKS)", rank_cks, out_name)
