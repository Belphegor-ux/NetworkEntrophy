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
from ME_mapping_entropy import rank_me_improved

if __name__ == "__main__":

    os.makedirs('results', exist_ok=True)
    G = nx.karate_club_graph()
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    out_name = "results/result_me_iter.png"
    run_iterative_benchmark(G, "Improved Mapping Entropy", rank_me_improved, out_name)
