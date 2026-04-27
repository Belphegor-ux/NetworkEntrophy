import networkx as nx
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import networkx as nx
import sys
import os

# Add both .. and ../static to the system path

from network_utils_iter import run_iterative_benchmark
from LLBC_contrast import rank_llbc_me

if __name__ == "__main__":

    os.makedirs('results', exist_ok=True)
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/football/football.gml'))
    G = nx.read_gml(dataset_path, label='id')
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    out_name = "results/result_llbc_iter.png"
    run_iterative_benchmark(G, "LLBCe and LLBMEe1", rank_llbc_me, out_name)
