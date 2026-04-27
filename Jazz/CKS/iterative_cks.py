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
from CKS_Link_K_Shell import rank_lks

if __name__ == "__main__":

    os.makedirs('results', exist_ok=True)
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/jazz/arenas-jazz/out.arenas-jazz'))
    G = nx.read_edgelist(dataset_path, comments='%')
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    out_name = "results/result_cks_iter.png"
    run_iterative_benchmark(G, "Link K-Shell (LKS)", rank_lks, out_name)
