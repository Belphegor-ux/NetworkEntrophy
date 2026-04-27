"""
DEPRECATED — per new_instructions.md §3, the EI/IE method has been dropped
from the active spec due to ambiguous original definitions. This file is
retained for historical reference and dashboard backward-compatibility only.
Do not extend or wire into new pipelines.
"""
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
from IE_informative_entropy import rank_ie

if __name__ == "__main__":

    os.makedirs('results', exist_ok=True)
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/jazz/arenas-jazz/out.arenas-jazz'))
    G = nx.read_edgelist(dataset_path, comments='%')
    G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    out_name = "results/result_ie_iter.png"
    run_iterative_benchmark(G, "Information Entropy (IE)", rank_ie, out_name)
