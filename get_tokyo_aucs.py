import os
import networkx as nx
import pandas as pd
import numpy as np
import sys

# Add src/utils to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/utils')))
from network_utils import NetworkDismantler

def get_auc(rgc):
    x = np.linspace(0, 1, len(rgc))
    if hasattr(np, "trapezoid"):
        return np.trapezoid(rgc, x)
    return np.trapz(rgc, x)

def calculate_aucs():
    gml_path = "PowerGrid_City/datasets/tokyo_grid.gml"
    csv_path = "results_city/tokyo_grid_metrics.csv"
    
    if not os.path.exists(gml_path) or not os.path.exists(csv_path):
        print("Missing files.")
        return

    G = nx.read_gml(gml_path)
    G = nx.convert_node_labels_to_integers(G)
    df = pd.read_csv(csv_path)
    dismantler = NetworkDismantler(G)
    
    metrics = [col for col in df.columns if col not in ['i', 'j']]
    results = []
    
    for m in metrics:
        # Static AUC
        reverse = False if m == 'Jaccard' else True
        rgc_static = dismantler.get_static_curve(df, m, reverse=reverse)
        auc_static = get_auc(rgc_static)
        results.append({'Metric': m, 'AUC_Static': auc_static})
        
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    calculate_aucs()
