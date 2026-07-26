import os
import networkx as nx
import pandas as pd
import numpy as np
import sys
import argparse

# Add current and parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../src/utils')))

from run_city_analysis import rank_ldc, rank_jaccard, rank_lks, rank_ci, rank_llbc_me
from network_utils_iter import run_iterative_benchmark

def main():
    parser = argparse.ArgumentParser(description="Run iterative analysis for a specific methodology.")
    parser.add_argument("--method", choices=['ldc', 'jaccard', 'lks', 'ci', 'llbc'], required=True, help="Methodology to run.")
    args = parser.parse_args()

    gml_path = os.path.join(current_dir, "datasets", "tokyo_grid.gml")
    if not os.path.exists(gml_path):
        print(f"Error: {gml_path} not found.")
        return
    
    G = nx.read_gml(gml_path)
    G = nx.convert_node_labels_to_integers(G)
    
    os.makedirs("results_city", exist_ok=True)
    
    # Iterative metric values are not comparable across removal steps, so the
    # meaningful record is the REMOVAL ORDER per edge (1-based), written here.
    if args.method == 'ldc':
        run_iterative_benchmark(G, "LDC", rank_ldc, "results_city/plot_iterative_ldc.png",
                                order_csv_path="results_city/tokyo_iter_order_ldc.csv")
    elif args.method == 'jaccard':
        run_iterative_benchmark(G, "Jaccard", rank_jaccard, "results_city/plot_iterative_jaccard.png",
                                reverse=False, order_csv_path="results_city/tokyo_iter_order_jaccard.csv")
    elif args.method == 'lks':
        run_iterative_benchmark(G, "LKS", rank_lks, "results_city/plot_iterative_lks.png",
                                order_csv_path="results_city/tokyo_iter_order_lks.csv")
    elif args.method == 'ci':
        run_iterative_benchmark(G, "Collective Influence", rank_ci, "results_city/plot_iterative_ci.png",
                                order_csv_path="results_city/tokyo_iter_order_ci.csv")
    elif args.method == 'llbc':
        run_iterative_benchmark(G, "LLBCe and LLBMEe1", rank_llbc_me, "results_city/plot_iterative_llbc.png",
                                reverse={"LLBCe": True, "LLBMEe1": False},
                                order_csv_path="results_city/tokyo_iter_order_llbc.csv")

if __name__ == "__main__":
    main()
