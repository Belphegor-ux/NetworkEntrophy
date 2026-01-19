#networkEntropy
Network Dismantling Analysis Project

Overview

This project performs a comparative analysis of various edge-ranking algorithms for network dismantling (targeted attacks). The goal is to identify which metrics are most effective at fragmenting a network into disconnected components. The analysis focuses on the Static Decomposition strategy using the Zachary's Karate Club benchmark network.

Files Description

Core Files

network_utils.py: A shared utility library containing the NetworkDismantler class and the run_and_plot function. This file handles the core logic for:

Reading the graph.

Simulating edge removal based on provided rankings.

Calculating the Relative Giant Component Size ($R_{gc}$).

Plotting the decay curves and calculating the Area Under the Curve (AUC).

Note: This file is not meant to be run directly.

Algorithm Scripts

Each of the following scripts implements a specific edge-ranking metric. Run these individual files to generate the corresponding analysis and plots.

run_ei.py: Implements Explosive Immunization (EI).

Metric: Estimates edge importance based on the "explosive" potential of connected hubs (Degree * Sum of Neighbors' Degrees).

Sorting: Descending (Higher score = Remove first).

run_ci.py: Implements Collective Influence (CI).

Metric: Measures the influence of an edge based on the collective influence of its endpoints within a specific radius ($l=3$).

Sorting: Descending (Higher score = Remove first).

run_cdc.py: Implements Link Degree Centrality (CDC/LDC).

Metric: Product of the degrees of the two endpoints of an edge.

Sorting: Descending (Higher score = Remove first).

run_cks.py: Implements Link K-Shell (CKS/LKS).

Metric: Product of the K-Shell (Core) numbers of the two endpoints.

Sorting: Descending (Higher score = Remove first).

run_ie.py: Implements Information Entropy (IE).

Metric: Calculates entropy based on the degree distribution within the edge's neighborhood.

Sorting: Descending (Higher score = Remove first).

run_me.py: Implements Mapping Entropy (ME) (specifically Link-Local Mapping Betweenness Entropy).

Metric: Combines Link Local Betweenness Centrality (LLBC) with the entropy of neighboring edges' LLBC.

Sorting: Descending (Higher score = Remove first).

run_epc.py: Implements Edge Betweenness Centrality (EPC).

Metric: Standard global Edge Betweenness Centrality (number of shortest paths passing through an edge).

Sorting: Descending (Higher score = Remove first).

run_jaccard.py: Implements Jaccard Index.

Metric: Measures neighborhood similarity (Intersection / Union of neighbors).

Sorting: Ascending (Lower score = Remove first). Low similarity implies the edge is a "bridge" between communities.

Reports

network_decomposition_report.md: A detailed report explaining the theoretical background, methodology (Static vs. Iterative), metrics (Degree Product), and expected results of the analysis.

How to Run

Ensure you have Python installed with the necessary libraries:

pip install networkx numpy matplotlib seaborn


Make sure all .py files are in the same directory.

Run the specific script for the algorithm you want to test. For example:

python run_jaccard.py


The script will generate a PNG plot (e.g., result_jaccard.png) in the same directory and print the AUC (Area Under the Curve) score to the console.

Requirements

Python 3.x

networkx

numpy

matplotlib

seaborn
