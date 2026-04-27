import os
import urllib.request
import pandas as pd
import networkx as nx

def download_file(url, target_path):
    print(f"Downloading {url}...")
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as response, open(target_path, 'wb') as out_file:
        out_file.write(response.read())
    print(f"Saved to {target_path}")

def build_city_grid():
    # Japan Power Grid (High-Voltage) from ComplexNetTSP/Power_grids
    nodes_url = "https://raw.githubusercontent.com/ComplexNetTSP/Power_grids/master/Countries/Japan/Nodes/highvoltage_vertices.csv"
    edges_url = "https://raw.githubusercontent.com/ComplexNetTSP/Power_grids/master/Countries/Japan/Edges/highvoltage_links.csv"
    
    datasets_dir = os.path.join("PowerGrid_City", "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    nodes_path = os.path.join(datasets_dir, "japan_nodes.csv")
    edges_path = os.path.join(datasets_dir, "japan_edges.csv")
    
    if not os.path.exists(nodes_path):
        download_file(nodes_url, nodes_path)
    if not os.path.exists(edges_path):
        download_file(edges_url, edges_path)
        
    # Load nodes
    # Format: v_id#lon#lat#...
    nodes_df = pd.read_csv(nodes_path, sep='#')
    # Filter for Eastern Japan (Kanto Region / Tokyo)
    # Bounding Box: lon [138.3, 141.0], lat [34.8, 37.1]
    tokyo_nodes = nodes_df[
        (nodes_df['lon'] >= 138.3) & (nodes_df['lon'] <= 141.0) &
        (nodes_df['lat'] >= 34.8) & (nodes_df['lat'] <= 37.1)
    ]
    
    tokyo_node_ids = set(tokyo_nodes['v_id'].values)
    print(f"Filtered {len(tokyo_node_ids)} nodes in Tokyo/Eastern Japan region.")
    
    # Load edges
    # Format: l_id#v_id_1#v_id_2#...
    edges_df = pd.read_csv(edges_path, sep='#')
    
    # Filter edges where both endpoints are in our set
    tokyo_edges = edges_df[
        edges_df['v_id_1'].isin(tokyo_node_ids) & edges_df['v_id_2'].isin(tokyo_node_ids)
    ]
    print(f"Filtered {len(tokyo_edges)} edges in Tokyo/Eastern Japan region.")
    
    # Construct Graph
    G = nx.Graph()
    for _, row in tokyo_edges.iterrows():
        u = int(row['v_id_1'])
        v = int(row['v_id_2'])
        if u != v: # Avoid self-loops
            G.add_edge(u, v)
        
    # Get Largest Connected Component
    if len(G) > 0:
        lcc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc_nodes).copy()
        print(f"Largest Connected Component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        
        # Save as GML
        nx.write_gml(G, os.path.join(datasets_dir, "tokyo_grid.gml"))
        print("Saved tokyo_grid.gml")
    else:
        print("No graph could be constructed with the filtered criteria.")

if __name__ == "__main__":
    build_city_grid()
