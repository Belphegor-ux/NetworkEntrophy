import os
import glob
import re

def refactor_script(filepath, is_iterative=False):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the __main__ block
    if "if __name__ == '__main__':" in content:
        main_str = "if __name__ == '__main__':"
    elif 'if __name__ == "__main__":' in content:
        main_str = 'if __name__ == "__main__":'
    else:
        return # No main block

    parts = content.split(main_str)
    
    plot_call_pattern = re.compile(r'(run_(?:and_plot|iterative_benchmark)\s*\(\s*G[^,]*\s*,\s*("[^"]+"|\'[^\']+\'|.*?)\s*,\s*([a-zA-Z0-9_]+)\s*,\s*("[^"]+"|\'[^\']+\'|.*?)[^\)]*\))')
    
    match = plot_call_pattern.search(parts[1])
    if match:
        full_call = match.group(1)
        title = match.group(2)
        rank_func = match.group(3)
        orig_out_name = match.group(4).strip('"\'')
        
        # We replace the entire parts[1] with our new sys.argv handling
        if is_iterative:
            run_func = "run_iterative_benchmark"
        else:
            run_func = "run_and_plot"

        # Check for reverse parameter
        reverse_param = ""
        if "reverse=False" in full_call:
            reverse_param = ", reverse=False"
            
        new_main = f"""
    import sys
    import os
    if len(sys.argv) > 2:
        dataset_path = sys.argv[1]
        out_dir = sys.argv[2]
        
        if dataset_path.endswith('.gml'):
            G = nx.read_gml(dataset_path, label='id')
        else:
            G = nx.read_edgelist(dataset_path, comments='%')
            
        # Convert string labels to ints if possible for consistency
        G = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
        
        os.makedirs(out_dir, exist_ok=True)
        out_name = os.path.join(out_dir, "{orig_out_name}")
    else:
        G = nx.karate_club_graph()
        out_name = "{orig_out_name}"
        
    {run_func}(G, {title}, {rank_func}, out_name{reverse_param})
"""
        new_content = parts[0] + main_str + new_main
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Refactored {filepath}")
    else:
        print(f"Could not parse run_and_plot in {filepath}")

# Static scripts
static_scripts = [
    "CKS_Link_K_Shell.py",
    "collective_influence.py",
    "IE_informative_entropy.py",
    "jaccard_index.py",
    "LDC_link_degree_centrality.py",
    "LLBC_contrast.py",
    "ME_mapping_entropy.py"
]

for s in static_scripts:
    if os.path.exists(s):
        refactor_script(s, is_iterative=False)

# Iterative scripts
iterative_scripts = glob.glob("iterative_analysis/*.py")
for s in iterative_scripts:
    if not s.endswith("network_utils_iter.py"):
        refactor_script(s, is_iterative=True)
