#!/bin/bash

echo "Spawning parallel agents for Football benchmarks..."

# Start background agents for each method
gemini-flow agent spawn worker --objective "cd Football/CKS && python CKS_Link_K_Shell.py && python iterative_cks.py" --gemini &
gemini-flow agent spawn worker --objective "cd Football/CI && python collective_influence.py && python iterative_ci.py" --gemini &
# DEPRECATED per new_instructions.md §3
# gemini-flow agent spawn worker --objective "cd Football/IE && python IE_informative_entropy.py && python iterative_ie.py" --gemini &
gemini-flow agent spawn worker --objective "cd Football/Jaccard && python jaccard_index.py && python iterative_jaccard.py" --gemini &
gemini-flow agent spawn worker --objective "cd Football/LDC && python LDC_link_degree_centrality.py && python iterative_ldc.py" --gemini &
gemini-flow agent spawn worker --objective "cd Football/LLBC && python LLBC_contrast.py && python iterative_llbc_me.py" --gemini &
gemini-flow agent spawn worker --objective "cd Football/ME && python ME_mapping_entropy.py && python iterative_me.py" --gemini &

echo "Waiting for all Football agents to finish..."
wait
echo "Football benchmarks completed! Results are in Football/[Method]/results/"
