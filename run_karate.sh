#!/bin/bash

echo "Spawning parallel agents for Karate benchmarks..."

# Start background agents for each method
gemini-flow agent spawn worker --objective "cd Karate/CKS && python CKS_Link_K_Shell.py && python iterative_cks.py" --gemini &
gemini-flow agent spawn worker --objective "cd Karate/CI && python collective_influence.py && python iterative_ci.py" --gemini &
gemini-flow agent spawn worker --objective "cd Karate/IE && python IE_informative_entropy.py && python iterative_ie.py" --gemini &
gemini-flow agent spawn worker --objective "cd Karate/Jaccard && python jaccard_index.py && python iterative_jaccard.py" --gemini &
gemini-flow agent spawn worker --objective "cd Karate/LDC && python LDC_link_degree_centrality.py && python iterative_ldc.py" --gemini &
gemini-flow agent spawn worker --objective "cd Karate/LLBC && python LLBC_contrast.py && python iterative_llbc_me.py" --gemini &
gemini-flow agent spawn worker --objective "cd Karate/ME && python ME_mapping_entropy.py && python iterative_me.py" --gemini &

echo "Waiting for all Karate agents to finish..."
wait
echo "Karate benchmarks completed! Results are in Karate/[Method]/results/"
