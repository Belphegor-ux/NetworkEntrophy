#!/usr/bin/env bash
# One-shot regeneration driver for the conference results refresh.
# Runs Tokyo (static + iterative) and all 7 "Networks to check" datasets
# through run_network_suite.py with size-appropriate step sizes.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe

echo "==================== TOKYO STATIC ===================="
$PY PowerGrid_City/run_city_analysis.py

echo "==================== TOKYO ITERATIVE ===================="
for m in ldc jaccard lks ci llbc; do
  echo "--- tokyo iterative: $m ---"
  $PY PowerGrid_City/run_iterative_analysis.py --method "$m"
done
echo "--- tokyo validation ---"
$PY PowerGrid_City/validate_metrics.py

echo "==================== NEW NETWORKS ===================="
# name|csv|step_size
runs=(
  "karate_check|Networks to check/karate.csv|1"
  "dolphins|Networks to check/dolphins.csv|1"
  "lesmis|Networks to check/lesmis_edges.csv|1"
  "football_new|Networks to check/football.csv|1"
  "baseball|Networks to check/Baseball steroid player_edges.csv|8"
  "transport|Networks to check/2transport_edges_re.csv|3"
  "hermaphrodite|Networks to check/hermaphrodite_gap_junction.csv|15"
)
for r in "${runs[@]}"; do
  IFS='|' read -r name csv step <<< "$r"
  echo "--- suite: $name (step=$step) ---"
  $PY run_network_suite.py --csv "$csv" --name "$name" --step-size "$step"
done

echo "==================== ALL DONE ===================="
