#!/usr/bin/env bash
# Run Jazz and Tokyo through the unified suite so all dashboard networks share
# the same 9-metric schema. Jazz is dense (2742 edges) -> larger step size.
set -u
cd "$(dirname "$0")"
PY=.venv/Scripts/python.exe

echo "--- suite: tokyo (step=1) ---"
$PY run_network_suite.py --csv "_derived_tokyo.csv" --name tokyo --step-size 1

echo "--- suite: jazz (step=30) ---"
$PY run_network_suite.py --csv "_derived_jazz.csv" --name jazz --step-size 30

echo "==================== JAZZ+TOKYO DONE ===================="
