#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
make build
mkdir -p results
# Prevent old output from being mistaken for this run's successful evidence.
if [ -e results/run.log ]; then
  echo 'results/run.log already exists. Rename results before another run.' >&2
  exit 1
fi
{ date -u; nvcc --version; nvidia-smi; } > results/environment.txt 2>&1
./edge_detect --count 512 --width 256 --height 256 --batch 32 --seed 42 --output results 2>&1 | tee results/run.log
python3 scripts/package_evidence.py
