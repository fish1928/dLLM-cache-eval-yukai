#!/bin/bash
#################################################
# Run all four dLLM-Cache thread evals one by one (each = 6 benchmarks x
# {no-cache reference, tuned cache}). Resume-safe: finished (task, variant)
# folders are skipped, so rerunning after a crash continues where it stopped.
#
#   CUDA_VISIBLE_DEVICES=0 bash run_eval_all.sh
#   LIMIT=20 bash run_eval_all.sh                 # smoke everything first
#   THREADS="dream_base dream_instruct" CUDA_VISIBLE_DEVICES=1 bash run_eval_all.sh
#################################################

set -u
cd "$(dirname "$0")"

THREADS=${THREADS:-"llada_base llada_instruct dream_base dream_instruct"}

for thread in $THREADS; do
    echo "==================== $thread ===================="
    bash "run_eval_${thread}.sh" || echo "[warn] run_eval_${thread}.sh failed, continuing"
done
