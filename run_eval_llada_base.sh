#!/bin/bash
#################################################
# dLLM-Cache official eval -- llada-base thread.
# Six benchmarks, each as no-cache reference + tuned cache run (intervals
# copied from the repo's scripts/run_LLaDA_*_base.sh).
# truthfulqa_gen has NO official recipe: gen settings + cache intervals are
# mirrored from the bbh base script (block 256, 50/6/1).
#
#   CUDA_VISIBLE_DEVICES=0 bash run_eval_llada_base.sh
#   LIMIT=20 FILTER_TASK=gsm8k bash run_eval_llada_base.sh    # smoke
#################################################

set -u
cd "$(dirname "$0")"

PRETRAINED="GSAI-ML/LLaDA-8B-Base"
PORT=${PORT:-29610}
FOLDER_OUT=${FOLDER_OUT:-results_baseline_eval/llada_base}
source eval_common.sh

# gen lengths follow the dllm-meta spec (see eval_common.sh), overriding the
# official scripts where they differ (minerva/mbpp/humaneval -> 512); when the
# official block_length equaled gen_length it scales with it, small semi-AR
# blocks (32) stay as published.
# Cache intervals (Kp=prompt, Kr=gen) follow the PAPER's appendix Table 13
# (arXiv 2506.06295v3), which is authoritative over the repo scripts where the
# two disagree (humaneval: paper 100/5, repo script 50/5).
#              task           fs  gen_kwargs                                                            cache intervals                                                extra_margs
run_llada_task gsm8k          4   "block_length=256,gen_length=256,steps=256,cfg_scale=0"               "prompt_interval_steps=25,gen_interval_steps=5"                -
run_llada_task minerva_math   4   "block_length=512,gen_length=512,steps=512,cfg_scale=0.0"             "prompt_interval_steps=50,gen_interval_steps=8,cfg_interval_steps=1" -
run_llada_task bbh            3   "block_length=256,gen_length=256,steps=256,cfg_scale=0.0"             "prompt_interval_steps=50,gen_interval_steps=6,cfg_interval_steps=1" -
run_llada_task mbpp           3   "block_length=32,gen_length=512,steps=512,cfg_scale=0.0,remasking=low_confidence" "prompt_interval_steps=25,gen_interval_steps=4"    -           --confirm_run_unsafe_code
run_llada_task humaneval      0   "block_length=512,gen_length=512,steps=512,cfg_scale=0.0"             "prompt_interval_steps=100,gen_interval_steps=5"               add_bos_token=True --confirm_run_unsafe_code
run_llada_task truthfulqa_gen 0   "block_length=256,gen_length=256,steps=256,cfg_scale=0.0"             "prompt_interval_steps=50,gen_interval_steps=6,cfg_interval_steps=1" -

echo "[done] llada_base -> $FOLDER_OUT"
