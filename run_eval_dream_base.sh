#!/bin/bash
#################################################
# dLLM-Cache official eval -- dream-base thread.
# Settings copied from scripts/run_Dream_*_base.sh (note their gsm8k task is
# gsm8k_cot at 8-shot; minerva uses the add_bos_token model_args form, the
# rest use alg=entropy; temperature 0.2 / top_p 0.95 everywhere; the official
# scripts never pass --apply_chat_template for Dream).
# truthfulqa_gen has NO official recipe: mirrored from bbh base (25/4/1).
#
#   CUDA_VISIBLE_DEVICES=1 bash run_eval_dream_base.sh
#################################################

set -u
cd "$(dirname "$0")"

PRETRAINED="Dream-org/Dream-v0-Base-7B"
PORT=${PORT:-29612}
FOLDER_OUT=${FOLDER_OUT:-results_baseline_eval/dream_base}
source eval_common.sh

#             task           fs  len  temp  style  cache intervals
run_dream_task gsm8k_cot      8   256  0.2   alg    "prompt_interval_steps=100,gen_interval_steps=8,cfg_interval_steps=1"
run_dream_task minerva_math   4   256  0.2   bos    "prompt_interval_steps=100,gen_interval_steps=4,cfg_interval_steps=1"
run_dream_task bbh            3   256  0.2   alg    "prompt_interval_steps=25,gen_interval_steps=4,cfg_interval_steps=1"
run_dream_task mbpp           3   256  0.2   alg    "prompt_interval_steps=25,gen_interval_steps=8,cfg_interval_steps=1"  --confirm_run_unsafe_code
run_dream_task humaneval      0   256  0.2   alg    "prompt_interval_steps=5,gen_interval_steps=1,cfg_interval_steps=1"   --confirm_run_unsafe_code
run_dream_task truthfulqa_gen 0   256  0.2   alg    "prompt_interval_steps=25,gen_interval_steps=4,cfg_interval_steps=1"

echo "[done] dream_base -> $FOLDER_OUT"
