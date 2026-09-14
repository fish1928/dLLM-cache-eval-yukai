#!/bin/bash
#################################################
# dLLM-Cache official eval -- dream-instruct thread.
# Settings copied from scripts/run_Dream_*_Instruct.sh (gsm8k task is
# gsm8k_cot at 8-shot; minerva uses the add_bos_token form; no
# --apply_chat_template anywhere -- faithful to the official scripts).
# truthfulqa_gen has NO official recipe: mirrored from bbh Instruct (10/2/1).
#
#   CUDA_VISIBLE_DEVICES=1 bash run_eval_dream_instruct.sh
#################################################

set -u
cd "$(dirname "$0")"

PRETRAINED="Dream-org/Dream-v0-Instruct-7B"
PORT=${PORT:-29613}
FOLDER_OUT=${FOLDER_OUT:-results_baseline_eval/dream_instruct}
source eval_common.sh

# gen lengths follow the dllm-meta spec (see eval_common.sh): minerva/mbpp/
# humaneval -> 512 (official scripts used 256; diffusion_steps scales with len)
run_dream_task gsm8k_cot      8   256  0.2   alg    "prompt_interval_steps=25,gen_interval_steps=2,cfg_interval_steps=1"   --confirm_run_unsafe_code
run_dream_task minerva_math   4   512  0.2   bos    "prompt_interval_steps=50,gen_interval_steps=1,cfg_interval_steps=1"
run_dream_task bbh            3   256  0.2   alg    "prompt_interval_steps=10,gen_interval_steps=2,cfg_interval_steps=1"
run_dream_task mbpp           3   512  0.2   alg    "prompt_interval_steps=10,gen_interval_steps=8,cfg_interval_steps=1"   --confirm_run_unsafe_code
run_dream_task humaneval      0   512  0.2   alg    "prompt_interval_steps=50,gen_interval_steps=1,cfg_interval_steps=1"   --confirm_run_unsafe_code
run_dream_task truthfulqa_gen 0   256  0.2   alg    "prompt_interval_steps=10,gen_interval_steps=2,cfg_interval_steps=1"

echo "[done] dream_instruct -> $FOLDER_OUT"
