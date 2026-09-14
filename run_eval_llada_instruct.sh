#!/bin/bash
#################################################
# dLLM-Cache official eval -- llada-instruct thread.
# Settings copied from scripts/run_LLaDA_*_Instruct.sh, INCLUDING their
# per-task choices about chat templating (gsm8k/bbh/mbpp use
# --apply_chat_template [--fewshot_as_multiturn], minerva chat-only,
# humaneval deliberately none -- faithful to the official scripts).
# truthfulqa_gen has NO official recipe: mirrored from bbh Instruct
# (block 256, 50/6/1) + chat template.
#
#   CUDA_VISIBLE_DEVICES=0 bash run_eval_llada_instruct.sh
#################################################

set -u
cd "$(dirname "$0")"

PRETRAINED="GSAI-ML/LLaDA-8B-Instruct"
PORT=${PORT:-29611}
FOLDER_OUT=${FOLDER_OUT:-results_baseline_eval/llada_instruct}
source eval_common.sh

# Cache intervals (Kp=prompt, Kr=gen) follow the PAPER's appendix Table 14
# (arXiv 2506.06295v3), authoritative over the repo scripts where the two
# disagree (bbh: paper 100/5, repo script 50/6).
run_llada_task gsm8k          4   "block_length=8,gen_length=256,steps=256,cfg_scale=0.0"               "prompt_interval_steps=50,gen_interval_steps=7"                -           --apply_chat_template --fewshot_as_multiturn
# minerva length follows the dllm-meta spec (512, official used 256); the
# official full block (block_length == gen_length) scales with it
run_llada_task minerva_math   0   "block_length=512,gen_length=512,steps=512,cfg_scale=0.0"             "prompt_interval_steps=50,gen_interval_steps=1,cfg_interval_steps=1" -     --apply_chat_template
run_llada_task bbh            3   "block_length=256,gen_length=256,steps=256,cfg_scale=0.0"             "prompt_interval_steps=100,gen_interval_steps=5,cfg_interval_steps=1" -    --apply_chat_template --fewshot_as_multiturn
run_llada_task mbpp           3   "block_length=32,gen_length=512,steps=512,cfg_scale=0.0,remasking=low_confidence" "prompt_interval_steps=100,gen_interval_steps=5"   -           --confirm_run_unsafe_code --apply_chat_template --fewshot_as_multiturn
run_llada_task humaneval      0   "block_length=32,gen_length=512,steps=512,cfg_scale=0.0"              "prompt_interval_steps=25,gen_interval_steps=5"                add_bos_token=True --confirm_run_unsafe_code
run_llada_task truthfulqa_gen 0   "block_length=256,gen_length=256,steps=256,cfg_scale=0.0"             "prompt_interval_steps=100,gen_interval_steps=5,cfg_interval_steps=1" -    --apply_chat_template

echo "[done] llada_instruct -> $FOLDER_OUT"
