#!/bin/bash
#################################################
# Shared launcher for the four run_eval_*.sh wrappers (dLLM-Cache official
# code, evaluated as a baseline for dllm-meta).
#
# Faithful to the repo's own scripts/ (model_args, gen_kwargs, fewshot, cache
# intervals are copied verbatim per task), with these deliberate changes:
#   - accelerate: --num_processes $NUM_PROCESSES (default 1) + your
#     CUDA_VISIBLE_DEVICES, instead of accelerate_config.yaml (hardcodes 8 GPUs)
#   - every run gets --trust_remote_code (their scripts only add it for bbh;
#     harmless elsewhere) and code tasks get --confirm_run_unsafe_code.
#     HF_ALLOW_CODE_EVAL / HF_DATASETS_TRUST_REMOTE_CODE are exported by
#     evaluation_script.py itself, and exported here too for older paths.
#   - per-task output folders + resume (a folder already holding a
#     results*.json is skipped; delete it to rerun)
#   - LIMIT env caps samples for smoke runs
#
# Env (read by the wrappers):
#   CUDA_VISIBLE_DEVICES  gpu selection            (default: inherit)
#   NUM_PROCESSES         accelerate processes     (default 1)
#   PORT                  accelerate main port     (default per wrapper)
#   BATCH_SIZE            default 2 (as official)
#   LIMIT                 lm_eval --limit          (default: full set)
#   FILTER_TASK           run only this task (exact lm_eval task name,
#                         note dream gsm8k is 'gsm8k_cot')
#   RUN_BASELINE=0        skip the no-cache reference run
#   RUN_CACHE=0           skip the cache run
#   FOLDER_OUT            results root (default results_baseline_eval/<thread>)
#################################################

# CANONICAL dllm-meta generation-length spec -- every baseline eval (this repo
# and future ones: fast-dllm, dkv, flashdlm) uses these lengths per task so
# results are comparable with the dllm-meta harness sweeps. ifeval/followbench
# are dllm-meta-only (oracle/router work) and are NOT run in baseline evals.
#   gsm8k          256   (dream: gsm8k_cot)
#   minerva_math   512
#   bbh            256
#   mbpp           512
#   humaneval      512
#   truthfulqa_gen 256

# DEVICE convenience (same convention as the dllm-meta scripts): DEVICE=cuda:1
# pins this run to ONE gpu and always wins (an inherited CUDA_VISIBLE_DEVICES
# from a jupyter/scheduler session is not a per-run choice). If the session
# already restricts CUDA_VISIBLE_DEVICES (e.g. "2,3"), cuda:N selects the N-th
# entry of that list -- matching torch device numbering, which is relative to
# the visible set.
DEVICE=${DEVICE:-}
if [ -n "$DEVICE" ]; then
    _idx="${DEVICE#cuda:}"
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -r -a _gpus <<< "$CUDA_VISIBLE_DEVICES"
        if [ "$_idx" -ge "${#_gpus[@]}" ]; then
            echo "[error] DEVICE=$DEVICE but CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES has only ${#_gpus[@]} gpu(s)"; exit 1
        fi
        export CUDA_VISIBLE_DEVICES="${_gpus[$_idx]}"
    else
        export CUDA_VISIBLE_DEVICES="$_idx"
    fi
    echo "[device] DEVICE=$DEVICE -> CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1

NUM_PROCESSES=${NUM_PROCESSES:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
LIMIT=${LIMIT:-}
FILTER_TASK=${FILTER_TASK:-}
RUN_BASELINE=${RUN_BASELINE:-1}
RUN_CACHE=${RUN_CACHE:-1}

_skip_task () {
    [ -n "$FILTER_TASK" ] && [ "$1" != "$FILTER_TASK" ]
}

_done_already () {
    # lm_eval nests results under a model subfolder; any results*.json counts
    [ -d "$1" ] && find "$1" -name "results*.json" 2>/dev/null | grep -q .
}

# _launch <model_class> <task> <fewshot> <variant> <model_args> <gen_kwargs|-> [extra flags...]
_launch () {
    local model_class=$1 task=$2 fewshot=$3 variant=$4 model_args=$5 gen_kwargs=$6
    shift 6

    local folder_task="$FOLDER_OUT/${task}_${variant}"
    if _done_already "$folder_task"; then
        echo "[skip] $folder_task has results"
        return 0
    fi

    local args_gen=()
    if [ "$gen_kwargs" != "-" ]; then
        args_gen=(--gen_kwargs "$gen_kwargs")
    fi

    echo "[eval] model=$model_class task=$task variant=$variant fewshot=$fewshot"
    accelerate launch --num_processes "$NUM_PROCESSES" --main_process_port "$PORT" \
        evaluation_script.py \
        --model "$model_class" \
        --tasks "$task" \
        --batch_size "$BATCH_SIZE" \
        --model_args "$model_args" \
        ${args_gen[@]+"${args_gen[@]}"} \
        --num_fewshot "$fewshot" \
        --output_path "$folder_task" \
        --log_samples \
        --trust_remote_code \
        ${LIMIT:+--limit "$LIMIT"} \
        "$@" \
        || echo "[warn] failed: $task/$variant -- continuing"
}

# run_llada_task <task> <fewshot> <gen_kwargs> <cache_args> <extra_model_args|-> [extra flags...]
# baseline model_args use the no-cache intervals; cache_args are the tuned
# intervals from the official per-task script
run_llada_task () {
    local task=$1 fewshot=$2 gen_kwargs=$3 cache_args=$4 extra_margs=$5
    shift 5
    _skip_task "$task" && return 0

    local margs_tail=""
    if [ "$extra_margs" != "-" ]; then
        margs_tail=",$extra_margs"
    fi

    if [ "$RUN_BASELINE" = "1" ]; then
        _launch LLaDA "$task" "$fewshot" nocache \
            "pretrained=$PRETRAINED,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,cache_order=0,is_feature_cache=False,is_cfg_cache=False$margs_tail" \
            "$gen_kwargs" "$@"
    fi
    if [ "$RUN_CACHE" = "1" ]; then
        _launch LLaDA "$task" "$fewshot" cache \
            "pretrained=$PRETRAINED,$cache_args,transfer_ratio=0.25,cache_order=0,is_feature_cache=True,is_cfg_cache=False$margs_tail" \
            "$gen_kwargs" "$@"
    fi
}

# run_dream_task <task> <fewshot> <len> <temperature> <style: alg|bos> <cache_args> [extra flags...]
run_dream_task () {
    local task=$1 fewshot=$2 len=$3 temperature=$4 style=$5 cache_args=$6
    shift 6
    _skip_task "$task" && return 0

    local margs_style="alg=entropy,alg_temp=0.0"
    if [ "$style" = "bos" ]; then
        margs_style="add_bos_token=true"    # the official minerva scripts use this form
    fi
    local margs_common="pretrained=$PRETRAINED,max_new_tokens=$len,diffusion_steps=$len,temperature=$temperature,top_p=0.95,$margs_style"

    if [ "$RUN_BASELINE" = "1" ]; then
        _launch dream "$task" "$fewshot" nocache \
            "$margs_common,prompt_interval_steps=-1,gen_interval_steps=-1,cfg_interval_steps=-1,transfer_ratio=0,is_feature_cache=False,is_cfg_cache=False" \
            - "$@"
    fi
    if [ "$RUN_CACHE" = "1" ]; then
        _launch dream "$task" "$fewshot" cache \
            "$margs_common,$cache_args,transfer_ratio=0.25,is_feature_cache=True,is_cfg_cache=False" \
            - "$@"
    fi
}
