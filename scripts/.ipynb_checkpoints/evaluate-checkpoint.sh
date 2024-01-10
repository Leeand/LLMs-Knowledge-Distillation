#!/bin/bash
export PYTHONPATH='.'

base_model=$1 # e.g., huggyllama/llama-7b
tune_ckpt_name=$2  # e.g., tune_log/llama_0.2 or student_tune_log/llama_0.2
prune_ckpt=$3 # e.g., prune_log/llama_prune
globalstep=("${@:4}")

for gs in "${globalstep[@]}";
do
    tune_id="${tune_ckpt_name##*/}"
    python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name/globalstep_$gs,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --output_path results/${tune_id}_$gs.json --no_cache
done
