#!/bin/bash
export PYTHONPATH='.'

base_model="huggyllama/llama-13b" # e.g., huggyllama/llama-13b
tune_ckpt_name="teacher_tune_log/llama_13000yahma_alpaca_vast_platform"  # e.g., teacher_tune_log/llama_13000yahma_alpaca_vast_platform

tune_id="${tune_ckpt_name##*/}"
python lm-evaluation-harness/main.py --model hf-causal-experimental --model_args  pretrained=$base_model,peft=$tune_ckpt_name,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0 --output_path results/${tune_id}.json --no_cache
# accelerate launch lm-evaluation-harness/main.py --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --device cuda:0,1,2,3 --output_path results/${tune_id}.json --no_cache

# accelerate launch '80GB' lm-evaluation-harness/main.py  --model hf-causal-experimental --model_args checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --output_path results/${tune_id}.json --no_cache
