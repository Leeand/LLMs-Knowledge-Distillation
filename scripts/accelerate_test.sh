base_model="huggyllama/llama-7b"
tune_ckpt_name="/workspace/LLM_Pruner_main/tune_log/dynamic_T_k0.000001_h2_t_DDP/globalstep_1084"  
prune_ckpt="prune_log/llama_prune"
tune_id="5"
accelerate launch --main_process_port 29501 ./lm_evaluation_harness/lm_eval --model hf --model_args  checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --output_path results_sweep/${tune_id}.json