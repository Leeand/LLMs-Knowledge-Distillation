prune_ckpt_path='llama_prune'

# need to change
tune_ckpt_path='llama_13000yahma_alpaca_vast_platform' 

# hyperparamaters_name='dynamic_T_k0.00001_h1_t_DDP' # update the name according the hyperparamaters

# echo "[START] - Start Pruning Model"
# CUDA_VISIBLE_DEVICES=0 python hf_prune.py --pruning_ratio 0.25 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_first --save_model
# echo "[FINISH] - Finish Pruning Model"

# Teacher
echo "[START] - Start Teacher LORA SFT"
CUDA_VISIBLE_DEVICES=0 python teacher_LORA_post_training.py \
    --data_path yahma/alpaca-cleaned \
    --output_dir teacher_tune_log/$tune_ckpt_path \
    --wandb_project llama_tune \
    --lora_r 8 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 128
echo "[FINISH] - Finish Teacher Training."




# student_tune_log/$tune_ckpt_path
# tune_log/$hyperparamaters_name/globalstep_3250 --global_step_start 3250

