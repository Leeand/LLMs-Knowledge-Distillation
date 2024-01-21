prune_ckpt_path='TinyLlama_1.1B_prune'
tune_ckpt_path='llama1B_13000yahma_alpaca_vast_platform'
teacher_tune_ckpt_path='llama_13000yahma_alpaca_vast_platform'
hyperparamaters_name='llama0.75B' 
 
echo "[START] - Start Student Distill"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 distill_training_wandb_sweep.py --teacher_model_config teacher_tune_log/$teacher_tune_ckpt_path/ --student_tune_model_config student_tune_log/$tune_ckpt_path --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$hyperparamaters_name --wandb_project grid_search_1.1B --lora_r 16 --num_epochs 2 --learning_rate 1e-4 --batch_size 10