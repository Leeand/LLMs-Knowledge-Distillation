prune_ckpt_path='llama_prune'
# need to change
tune_ckpt_path='llama_13000yahma_alpaca_vast_platform' 
# Student
echo "[START] - Start Student Tuning"
CUDA_VISIBLE_DEVICES=1 python student_post_training.py \
    --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin \
    --data_path yahma/alpaca-cleaned \
    --output_dir student_tune_log/$tune_ckpt_path \
    --wandb_project student_llama_tune \
    --lora_r 16 \
    --num_epochs 2 \
    --learning_rate 1e-4 \
    --batch_size 64
echo "[FINISH] - Finish Prune and Post-Training."
echo "[INFO] - The prune d model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the Student LORA weight is at {student_tune_log/$tune_ckpt_path}/"

