# 定义不同的kd_loss_weight和hard_label_weight值
prune_ckpt_path='llama_prune'
tune_ckpt_path='llama_13000yahma_alpaca'
kd_loss_weights=(0.000001 0.00001 0.0001 0.001)
hard_label_weights=(0.5 1 2)

# 循环遍历不同的权重组合
for kd_loss_weight in "${kd_loss_weights[@]}"; do
    for hard_label_weight in "${hard_label_weights[@]}"; do
        # 更新文件名以包含权重信息
        hyperparamaters_name="dynamic_T_k${kd_loss_weight}_h${hard_label_weight}_t_DDP"
        output_dir_name="tune_log/${hyperparamaters_name}"

        echo "[START] - Running with kd_loss_weight=${kd_loss_weight}, hard_label_weight=${hard_label_weight}"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node 8 distill_training.py \
            --teacher_model_config teacher_tune_log/$tune_ckpt_path/ \
            --student_tune_model_config student_tune_log/$tune_ckpt_path \
            --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin \
            --data_path yahma/alpaca-cleaned \
            --output_dir $output_dir_name \
            --wandb_project improvement_project \
            --lora_r 8 \
            --num_epochs 2 \
            --learning_rate 1e-4 \
            --batch_size 3 \
            --kd_loss_weight $kd_loss_weight \
            --hard_label_weight $hard_label_weight \
            --temperature 20 \
            --kd_type dynamic_temperature
        echo "[FINISH] - Finished Running with kd_loss_weight=${kd_loss_weight}, hard_label_weight=${hard_label_weight}"
        echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {$output_dir_name}/"
    done
done

echo "You can use the command:"
echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
echo "to use the pruned model"
