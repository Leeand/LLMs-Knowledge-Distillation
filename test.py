import subprocess
import sys
sys.path.append("../LLMPruner")

sweep_config="llama5.4B_k8_h0.5_t0.3_dt_in_icconfig0_kdtoriginal_kd"
base_model = "huggyllama/llama-7b"
# tune_ckpt_name = output_path+"/globalstep_3250"
tune_ckpt_name = "tune_log/llama5.4B_k8_h0.5_t0.3_dt_in_icconfig0_kdtoriginal_kd/globalstep_1084"
prune_ckpt = "prune_log/llama_prune"
tune_id = "llama5.4B_k8_h0.5_t0.3_dt_in_icconfig0_kdtoriginal_kd"
shell_command = f"""
    base_model="{base_model}" 
    tune_ckpt_name="{tune_ckpt_name}"  
    prune_ckpt="{prune_ckpt}" 
    tune_id="{tune_id}"
    accelerate launch --main_process_port 29501 ./lm_evaluation_harness/lm_eval --model hf --model_args  checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --output_path results_sweep/${{tune_id}}_test01182.json
"""
print("-------------------------------------------")
print(shell_command)
process = subprocess.run(shell_command, shell=True, capture_output=True, text=True)
print(process.stderr)
test_result = process.stdout
print(test_result)