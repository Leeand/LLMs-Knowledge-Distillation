'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''
import pdb
import json
import os
import sys
import argparse
from typing import List
import pickle
from datasets import load_from_disk
from pathlib import Path
import datasets
import torch
import transformers
from datasets import load_dataset
from textbrewer import GeneralDistiller, TrainingConfig, DistillationConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, LlamaTokenizer, \
    LlamaForCausalLM
from transformers import AutoConfig, AutoModel
from LLMPruner.peft import PeftModel, PeftConfig
import pdb
import time
from packaging import version
from CustomDistiller import CustomDistiller
from LLMPruner.peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
import wandb
from textbrewer.distiller_utils import auto_forward
from LLMPruner.utils.prompter import Prompter, ZeroPrompter
from LLMPruner.datasets.ppl_dataset import get_loaders
from utils import DistillDataCollatorForSeq2Seq, evaluate_metric
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import fnmatch
from lm_evaluation_harness.lm_eval import tasks,evaluator
from accelerate import Accelerator, DistributedType, find_executable_batch_size
import subprocess
import gc

local_rank = int(os.environ.get('LOCAL_RANK', -1))
print("Available Device:GPU-", local_rank)
device = torch.device("cuda", local_rank)
dist.init_process_group(backend='nccl') 
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

def evaluate_one_epoch(student_model, choosen_tasks, output_path):

    accelerator = Accelerator("/root/.cache/huggingface/accelerate/default_config.yaml")
    eval_student_model = accelerator.prepare_model(student_model, evaluation_mode=True)
    task_names = choosen_tasks.split(",")
    print(f"Selected Tasks: {task_names}")
    results = evaluator.simple_evaluate(
        model=eval_student_model,
        tasks=task_names
    )
    dumped = json.dumps(results, indent=2)
    print(dumped)

    if output_path:
        directory_path = os.path.dirname(output_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(output_path, "w") as f:
            f.write(dumped)

    print(evaluator.make_table(results))
    return results

def main():
    print("we get in rank 0")
    if local_rank == 0:
        # wandb
        print("we get in rank 0")
        run = wandb.init(project=os.environ["WANDB_PROJECT"], name="temporary_name")
        wandbsweep_config = run.config
        # 将参数写入文件
        config_params = {
            'hard_label_weight': wandbsweep_config.hard_label_weight,
            'kd_loss_weight': wandbsweep_config.kd_loss_weight,
            'temperature': wandbsweep_config.temperature,
            'dtnormalization_type': wandbsweep_config.dtnormalization_type,
            'intermediate_normalization_type': wandbsweep_config.intermediate_normalization_type,
            'intermediate_control_config': wandbsweep_config.intermediate_control_config,
            'kd_type': wandbsweep_config.kd_type
        }
        with open('sweep_params.json', 'w') as outfile:
            json.dump(config_params, outfile)
    # 同步点确保主进程写文件后其他进程再继续
    dist.barrier()
    print("we get after barrier 1")
    # 在所有进程中读取参数文件
    if local_rank != 0:
        with open('sweep_params.json', 'r') as infile:
            config_params = json.load(infile)
    # 使用读取的参数设置args，这段代码对所有进程都适用
    print("we get paramaters")
    dist.barrier()
    args.hard_label_weight = config_params['hard_label_weight']
    args.kd_loss_weight = config_params['kd_loss_weight']
    args.temperature = config_params['temperature']
    args.dtnormalization_type = config_params['dtnormalization_type']
    args.intermediate_normalization_type = config_params['intermediate_normalization_type']
    args.intermediate_control_config = config_params['intermediate_control_config']
    args.kd_type = config_params['kd_type']
    print("we pass paramaters")
    '''args.hard_label_weight = wandbsweep_config.hard_label_weight
    args.kd_loss_weight = wandbsweep_config.kd_loss_weight
    args.temperature = wandbsweep_config.temperature
    args.dtnormalization_type = wandbsweep_config.dtnormalization_type
    args.intermediate_normalization_type = wandbsweep_config.intermediate_normalization_type
    args.intermediate_control_config = wandbsweep_config.intermediate_control_config
    args.kd_type = wandbsweep_config.kd_type'''
        
    dist.barrier()

    sweep_config = f'k{str(args.kd_loss_weight)}_h{str(args.hard_label_weight)}_t{str(args.temperature)}_dt{str(args.dtnormalization_type)}_in{str(args.intermediate_normalization_type)}_ic{str(args.intermediate_control_config)}_kdt{str(args.kd_type)}'
    dist.barrier()
    
    if local_rank==0:
        wandb.run.name = sweep_config
        wandb.run.save()  
    print("we save paramaters")  
    dist.barrier()
    # Load Pruned Model
    pruned_dict = torch.load(args.prune_model, map_location='cpu')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    if not args.no_instruction:
        prompter = Prompter(template_name = args.prompt_template_name)
    else:
        prompter = ZeroPrompter()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # if device == 'cuda:0':
    #     model.half()

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if 'lamini' in args.data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                None,
                data_point["response"],
            )
        elif 'alpaca' in args.data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"], 
            )
        elif 'boolq' in args.data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["question"],
                data_point["passage"],
                data_point["answer"], 
            )
        else:
            raise NotImplementedError

        tokenized_full_prompt = tokenize(full_prompt)
        if not args.train_on_inputs:
            if 'alpaca' in args.data_path.lower() or 'lamini' in args.data_path.lower():
                user_prompt = prompter.generate_prompt(
                    data_point["instruction"], data_point["input"] if 'input' in data_point.keys() else None,
                )
                tokenized_user_prompt = tokenize(
                    user_prompt, add_eos_token=args.add_eos_token
                )
                user_prompt_len = len(tokenized_user_prompt["input_ids"])

                if args.add_eos_token:
                    user_prompt_len -= 1

                tokenized_full_prompt["labels"] = [
                    -100
                ] * user_prompt_len + tokenized_full_prompt["labels"][
                    user_prompt_len:
                ]  # could be sped up, probably
            elif 'boolq' in args.data_path.lower():
                user_prompt = prompter.generate_prompt(
                    data_point["question"],
                    data_point["passage"],
                )
                tokenized_user_prompt = tokenize(
                    user_prompt, add_eos_token=args.add_eos_token
                )
                user_prompt_len = len(tokenized_user_prompt["input_ids"])

                if args.add_eos_token:
                    user_prompt_len -= 1
                
                # print("=============================================")
                # print(len(tokenized_full_prompt["input_ids"]))
                # print(user_prompt_len)
                # if len(tokenized_full_prompt["input_ids"]) == user_prompt_len:
                #     print(full_prompt.replace("\n", "\\n"))
                #     print(user_prompt.replace("\n", "\\n"))
                #     pdb.set_trace()
                # print("=============================================")

                tokenized_full_prompt["labels"] = [
                    -100
                ] * user_prompt_len + tokenized_full_prompt["labels"][
                    user_prompt_len:
                ]  # could be sped up, probably
        # pdb.set_trace()
        return tokenized_full_prompt

    def split_and_tokenizer(test_data, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(test_data[field_name]), return_tensors='pt').input_ids[0]
        nsamples = test_ids.numel() // seq_len

        test_set = []
        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_set.append({
                'input_ids': batch,
                'labels': batch
            })
        return test_set

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print("device:", device)

    # Prepare For LoRA

    model = prepare_model_for_int8_training(model)
    model.config.output_hidden_states=True
    # pdb.set_trace()
    if args.student_tune_model_config is None:
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        model = PeftModel.from_pretrained(
            model,
            args.student_tune_model_config,
            torch_dtype=torch.float16,
        )


    # Load Train Dataset
    data = load_dataset(args.data_path)
    # print(args.data_path)、
    # print(data)
    if args.cache_dataset and os.path.exists('datasets/cache/{}_train'.format(args.data_path)) and os.path.exists('datasets/cache/{}_val'.format(args.data_path)):
        cache_file = 'datasets/cache/{}'.format(args.data_path)
        train_data, val_data = load_from_disk(cache_file+'_train'), load_from_disk(cache_file+'_val')
    else:
        train_val = data["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        train_data= train_data.select(range(13000))

        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )

        if args.cache_dataset and args.local_rank == -1:
            cache_file = 'datasets/cache/{}'.format(args.data_path)
            cache_dir = '/'.join(cache_file.split('/')[:-1])
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)
            train_data.save_to_disk(cache_file+'_train')
            val_data.save_to_disk(cache_file+'_val')
            print("dataset cache save success!")
    # Load Extra Validation Dataset
    if args.extra_val_dataset:
        from LLMPruner.datasets.ppl_dataset import get_wikitext2, get_ptb
        seq_len = 128
        for extra_dataset in args.extra_val_dataset.split(','):
            if 'wikitext2' in extra_dataset:
                _, test_data = get_wikitext2(seq_len, None)
                test_data = split_and_tokenizer(test_data, tokenizer, seq_len, field_name='text')
            if 'ptb' in extra_dataset:
                _, test_data = get_ptb(seq_len, None)
                test_data = split_and_tokenizer(test_data, tokenizer, seq_len, field_name='sentence')
            val_data[extra_dataset] = test_data


    # Teacher Llama-13B Student Llama-7B/tiny/460mb
    student_model = model
    student_tokenizer = tokenizer
    
    teacher_model = LlamaForCausalLM.from_pretrained(
            args.base_model,
            low_cpu_mem_usage=True if args.torch_version >= 1.9 else False,
            output_hidden_states=True
        )
    teacher_tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    if args.teacher_model_config is not None:
        teacher_model = PeftModel.from_pretrained(
            teacher_model,
            args.teacher_model_config,
            torch_dtype=torch.float16,
        )
    
    
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)
    

    teacher_num_trainable_params = sum(p.numel() for p in teacher_model.parameters() if p.requires_grad)
    print("Teacher Number of trainable parameters:", teacher_num_trainable_params)
    for name, param in student_model.named_parameters():
        if 'lora' in name:  
            param.requires_grad = True
    student_num_trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print("Student Number of trainable parameters:", student_num_trainable_params)

    # teacher_model = DDP(teacher_model, device_ids=[local_rank], output_device=local_rank)
    student_model = DDP(student_model, device_ids=[local_rank], output_device=local_rank)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # This will wrap your model and parallelize model computations
    #     student_model = torch.nn.DataParallel(student_model, device_ids=[0,1,2,3], output_device=0)
    #     teacher_model = torch.nn.DataParallel(teacher_model, device_ids=[0,1,2,3], output_device=0)

    teacher_model.eval()

    columns = ["input_ids", "attention_mask", "labels"]
    ignore_colums = list(set(train_data.column_names) - set(columns)) 
    if version.parse(datasets.__version__) < version.parse("1.4.0"):
        train_data = train_data.set_format(
            type=train_data.format["type"], columns=columns, format_kwargs=train_data.format["format_kwargs"]
        )
    else:
        train_data = train_data.remove_columns(ignore_colums)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, 
                                  collate_fn = transformers.DataCollatorForSeq2Seq(student_tokenizer, 
                                  pad_to_multiple_of=8, return_tensors="pt", padding=True),
                                  sampler=DistributedSampler(train_data))

    num_epochs = args.num_epochs
    print("number of steps",len(train_dataloader) * num_epochs)
    num_training_steps = len(train_dataloader) * num_epochs
    # Optimizer and learning rate scheduler
    optimizer = AdamW(student_model.parameters(), lr=args.learning_rate)

    scheduler_class = get_linear_schedule_with_warmup
    # arguments dict except 'optimizer'
    scheduler_args = {'num_warmup_steps': int(0.1 * num_training_steps), 'num_training_steps': num_training_steps}
    
    sweep_config = f'k{str(args.kd_loss_weight)}_h{str(args.hard_label_weight)}_t{str(args.temperature)}_dt{str(args.dtnormalization_type)}_in{str(args.intermediate_normalization_type)}_icc{str(args.intermediate_control_config)}_kdt{str(args.kd_type)}'
    output_path = args.output_dir+sweep_config

    #  最后一个block ，靠近output。然后再考虑加入其他的。 mse。算mse 之前进行归一化就可以。
    intermediate_matches_configs=[]
    if args.intermediate_control_config == '':
        intermediate_matches_configs = []
    elif args.intermediate_control_config == '7b-config1-last':
        intermediate_matches_configs = [
            {"layer_T": 40, "layer_S": 32, "feature": "hidden", "loss": "hidden_mse", "weight": args.intermediate_weight, "proj": ["linear", 4096, 5120]}
            ]
    elif args.intermediate_control_config == '7b-config2-middle':
        intermediate_matches_configs = [
            {"layer_T": 40, "layer_S": 32, "feature": "hidden", "loss": "hidden_mse", "weight": args.intermediate_weight, "proj": ["linear", 4096, 5120]},
            {"layer_T": 20, "layer_S": 16, "feature": "hidden", "loss": "hidden_mse", "weight": args.intermediate_weight, "proj": ["linear", 4096, 5120]}
            ]
    elif args.intermediate_control_config == '7b-config3-middle':
        intermediate_matches_configs = [
            {"layer_T": 40, "layer_S": 32, "feature": "hidden", "loss": "hidden_mse", "weight": args.intermediate_weight, "proj": ["linear", 4096, 5120]},
            {"layer_T": 20, "layer_S": 16, "feature": "hidden", "loss": "hidden_mse", "weight": args.intermediate_weight, "proj": ["linear", 4096, 5120]},
            {"layer_T": 15, "layer_S": 12, "feature": "hidden", "loss": "hidden_mse", "weight": args.intermediate_weight, "proj": ["linear", 4096, 5120]}
            ]
    elif args.intermediate_control_config == 'tinyl_last':
        intermediate_matches_configs =[
            {"layer_T": 40, "layer_S": 22, "feature": "hidden", "loss": "hidden_mse", "weight": args.intermediate_weight, "proj": ["linear", 2048, 5120]}]
    elif args.intermediate_control_config == 'tinyl_last_mid':
        intermediate_matches_configs =[
            {"layer_T": 40, "layer_S": 22, "feature": "hidden", "loss": "hidden_mse", "weight": args.intermediate_weight, "proj": ["linear", 2048, 5120]},
            {"layer_T": 20, "layer_S": 11, "feature": "hidden", "loss": "hidden_mse", "weight": args.intermediate_weight, "proj": ["linear", 2048, 5120]}
            ]
    elif args.intermediate_control_config == 'tinyl_last_mid_start':
        intermediate_matches_configs =[
            {"layer_T": 40, "layer_S": 22, "feature": "hidden", "loss": "hidden_mse", "weight": args.intermediate_weight, "proj": ["linear", 2048, 5120]},
            {"layer_T": 30, "layer_S": 16, "feature": "hidden", "loss": "hidden_mse", "weight": args.intermediate_weight, "proj": ["linear", 2048, 5120]},
            {"layer_T": 20, "layer_S": 11, "feature": "hidden", "loss": "hidden_mse", "weight": args.intermediate_weight, "proj": ["linear", 2048, 5120]}
            ]

        
        
        
    def simple_adaptor(batch, model_outputs):
        return {'logits': model_outputs.logits, 'hidden': model_outputs.hidden_states, 'losses': model_outputs.loss}


    hard_label_weight = args.hard_label_weight
    kd_loss_weight = args.kd_loss_weight
    print("=========================================================================")
    print("hard_label_weight is {}, kd_loss_weight is {}".format(hard_label_weight,kd_loss_weight))
    print("the temperature of kd loss is {}".format(args.temperature))
    print("=========================================================================")

    distill_config = DistillationConfig(
        temperature=args.temperature,
        intermediate_matches=intermediate_matches_configs,
        hard_label_weight = hard_label_weight,
        kd_loss_weight = kd_loss_weight
    )
    train_config = TrainingConfig(
        output_dir=output_path,
        ckpt_epoch_frequency=1, #多少个epoch保存1次
        ckpt_frequency = 10 #每一个epoch保存多少次
    )

    distiller = CustomDistiller(
        train_config=train_config, distill_config=distill_config,
        model_T=teacher_model, model_S=student_model,
        adaptor_T=simple_adaptor, adaptor_S=simple_adaptor,
        logits_pro = ["linear",32000, 32000], # 32000 50304
        global_step_start = args.global_step_start,
        use_softmax = args.use_softmax,
        dt_normalization_type = args.dtnormalization_type,
        intermediate_normalization_type = args.intermediate_normalization_type,
        kd_type = args.kd_type, #'dynamic_kd',
        intermediate_control_config=args.intermediate_control_config,
        layer_weight=args.intermediate_weight
        )
    
    
    # logits_pro = ["linear",student voc size, teac]
    '''with distiller:
        # pdb.set_trace()
        distiller.train(optimizer, train_dataloader, num_epochs, max_grad_norm=1)
        # , scheduler_class=scheduler_class, scheduler_args=scheduler_args'''

    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    model.state_dict = old_state_dict
    model.save_pretrained(args.output_dir)
    
    teacher_model = teacher_model.to('cpu')
    del teacher_model
    student_model = student_model.to('cpu')
    del model, student_model, optimizer
    torch.cuda.empty_cache()
    gc.collect()
    print(torch.cuda.memory_summary())
    
    start_time = time.time()
    base_model = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    # TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
    # ahxt/LiteLlama-460M-1T
    tune_ckpt_name = output_path+"/globalstep_1084"
    # tune_ckpt_name = "tune_log/middle_layer_k1_h1_t20_config2_lw1_DDP_r16_newteacher/globalstep_3250"
    prune_ckpt = "prune_log/TinyLlama_1.1B_prune"
    # LiteLlama_460M_prune
    tune_id = sweep_config
    if local_rank==0:
        # eval_results = evaluate_one_epoch(student_model, choosen_tasks='openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq', output_path='./results/result_k'+str(args.kd_loss_weight)+'_h'+str(args.hard_label_weight)+'_t'+str(args.temperature)+'.json')

        shell_command = f"""
            base_model="{base_model}" 
            tune_ckpt_name="{tune_ckpt_name}"  
            prune_ckpt="{prune_ckpt}" 
            tune_id="{tune_id}"
            accelerate launch --main_process_port 29501 ./lm_evaluation_harness/lm_eval --model hf --model_args  checkpoint=$prune_ckpt/pytorch_model.bin,peft=$tune_ckpt_name,config_pretrained=$base_model --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq --output_path results_sweep/${{tune_id}}.json
        """
        print("-------------------------------------------")
        print(shell_command)
        process = subprocess.run(shell_command, shell=True, capture_output=True, text=True)
        print(process.stderr)
        test_result = process.stdout
        print(test_result)

        eval_results = {}
        with open(f'results_sweep/{tune_id}.json', 'r') as file:
            eval_results = json.load(file)

        processed_results={}
        for item,value in eval_results['results'].items():
            if 'acc_norm' in value.keys() or 'acc_norm,none' in value.keys():
                processed_results[item]=value['acc_norm,none'] if 'acc_norm,none' in value.keys() else value['acc_norm']
            else: 
                processed_results[item]=value['acc,none'] if 'acc,none' in value.keys() else value['acc']
        print(processed_results)
        wandb.log(processed_results)
        wandb.finish()
    
    dist.barrier()
    end_time = time.time()
    duration = end_time - start_time
    print(f"Evaluation execution time: {duration} seconds")

if __name__ == "__main__":
    
    global args
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--teacher_model_config', type=str, default=None, help='tuned teacher model ckpt config')
    parser.add_argument('--base_model', type=str, default="huggyllama/llama-13b", help='base model name')
    parser.add_argument('--prune_model', type=str, help='prune model name')
    parser.add_argument('--student_tune_model_config', type=str, help='student LORA model name')
    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
    parser.add_argument('--cache_dataset', action="store_true", default=True)
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
    parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
    parser.add_argument('--prompt_template_name', type=str, default="alpaca", help="The prompt template to use, will default to alpaca.")
    parser.add_argument('--no_instruction', action='store_true', default=False, help="Whether to use the instruction template or not.")
    parser.add_argument('--hard_label_weight', type=float, default=1, help='student model label weight')
    parser.add_argument('--kd_loss_weight', type=float, default=1, help='the weight of kd loss between teacher and student models')
    parser.add_argument('--temperature', type=float, default=8, help='the temperature of kd loss')
    parser.add_argument('--kd_type', type=str, default="original_kd", help="Make sure the kd_type is in ['original_kd','dynamic_kd','focal_loss','dynamic_temperature']")
    parser.add_argument('--dtnormalization_type', type=str, default="", help="Make sure the dtnormalization_type is in ['','softmax','minmax','standardize']")
    parser.add_argument('--intermediate_normalization_type', type=str, default="", help="Make sure the intermediate_normalization_type is in ['','softmax','minmax']")
    parser.add_argument('--use_softmax', default=False, action="store_true", help='softmax in original_kd')
    parser.add_argument('--intermediate_control_config', type=str, default='config1', help='Choose the intermediate matches configuration')
    parser.add_argument('--intermediate_weight', type=float, default=0.1, help='Weight for intermediate matches')

    # Lora Configuration
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')

    # llm hyperparameters
    parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
    parser.add_argument('--add_eos_token', default=False, action="store_true")
    parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")
   
    # wandb params
    parser.add_argument('--wandb_project', type=str, default="")
    parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")

    #ddp
    parser.add_argument('--local-rank', type=int, default=-1)

    #checkpoint
    parser.add_argument('--global_step_start', type=int, default=0)
   
    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version
    print(vars(args))
    
    # Set WanDB
    os.environ["WANDB_PROJECT"] = args.wandb_project

    if local_rank==0:
        # 可选 grid random bayes
        sweep_config = {
            'method': 'grid'
        }
        with open('parameters_dict.json', 'r') as file:
            parameters_dict = json.load(file)
        sweep_config['parameters'] = parameters_dict
        sweep_id = wandb.sweep(sweep_config, project=os.environ["WANDB_PROJECT"]+"_wandb_sweep")
        print(sweep_id)
        wandb.agent(sweep_id, main)
    else:
        main()

