import json

tune_id = 'k8_h0.5_t0.3_dt_in_icconfig0_kdtoriginal_kd'
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
