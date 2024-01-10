from textbrewer import GeneralDistiller
from textbrewer.distiller_utils import *
from textbrewer.distiller_basic import BasicDistiller
import pdb
from LLMPruner.peft import get_peft_model_state_dict
import wandb
import torch.distributed as dist
from utils import dynamic_kd_loss, focal_loss, dynamic_temperature, minmax_normalize, softmax_normalize, standardize_tensor


class CustomDistiller(GeneralDistiller):

    def __init__(self, train_config,
                 distill_config,
                 model_T,
                 model_S,
                 adaptor_T,
                 adaptor_S,
                 logits_pro,
                 global_step_start,
                 use_softmax,
                 dt_normalization_type,
                 intermediate_normalization_type,
                 kd_type : Optional[str] = "original_kd",
                 intermediate_control_config='config0',
                 layer_weight=0.1,
                 custom_matches: Optional[List[CustomMatch]] = None):
        # custom_matches=[{'module_T': module_T, 'module_S':module_S,
        #                 'loss': loss, 'weight': weight},...]
        
        super(CustomDistiller, self).__init__(train_config, distill_config, model_T, model_S, adaptor_T, adaptor_S)
        if dist.get_rank() == 0:
            wandb.login()
            experiment_name = f"middle_layer_DynamicT_k{self.d_config.kd_loss_weight}_h{self.d_config.hard_label_weight}_lw{layer_weight}_{intermediate_control_config}"
            run = wandb.init(
                name=experiment_name,
                # Set the project where this run will be logged
                project = os.environ["WANDB_PROJECT"],
                tags=['A100_8_SXM', 'Distillation_baseline_Middle'],
                notes='baseline plus intermedia distillation'
                # Track hyperparameters and run metadata
                )

        self.global_step_start = global_step_start
        self.use_softmax = use_softmax
        assert kd_type in ['original_kd','dynamic_kd','focal_loss','dynamic_temperature'],"kd_type is not in ['original_kd','dynamic_kd','focal_loss','dynamic_temperature']"
        self.kd_type = kd_type
        self.normalization_type = dt_normalization_type
        assert dt_normalization_type in ['','minmax','softmax','standardize'],"normalization_type is not in ['','minmax','softmax','standardize']"
        self.intermediate_normalization_type =  intermediate_normalization_type
        assert intermediate_normalization_type in ['','minmax','softmax'],"intermediate_normalization_type is not in ['','minmax','softmax']"
        self.dynamic_kd_loss = dynamic_kd_loss
        self.focal_loss = focal_loss
        self.dynamic_temperature= dynamic_temperature 
        self.projs = []
        self.projs_group = []
        for im in self.d_config.intermediate_matches:
            if im.proj is not None:
                projection = im.proj[0]
                dim_in = im.proj[1]
                dim_out = im.proj[2]
                self.projs_group.append(im.proj[3])
                self.projs.append(PROJ_MAP[projection](dim_in, dim_out))
                self.projs[-1].to(self.t_config.device)
            else:
                self.projs.append(None)
                self.projs_group.append(None)

        self.logits_projs=[]
        if  logits_pro is not None:
                projection = logits_pro[0]
                dim_in = logits_pro[1]
                dim_out = logits_pro[2]
                self.logits_projs.append(PROJ_MAP[projection](dim_in, 4096))
                self.logits_projs.append(PROJ_MAP[projection](4096, dim_out))
        for layer in self.logits_projs:
                layer.to(self.t_config.device)

        self.has_custom_matches = False
        if custom_matches:
            self.handles_T = []
            self.handles_S = []
            self.custom_matches_cache = {'hook_outputs_T': [], 'hook_outputs_S': [], 'match_proj_funcs': [],
                                         'match_weights': [], 'match_losses': [], 'match_proj_groups': []}
            for match in custom_matches:
                self.add_match(match)
            self.has_custom_matches = True

        self.d_config.is_caching_logits = False

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.FileHandler('distill.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def save_and_callback(self, global_step, step, epoch, callback):
        if self.has_custom_matches:
            handles_T = self.model_T._forward_hooks
            handles_S = self.model_S._forward_hooks
            self.model_S._forward_hooks = OrderedDict()  # clear hooks
            self.model_T._forward_hooks = OrderedDict()

        if self.rank != 0:
            torch.distributed.barrier()    # save and eval with single process
        else:
            self.logger.info(f"Saving at global step {global_step}, epoch step {step + 1} epoch {epoch+1}")
            coreModel = self.model_S.module if hasattr(self.model_S, "module") else self.model_S
            state_dict = coreModel.state_dict()
            old_state_dict = self.model_S.state_dict
            self.model_S.state_dict = (
                lambda self, *_, **__: get_peft_model_state_dict(
                    self, old_state_dict()
                )
            ).__get__(self.model_S, type(self.model_S))

            self.model_S.state_dict = old_state_dict


            if hasattr(self.model_S.module, "save_pretrained"):
                print("saving at "+self.t_config.output_dir+"/globalstep_"+str(global_step+int(self.global_step_start)))
                self.model_S.module.save_pretrained(self.t_config.output_dir+"/globalstep_"+str(global_step+int(self.global_step_start)))

            if self.local_rank == 0:
                torch.distributed.barrier()
        if callback is not None:
            logger.info("Running callback function...")
            callback(model=self.model_S, step=global_step)
            self.model_S.train()


        if self.has_custom_matches:
            self.model_S._forward_hooks = handles_S  # restore hooks
            self.model_T._forward_hooks = handles_T

    def train_on_batch(self, batch, args):
        
        (teacher_batch, results_T), (student_batch, results_S) = get_outputs_from_batch(batch, self.t_config.device,
                                                                                        self.model_T, self.model_S,
                                                                                        args)

        results_T = post_adaptor(self.adaptor_T(teacher_batch, results_T))
        results_S = post_adaptor(self.adaptor_S(student_batch,  results_S))

        total_loss, losses_dict = self.compute_loss(results_S, results_T, teacher_batch, student_batch)

        return total_loss, losses_dict

    def compute_loss(self, results_S, results_T, teacher_batch, student_batch):
        
        losses_dict = dict()
        
        total_loss = 0
        if 'logits' in results_T and 'logits' in results_S:
            logits_list_T = results_T['logits']  # list of tensor
            logits_list_S = results_S['logits']  # list of tensor

            if self.kd_type == 'original_kd' and self.use_softmax:
                logits_list_T = F.softmax(logits_list_T[0], dim=-1)
                logits_list_S = F.softmax(logits_list_S[0], dim=-1)
            
            total_kd_loss = 0

            if 'logits_mask' in results_S:
                masks_list_S = results_S['logits_mask']
                logits_list_S = select_logits_with_mask(logits_list_S, masks_list_S)  # (mask_sum, num_of_class)
            if 'logits_mask' in results_T:
                masks_list_T = results_T['logits_mask']
                logits_list_T = select_logits_with_mask(logits_list_T, masks_list_T)  # (mask_sum, num_of_class)


            if self.d_config.probability_shift is True:
                labels_list = results_S['labels']
                for l_T, l_S, labels in zip(logits_list_T, logits_list_S, labels_list):
                    l_T = probability_shift_(l_T, labels)
                    # if results_T['logits'][0].shape[-1] != results_S['logits'][0].shape[-1]:

                    for logits_layer in self.logits_projs:
                        l_S = logits_layer(l_S)

                    if self.d_config.temperature_scheduler is not None:
                        temperature = self.d_config.temperature_scheduler(l_S, l_T, self.d_config.temperature)
                    else:
                        temperature = self.d_config.temperature
                    if self.kd_type == 'original_kd':
                        total_kd_loss += self.kd_loss(l_S, l_T, temperature)
                    elif self.kd_type == 'dynamic_kd':
                        total_kd_loss += self.dynamic_kd_loss(l_S, l_T, temperature)
                    elif self.kd_type == 'focal_loss':
                        total_kd_loss += self.focal_loss(l_S, l_T, temperature, teacher_batch["labels"])
                    elif self.kd_type == 'dynamic_temperature':
                        total_kd_loss += self.dynamic_temperature(l_S, l_T, self.normalization_type)
                
            else:
                for l_T, l_S in zip(logits_list_T, logits_list_S):
                    # if results_T['logits'][0].shape[-1] != results_S['logits'][0].shape[-1]:

                    for logits_layer in self.logits_projs:
                        l_S = logits_layer(l_S)

                    if self.d_config.temperature_scheduler is not None:
                        temperature = self.d_config.temperature_scheduler(l_S, l_T, self.d_config.temperature)
                    else:
                        temperature = self.d_config.temperature
                    
                    if self.kd_type == 'original_kd':
                        total_kd_loss += self.kd_loss(l_S, l_T, temperature)
                    elif self.kd_type == 'dynamic_kd':
                        total_kd_loss += self.dynamic_kd_loss(l_S, l_T, temperature)
                    elif self.kd_type == 'focal_loss':
                        total_kd_loss += self.focal_loss(l_S, l_T, temperature, teacher_batch["labels"])
                    elif self.kd_type == 'dynamic_temperature':
                        total_kd_loss += self.dynamic_temperature(l_S, l_T,self.normalization_type)
                        
            total_loss += total_kd_loss * self.d_config.kd_loss_weight
            losses_dict['unweighted_kd_loss'] = total_kd_loss

        inters_T = {feature: results_T.get(feature, []) for feature in FEATURES}
        inters_S = {feature: results_S.get(feature, []) for feature in FEATURES}
        inputs_mask_T = results_T.get('inputs_mask', None)
        inputs_mask_S = results_S.get('inputs_mask', None)
        # pdb.set_trace()
        for ith, inter_match in enumerate(self.d_config.intermediate_matches):
            layer_T = inter_match.layer_T
            layer_S = inter_match.layer_S
            feature = inter_match.feature
            loss_type = inter_match.loss
            match_weight = inter_match.weight
            match_loss = MATCH_LOSS_MAP[loss_type]


            if type(layer_S) is list and type(layer_T) is list:
                inter_S = [inters_S[feature][s] for s in layer_S]
                inter_T = [inters_T[feature][t] for t in layer_T]
                name_S = '-'.join(map(str, layer_S))
                name_T = '-'.join(map(str, layer_T))
                if self.projs[ith]:
                    # inter_T = [self.projs[ith](t) for t in inter_T]
                    inter_S = [self.projs[ith](s) for s in inter_S]
            else:
                inter_S = inters_S[feature][layer_S]
                inter_T = inters_T[feature][layer_T]
                name_S = str(layer_S)
                name_T = str(layer_T)
                if self.projs[ith]:
                    # inter_T = self.projs[ith](inter_T)
                    inter_S = inter_S.float()
                    inter_S = self.projs[ith](inter_S)

            if len(self.intermediate_normalization_type)>0:
                if self.intermediate_normalization_type=='minmax':
                    inter_S = minmax_normalize(inter_S)
                    inter_T = minmax_normalize(inter_T)
                elif self.intermediate_normalization_type=='softmax':
                    inter_S = softmax_normalize(inter_S)
                    inter_T = softmax_normalize(inter_T)
                elif self.intermediate_normalization_type=='standardize':
                    inter_S= standardize_tensor(inter_S)
                    inter_T= standardize_tensor(inter_T)
            
            intermediate_loss = match_loss(inter_S, inter_T, mask=inputs_mask_S)
            total_loss += intermediate_loss * match_weight
            losses_dict[f'unweighted_{feature}_{loss_type}_{name_S}_{name_T}'] = intermediate_loss

        if self.has_custom_matches:
            for hook_T, hook_S, match_weight, match_loss, proj_func in \
                    zip(self.custom_matches_cache['hook_outputs_T'], self.custom_matches_cache['hook_outputs_S'],
                        self.custom_matches_cache['match_weghts'], self.custom_matches_cache['match_losses'],
                        self.custom_matches_cache['match_proj_funcs']):
                if proj_func is not None:
                    hook_S = proj_func(hook_S)
                total_loss += match_weight * match_loss(hook_S, hook_T, inputs_mask_S, inputs_mask_T)
            self.custom_matches_cache['hook_outputs_T'] = []
            self.custom_matches_cache['hook_outputs_S'] = []

        if 'losses' in results_S:
            total_hl_loss = 0
            for loss in results_S['losses']:
                # in case of multi-GPU
                total_hl_loss += loss.mean()
            total_loss += total_hl_loss * self.d_config.hard_label_weight
            losses_dict['unweighted_hard_label_loss'] = total_hl_loss
        formated_losses_dict={}
        for key, value in losses_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            formated_losses_dict[key]=value
        self.logger.info(formated_losses_dict)
        if dist.get_rank() == 0:
            wandb.log(formated_losses_dict)
        print(losses_dict)
        return total_loss, losses_dict

    def add_match(self, match: CustomMatch):
        if type(match.module_T) is str or type(match.module_S) is str:
            raise NotImplementedError
        else:
            module_T = match.module_T
            module_S = match.module_S
            weight = match.weight
            loss = match.loss
            proj_func = match.proj_func
            proj_group = match.proj_group
        self.add_match_by_module(module_T, module_S, proj_func, proj_group, weight, loss)

    def add_match_by_module(self, module_T: torch.nn.Module,
                            module_S: torch.nn.Module,
                            proj_func, proj_group,
                            match_weight, match_loss):

        self.handles_T = module_T.register_forward_hook(self._hook_T)
        self.handles_S = module_S.register_forward_hook(self._hook_S)
        self.custom_matches_cache['match_weights'].append(match_weight)
        self.custom_matches_cache['match_losses'].append(match_loss)
        self.custom_matches_cache['match_proj_funcs'].append(proj_func)
        if isinstance(proj_func, nn.Module):
            self.custom_matches_cache['match_proj_funcs'][-1].to(self.t_config.device)
        self.custom_matches_cache['match_proj_groups'].append(proj_group)

    def _hook_T(self, module, input, output):
        self.custom_matches_cache['hook_outputs_T'].append(output)

    def _hook_S(self, module, input, output):
        self.custom_matches_cache['hook_outputs_S'].append(output)

