import os
from typing import Any, Dict
import torch
from model.blip2_opt import Blip2OPT
from model.blip2_llama import Blip2Llama
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
from valid_result.solve import accuratecy
import torch.distributed as dist
from peft import LoraConfig, TaskType
from model.help_funcs import caption_evaluate, AttrDict


def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    
    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)


# def load_ignore_mismatch(model, state_dict):
#     keys = set(model.state_dict().keys())
#     extra_keys = set()
#     for key in state_dict:
#         if key not in keys:
#             extra_keys.add(key)
#     missing_keys = set()
#     for key in keys:
#         if key not in state_dict:
#             missing_keys.add(key)
#     ## try to print keys that are not included
#     model.load_state_dict(state_dict, strict=False)
    

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict
# peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
class Blip2Stage2(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.llm_tune != 'full':
            to_be_removed = []
            for key in checkpoint['state_dict']:
                if key.startswith('blip2opt.opt_model') or key.startswith('blip2opt.llm_model'):
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint['state_dict'].pop(key)
        if isinstance(self.args.save_every_n_epochs, int) and self.args.save_every_n_epochs > 0:
            if self.llm_tune == 'lora' and (self.current_epoch + 1) % self.args.save_every_n_epochs == 0:
                if self.local_rank == 0: # manually fix a bug in peft module
                    if self.args.peft_config:
                        peft_config = LoraConfig(**LoraConfig.from_json_file(self.args.peft_config))
                    else:
                        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=self.args.lora_r, lora_alpha=self.args.lora_alpha, lora_dropout=self.args.lora_dropout)
                    if hasattr(self.blip2opt, 'opt_model'):
                        self.blip2opt.opt_model.peft_config['default'] = peft_config
                        self.blip2opt.opt_model.save_pretrained(os.path.join("logger.save_dir", f'lora_epoch_{self.current_epoch}'))
                    elif hasattr(self.blip2opt, 'llm_model'):
                        self.blip2opt.llm_model.peft_config['default'] = peft_config
                        self.blip2opt.llm_model.save_pretrained(os.path.join("logger.save_dir'", f'lora_epoch_{self.current_epoch}'))
        return super().on_save_checkpoint(checkpoint)
    
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        if not hasattr(args, 'do_sample'):
            args.do_sample = False
        self.caption_eval_epoch = args.caption_eval_epoch
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_len = args.max_len
        self.min_len = args.min_len
        self.reaction_weight = args.reaction_weight
        self.llm_tune = args.llm_tune
        if args.opt_model.find('galactica') >= 0:
            self.blip2opt = Blip2OPT(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune, args.peft_dir, args.opt_model, args.prompt, args)
        elif args.opt_model.find('llama') >= 0 or args.opt_model.find('vicuna') >= 0:
            self.blip2opt = Blip2Llama(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune, args.peft_dir, args.opt_model, args.prompt, args)
        else:
            raise NotImplementedError()
        self.tokenizer = self.blip2opt.init_tokenizer()
        self.save_hyperparameters(args)

    def load_from_stage1_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['state_dict']
        graph_encoder_dict = get_module_state_dict(state_dict, 'blip2qformer.graph_encoder')
        qformer_dict = get_module_state_dict(state_dict, 'blip2qformer.Qformer')
        ln_graph_dict = get_module_state_dict(state_dict, 'blip2qformer.ln_graph')
        qs_weight = get_module_state_dict(state_dict, 'blip2qformer.query_tokens')
        load_ignore_unexpected(self.blip2opt.Qformer, qformer_dict)
        self.blip2opt.graph_encoder.load_state_dict(graph_encoder_dict)
        self.blip2opt.ln_graph.load_state_dict(ln_graph_dict)
        self.blip2opt.query_tokens.data.copy_(qs_weight)
        return self
    
    # def load_from_stage1_checkpoint(self, path):
    #     ckpt = torch.load(path, map_location='cpu')
    #     state_dict = ckpt['state_dict']
    #     state_dict = {k[13:]: v for k,v in state_dict.items()}
    #     load_ignore_mismatch(self.blip2opt, state_dict)
    #     return self
    
    def configure_optimizers(self):
        self.trainer.reset_train_dataloader()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def test_epoch_end(self, outputs):
        print("Entering test_epoch_end")
        
        list_predictions, list_targets = zip(*outputs)
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]
        print("After extracting predictions and targets")
        
        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]

        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_targets, targets)
        print("After all_gather_object")

        if self.global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            print("After processing predictions and targets")

            self.save_predictions_test(all_predictions, all_targets)
            print("After saving predictions")

        print("Exiting test_epoch_end")

    def save_predictions_valid(self, predictions, targets):
        assert len(predictions) == len(targets)
        print("----------------")
        print(len(predictions))
        print("----------------")
        #with open(os.path.join(self.logger.log_dir, 'predictions.txt'), 'w', encoding='utf8') as f:
        with open(os.path.join('valid_result/', 'predictions.txt'), 'w', encoding='utf8') as f:
            for p, t in zip(predictions, targets):
                line = {'prediction': p, 'target': t}
                f.write(json.dumps(line, ensure_ascii=True) + '\n')
        if len(predictions)!=40:
            accuratecy()
    def save_predictions_test(self, predictions, targets):
        assert len(predictions) == len(targets)
        #with open(os.path.join(self.logger.log_dir, 'predictions.txt'), 'w', encoding='utf8') as f:
        with open(os.path.join('test_result/', 'predictions.txt'), 'w', encoding='utf8') as f:
            for p, t in zip(predictions, targets):
                line = {'prediction': p, 'target': t}
                f.write(json.dumps(line, ensure_ascii=True) + '\n')

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        graphs1,graphs2, prompt_tokens, texts = batch
        ###============== Captioning Results ===================###
        samples = {'graphs1': graphs1, 'graphs2': graphs2,'prompt_tokens': prompt_tokens}
        predictions = self.blip2opt.generate(
            samples, 
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len
        )
        return predictions, texts

    # @torch.no_grad()
    # def validation_step(self, batch, batch_idx, dataloader_idx=0):
    #     if dataloader_idx == 0:
    #         _, _, prompt_lens = batch
    #         batch_size = prompt_lens.shape[0]
    #         loss = self.blip2opt(batch)
    #         ###============== Overall Loss ===================###
    #         self.log("val molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
    #         return loss['loss']
    #     elif dataloader_idx == 1:
    #         reaction_tokens, _, _ = batch
    #         batch_size = reaction_tokens.input_ids.shape[0]
    #         loss = self.blip2opt.forward_reaction(batch)
    #         ###============== Overall Loss ===================###
    #         self.log("val reaction loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
    #         return loss['loss']
    #     else:
    #         raise NotImplementedError
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        if self.args.DDI == True:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            graphs1,graphs2, prompt_tokens, texts = batch
            ###============== Captioning Results ===================###
            samples = {'graphs1': graphs1, 'graphs2': graphs2,'prompt_tokens': prompt_tokens}
            predictions = self.blip2opt.generate(
                samples, 
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len
            )
            return predictions, texts
        elif self.args.solve == True:
            if dataloader_idx == 0:
                _, _, _,text_tokens = batch
                text_tokens=text_tokens
                batch_size = text_tokens.input_ids.shape[0]
                loss = self.blip2opt(batch)
                ###============== Overall Loss ===================###
                self.log("val molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
                return loss['loss']
            elif dataloader_idx == 1:
                if self.current_epoch != 0:
                    if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                        return 
                    graphs1,graphs2, prompt_tokens, texts = batch
                    ###============== Captioning Results ===================###
                    samples = {'graphs1': graphs1, 'graphs2': graphs2,'prompt_tokens': prompt_tokens}
                    predictions = self.blip2opt.generate(
                        samples, 
                        do_sample=self.do_sample,
                        num_beams=self.num_beams,
                        max_length=self.max_len,
                        min_length=self.min_len
                    )
                    return predictions, texts
                '''
                else:
                    graphs1,graphs2, prompt_tokens, texts = batch
                    ###============== Captioning Results ===================###
                    samples = {'graphs1': graphs1, 'graphs2': graphs2,'prompt_tokens': prompt_tokens}
                    predictions = self.blip2opt.generate(
                        samples, 
                        do_sample=self.do_sample,
                        num_beams=self.num_beams,
                        max_length=self.max_len,
                        min_length=self.min_len
                    )
                    return predictions, texts
                '''
            elif dataloader_idx == 2:
                reaction_tokens, _, _ = batch
                batch_size = reaction_tokens.input_ids.shape[0]
                loss = self.blip2opt.forward_reaction(batch)
                ###============== Overall Loss ===================###
                self.log("val reaction loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
                return loss['loss']
            else:
                raise NotImplementedError
        elif self.args.fangguangtuan == True:
            if dataloader_idx == 0:
                _, _, _,text_tokens = batch
                text_tokens=text_tokens
                batch_size = text_tokens.input_ids.shape[0]
                loss = self.blip2opt(batch)
                ###============== Overall Loss ===================###
                self.log("val molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
                return loss['loss']
            elif dataloader_idx == 1:
                if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                    return 
                graphs1,graphs2, prompt_tokens, texts = batch
                ###============== Captioning Results ===================###
                samples = {'graphs1': graphs1, 'graphs2': graphs2,'prompt_tokens': prompt_tokens}
                predictions = self.blip2opt.generate(
                    samples, 
                    do_sample=self.do_sample,
                    num_beams=self.num_beams,
                    max_length=self.max_len,
                    min_length=self.min_len
                )
                return predictions, texts
            elif dataloader_idx == 2:
                reaction_tokens, _, _ = batch
                batch_size = reaction_tokens.input_ids.shape[0]
                loss = self.blip2opt.forward_reaction(batch)
                ###============== Overall Loss ===================###
                self.log("val reaction loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
                return loss['loss']
            else:
                raise NotImplementedError
        else :
            if dataloader_idx == 0:
                _, _,text_tokens = batch
                text_tokens=text_tokens
                batch_size = text_tokens.input_ids.shape[0]
                loss = self.blip2opt(batch)
                ###============== Overall Loss ===================###
                self.log("val molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
                return loss['loss']
            elif dataloader_idx == 1:
                if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                    return 
                graphs,prompt_tokens, texts = batch
                ###============== Captioning Results ===================###
                samples = {'graphs': graphs, 'prompt_tokens': prompt_tokens}
                predictions = self.blip2opt.generate(
                    samples, 
                    do_sample=self.do_sample,
                    num_beams=self.num_beams,
                    max_length=self.max_len,
                    min_length=self.min_len
                )
                return predictions, texts
            elif dataloader_idx == 2:
                reaction_tokens, _, _ = batch
                batch_size = reaction_tokens.input_ids.shape[0]
                loss = self.blip2opt.forward_reaction(batch)
                ###============== Overall Loss ===================###
                self.log("val reaction loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
                return loss['loss']
            else:
                raise NotImplementedError

    def validation_epoch_end(self, outputs):
        if self.current_epoch != 0:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            caption_outputs = outputs[1]
            list_predictions, list_targets = zip(*caption_outputs)
            predictions = [i for ii in list_predictions for i in ii]
            targets = [i for ii in list_targets for i in ii]
            all_predictions = [None for _ in range(self.trainer.world_size)]
            all_targets = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
            if self.global_rank == 0:
                all_predictions = [i for ii in all_predictions for i in ii]
                all_targets = [i for ii in all_targets for i in ii]
                self.save_predictions_valid(all_predictions, all_targets)
        '''
        else:
            caption_outputs = outputs[1]
            list_predictions, list_targets = zip(*caption_outputs)
            predictions = [i for ii in list_predictions for i in ii]
            targets = [i for ii in list_targets for i in ii]
            all_predictions = [None for _ in range(self.trainer.world_size)]
            all_targets = [None for _ in range(self.trainer.world_size)]
            dist.all_gather_object(all_predictions, predictions)
            dist.all_gather_object(all_targets, targets)
            if self.global_rank == 0:
                all_predictions = [i for ii in all_predictions for i in ii]
                all_targets = [i for ii in all_targets for i in ii]
                self.save_predictions_valid(all_predictions, all_targets)
            ## fixme: I am not sure if the max length is the same as previous experiments
        '''

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        if isinstance(batch, list) and len(batch) == 2:
            molecule_batch, reaction_batch = batch
            batch_size = molecule_batch[-1].size(0)
            ###============== molecule Loss ===================###
            molecule_loss = self.blip2opt(molecule_batch)['loss']
            self.log("molecule loss", float(molecule_loss), batch_size=batch_size, sync_dist=True)
            
            ###============== reaction Loss ===================###
            reaction_loss = self.blip2opt.forward_reaction(reaction_batch)['loss']
            self.log("reaction loss", float(reaction_loss), batch_size=batch_size, sync_dist=True)

            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
            return molecule_loss + self.reaction_weight * reaction_loss
        else:
            batch_size = batch[-1].input_ids.size(0)
            ###============== Overall Loss ===================###
            loss = self.blip2opt(batch)
            self.log("molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
            return loss['loss']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--tune_gnn', action='store_true', default=False)
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_name', type=str, default='scibert')
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        # OPT
        parser.add_argument('--opt_model', type=str, default="facebook/galactica-1.3b")
        # parser.add_argument('--prompt', type=str, default='a molecule of ')
        parser.add_argument('--num_beams', type=int, default=5)
        parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--max_len', type=int, default=256)
        parser.add_argument('--min_len', type=int, default=8)
        parser.add_argument('--llm_tune', type=str, default='freeze')
        parser.add_argument('--peft_config', type=str, default=None)
        parser.add_argument('--peft_dir', type=str, default='')

        parser.add_argument('--save_every_n_epochs', type=int, default=0)
        ## quantization
        parser.add_argument('--load_in_8bit', action='store_true', default=False)

        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)

        # optimization
        parser.add_argument('--reaction_weight', type=float, default=1.0)
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        parser.add_argument('--stage1_path', type=str, default='')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=10)
        return parent_parser


