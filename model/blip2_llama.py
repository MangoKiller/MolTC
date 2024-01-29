"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel

from lavis.models.blip2_models.blip2 import (
    # Blip2Base,
    disabled_train,
)
from model.blip2 import Blip2Base
from transformers import LlamaTokenizer
from model.modeling_llama import LlamaForCausalLM


 
llama_model_list = [
    "decapoda-research/llama-13b-hf",
    "decapoda-research/llama-7b-hf",
]

def mask_by_len(input, lens, fill_value=0):
    '''
    input: shape = [N, D]
    lens: shape = [N]
    '''
    mask = torch.arange(input.shape[1], device=input.device).reshape(1, -1)
    mask = mask < lens.reshape(-1, 1)
    input[mask] = fill_value
    return input

# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2Llama(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    def __init__(
        self,
        bert_name,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        lora_tuning=False,
        peft_dir='',
        llm_model="decapoda-research/llama-7b-hf",
        prompt="",
        args=None,
    ):
        super().__init__()
        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
        ### remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        ## initialize opt model
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(llm_model, use_fast=False, padding_side='right')
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_model = LlamaForCausalLM.from_pretrained(llm_model, torch_dtype=torch.bfloat16)
        # self.llm_model = LlamaForCausalLM.from_pretrained(llm_model)
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))
        
        self.lora_tuning = lora_tuning
        if lora_tuning:
            if peft_dir:
                self.llm_model = PeftModel.from_pretrained(self.llm_model, peft_dir, is_trainable=True)
            else:
                peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
                self.llm_model = get_peft_model(self.llm_model, peft_config)
                self.llm_model.print_trainable_parameters()
        else:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

        ## fixme: this is different from the original BLIP2
        self.eos_token_id = self.llm_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]
        self.pad_token_id = self.llm_tokenizer.pad_token_id

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )
        
        ## fixme: no prompt yet
        self.prompt = prompt
        # prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def forward(self, batch):
        graphs, text_tokens, prompt_lens = batch
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        if not self.tune_gnn:
            graph_embeds = graph_embeds.detach()
        graph_embeds = self.ln_graph(graph_embeds, graph_masks)
        device = graph_embeds.device
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
            return_dict=True,
        )
        inputs_llm = self.llm_proj(query_output.last_hidden_state)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(device)
        targets = text_tokens.input_ids.masked_fill(
            text_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets = mask_by_len(targets, prompt_lens, -100) # do not apply loss to the prompt
            # targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt
        
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        # if self.lora_tuning:
        #     inputs_embeds = self.llm_model.model.get_decoder().embed_tokens(text_tokens.input_ids)
        # else:
        #     inputs_embeds = self.llm_model.model.decoder.embed_tokens(text_tokens.input_ids)
        inputs_embeds = self.llm_model.get_input_embeddings()(text_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, text_tokens.attention_mask], dim=1)

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            # use_cache=False,
        )
        loss = outputs.loss
        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        graphs = samples['graphs']
        prompt_tokens = samples['prompt_tokens']
        # prompt_lens = samples['prompt_lens']
        with self.maybe_autocast():
            graph_embeds, graph_masks = self.graph_encoder(graphs)
            graph_embeds = self.ln_graph(graph_embeds)

            query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=graph_embeds,
                encoder_attention_mask=graph_masks,
                return_dict=True,
            )

            device = graph_embeds.device
            inputs_llm = self.llm_proj(query_output.last_hidden_state)
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long, device=device)

            attention_mask = torch.cat([atts_llm, prompt_tokens.attention_mask], dim=1)
            
            if False:
                if do_sample:
                    query_embeds = inputs_llm.repeat_interleave(num_captions, dim=0)
                    num_beams = 1
                else:
                    query_embeds = inputs_llm.repeat_interleave(num_beams, dim=0)

                outputs = self.llm_model.generate(
                    input_ids=prompt_tokens.input_ids,
                    query_embeds=query_embeds,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_length=min_length,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )

                prompt_length = prompt_tokens.input_ids.shape[1]
                output_text = self.opt_tokenizer.batch_decode(
                    outputs[:, prompt_length:], skip_special_tokens=True
                )
            else:
                inputs_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids)
                inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                attention_mask = torch.cat([atts_llm, prompt_tokens.attention_mask], dim=1)

                outputs = self.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    # use_cache=False,
                )
                # outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
                output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]
            return output_text