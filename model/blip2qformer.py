"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

# from lavis.common.registry import registry
# from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput
from lavis.common.dist_utils import is_dist_avail_and_initialized
from model.blip2 import Blip2Base
from pytorch_lightning.utilities import distributed

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    print('running here')
    return output

@torch.no_grad()
def pl_concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # if use distributed training
    if not is_dist_avail_and_initialized():
        return tensor

    tensors_gather = distributed.gather_all_tensors(tensor)
    output = torch.cat(tensors_gather, dim=0)
    return output


# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
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
        gtm,
        lm,
        bert_name,
        temperature,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
    ):
        super().__init__()
        self.gtm = gtm
        self.lm = lm
        
        self.tokenizer = self.init_tokenizer()

        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq)
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.graph_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.gtm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temperature = temperature

    
    def contrast(self, features_graph, features_text, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        '''
        batch_size = features_graph.size(0)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B]

        logits_per_graph = sim_g2t / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)  # 大小为B
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph, logits_per_text, loss
        else:
            return loss

    def contrast_global(self, features_graph, features_text, features_graph_all, features_text_all, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        features_text_all: shape = [B * num_gpus, D]
        features_graph_all: shape = [B * num_gpus, num_qs, D]
        '''
        bs = features_graph.size(0)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text_all.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B * num_gpus, D, 1]; output shape = [B, B * num_gpus, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B * num_gpus]

        logits_per_graph = sim_g2t / self.temperature
    

        sim_t2q = (features_text.unsqueeze(1).unsqueeze(1) @ features_graph_all.permute(0, 2, 1)).squeeze() # shape = [B, 1, 1, D]; [B*num_gpus, D, num_qs]; output shape = [B, B*num_gpus, 1, num_qs]
        sim_t2g, _ = sim_t2q.max(-1)
        logits_per_text = sim_t2g / self.temperature

        # labels = torch.arange(bs, dtype=torch.long, device=self.device)
        rank = dist.get_rank()
        labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph[:, rank*bs:rank*bs+bs], logits_per_text[:, rank*bs:rank*bs+bs], loss
        else:
            return loss
        
    def forward_old(self, batch):
        ## v1: not gather results from all gpus
        ###============== Image-text Contrastive ===================###
        graph, text, mask = batch
        batch_node, batch_mask = self.graph_encoder(graph)
        batch_node = batch_node.detach()
        batch_size = batch_node.shape[0]

        batch_node = self.ln_graph(batch_node, batch_mask)
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            use_cache=True,
            return_dict=True,
        )
        graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
        sim_g2t, sim_t2g, loss_gtc = self.contrast(graph_feats, text_feats, return_sim=True)


        ###============== Image-text Matching ===================###
        loss_gtm = 0
        if self.gtm:
            g_emb = batch_node
            g_mask = batch_mask
            text_ids = text.clone()
            with torch.no_grad():
                weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4
                weights_t2g.fill_diagonal_(0)
                weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
                weights_g2t.fill_diagonal_(0)

            # select a negative graph for each text
            graph_embeds_neg = []
            graph_mask_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                graph_embeds_neg.append(g_emb[neg_idx])
                graph_mask_neg.append(g_mask[neg_idx])
            
            graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)
            graph_mask_neg = torch.stack(graph_mask_neg, dim=0)

            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                text_ids_neg.append(text_ids[neg_idx])
                text_atts_neg.append(mask[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat(
                [text_ids, text_ids, text_ids_neg], dim=0
            )  # pos, pos, neg
            text_atts_all = torch.cat(
                [mask, mask, text_atts_neg],
                dim=0,
            )

            query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long, device=text.device)
            attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

            graph_embeds_all = torch.cat([g_emb, graph_embeds_neg, g_emb], dim=0)  # pos, neg, pos
            graph_atts_all = torch.cat([g_mask, graph_mask_neg, g_mask], dim=0)

            output_itm = self.Qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=graph_embeds_all,
                encoder_attention_mask=graph_atts_all,
                return_dict=True,
            )

            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :] # keep query tokens only
            vl_output = self.gtm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            itm_labels = torch.cat(
                [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                dim=0,
            ).to(text.device)
            loss_gtm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        loss_lm = 0
        if self.lm:
            decoder_input_ids = text.clone()
            decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
            labels = decoder_input_ids.masked_fill(
                decoder_input_ids == self.tokenizer.pad_token_id, -100
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=text.device)
            
            attention_mask = torch.cat([query_atts, mask], dim=1)
            lm_output = self.Qformer(
                decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=query_output.past_key_values,
                return_dict=True,
                labels=labels,
            )

            loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_gtc + loss_gtm + loss_lm,
            loss_itc=loss_gtc,
            loss_itm=loss_gtm,
            loss_lm=loss_lm,
        )


    def forward(self, batch):
        ## v2: gather results from all gpus
        ###============== Image-text Contrastive ===================###
        #这里输入的就是正常的图文配对
        graph, text, mask = batch
        batch_node, batch_mask = self.graph_encoder(graph)
        if not self.tune_gnn:
            batch_node = batch_node.detach()
        batch_size = batch_node.shape[0]

        batch_node = self.ln_graph(batch_node, batch_mask)
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            use_cache=True,
            return_dict=True,
        )
        graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
        
        text_feats, graph_feats = F.normalize(text_feats, p=2, dim=-1), F.normalize(graph_feats, p=2, dim=-1)
        text_feats_all, graph_feats_all = pl_concat_all_gather(text_feats), pl_concat_all_gather(graph_feats) # shape = [B * num_gpus, D]
        sim_g2t, sim_t2g, loss_gtc = self.contrast_global(graph_feats, text_feats, graph_feats_all, text_feats_all, return_sim=True)


        ###============== Image-text Matching ===================###
        loss_gtm = 0
        if self.gtm:
            ## not aggregate global tensor because of their different shapes
            g_emb_world = batch_node
            g_mask_world = batch_mask
            text_ids_world = text
            text_mask_world = mask
            with torch.no_grad():
                weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4
                weights_t2g.fill_diagonal_(0)
                weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
                weights_g2t.fill_diagonal_(0)

            # select a negative graph for each text
            graph_embeds_neg = []
            graph_mask_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                graph_embeds_neg.append(g_emb_world[neg_idx])
                graph_mask_neg.append(g_mask_world[neg_idx])
            
            graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)
            graph_mask_neg = torch.stack(graph_mask_neg, dim=0)

            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                text_ids_neg.append(text_ids_world[neg_idx])
                text_atts_neg.append(text_mask_world[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat(
                [text, text, text_ids_neg], dim=0
            )  # pos, pos, neg
            text_atts_all = torch.cat(
                [mask, mask, text_atts_neg],
                dim=0,
            )

            query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long, device=text.device)
            attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

            graph_embeds_all = torch.cat([batch_node, graph_embeds_neg, batch_node], dim=0)  # pos, neg, pos
            graph_atts_all = torch.cat([batch_mask, graph_mask_neg, batch_mask], dim=0)

            output_itm = self.Qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=graph_embeds_all,
                encoder_attention_mask=graph_atts_all,
                return_dict=True,
            )

            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :] # keep query tokens only
            vl_output = self.gtm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            itm_labels = torch.cat(
                [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                dim=0,
            ).to(text.device)
            loss_gtm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        loss_lm = 0
        if self.lm:
            decoder_input_ids = text.clone()
            decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
            labels = decoder_input_ids.masked_fill(
                decoder_input_ids == self.tokenizer.pad_token_id, -100
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=text.device)
            
            attention_mask = torch.cat([query_atts, mask], dim=1)
            lm_output = self.Qformer(
                decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=query_output.past_key_values,
                return_dict=True,
                labels=labels,
            )

            loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_gtc + loss_gtm + loss_lm,
            loss_itc=loss_gtc,
            loss_itm=loss_gtm,
            loss_lm=loss_lm,
        )
    
    def forward_v3(self, batch):
        ## v3: use smiles instruction
        ###============== Image-text Contrastive ===================###
        graphs, text_tokens, prompt_tokens = batch
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        if not self.tune_gnn:
            graph_embeds = graph_embeds.detach()
        graph_embeds = self.ln_graph(graph_embeds, graph_masks)

        device = text_tokens.input_ids.device
        batch_size = graph_embeds.shape[0]
        
        ## 
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)
        attention_mask_gtc = torch.cat([query_atts, prompt_tokens.attention_mask], dim=1)
        query_output = self.Qformer.bert(
            input_ids=prompt_tokens,
            query_embeds=query_tokens,
            attention_mask=attention_mask_gtc,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
            use_cache=True,
            return_dict=True,
        )

        query_output = query_output.last_hidden_state[:, : query_tokens.size(1), :] # keep query tokens only
        graph_feats = self.graph_proj(query_output) # shape = [B, num_q, D]
        text_output = self.Qformer.bert(text_tokens.input_ids, attention_mask=text_tokens.attention_mask, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
        
        text_feats, graph_feats = F.normalize(text_feats, p=2, dim=-1), F.normalize(graph_feats, p=2, dim=-1)
        text_feats_all, graph_feats_all = pl_concat_all_gather(text_feats), pl_concat_all_gather(graph_feats) # shape = [B * num_gpus, D]
        sim_g2t, sim_t2g, loss_gtc = self.contrast_global(graph_feats, text_feats, graph_feats_all, text_feats_all, return_sim=True)


        ###============== Image-text Matching ===================###
        loss_gtm = 0
        if self.gtm:
            ## not aggregate global tensor because of their different shapes
            g_emb_world = graph_embeds
            g_mask_world = graph_masks
            text_ids_world = text_tokens.input_ids
            text_mask_world = text_tokens.attention_mask
            with torch.no_grad():
                weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4
                weights_t2g.fill_diagonal_(0)
                weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
                weights_g2t.fill_diagonal_(0)

            # select a negative graph for each text
            graph_embeds_neg = []
            graph_mask_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                graph_embeds_neg.append(g_emb_world[neg_idx])
                graph_mask_neg.append(g_mask_world[neg_idx])
            
            graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)
            graph_mask_neg = torch.stack(graph_mask_neg, dim=0)

            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                text_ids_neg.append(text_ids_world[neg_idx])
                text_atts_neg.append(text_mask_world[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat(
                [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
            )  # pos, pos, neg
            text_atts_all = torch.cat(
                [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
                dim=0,
            )

            query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long, device=text_tokens.input_ids.device)
            attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

            graph_embeds_all = torch.cat([graph_embeds, graph_embeds_neg, graph_embeds], dim=0)  # pos, neg, pos
            graph_atts_all = torch.cat([graph_masks, graph_mask_neg, graph_masks], dim=0)

            output_itm = self.Qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=graph_embeds_all,
                encoder_attention_mask=graph_atts_all,
                return_dict=True,
            )

            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :] # keep query tokens only
            vl_output = self.gtm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            itm_labels = torch.cat(
                [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                dim=0,
            ).to(text_tokens.input_ids.device)
            loss_gtm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        loss_lm = 0
        if self.lm:
            decoder_input_ids = text_tokens.input_ids.clone()
            decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
            labels = decoder_input_ids.masked_fill(
                decoder_input_ids == self.tokenizer.pad_token_id, -100
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=text_tokens.input_ids.device)
            
            attention_mask = torch.cat([query_atts, prompt_tokens.attention_mask, text_tokens.attention_mask], dim=1)
            lm_output = self.Qformer(
                decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=query_output.past_key_values,
                return_dict=True,
                labels=labels,
            )

            loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_gtc + loss_gtm + loss_lm,
            loss_itc=loss_gtc,
            loss_itm=loss_gtm,
            loss_lm=loss_lm,
        )
    
    def graph_forward(self, graph):
        batch_node, batch_mask = self.graph_encoder(graph)
        batch_node = self.ln_graph(batch_node, batch_mask)
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            use_cache=False,
            return_dict=True,
        )
        graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        graph_feats = F.normalize(graph_feats, p=2, dim=-1)
        return graph_feats, batch_node, batch_mask

    def text_forward(self, text, mask):
        text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :] )
        text_feats = F.normalize(text_feats, dim=-1, p=2)
        return text_feats
    
    def compute_gtm(self, batch_node, batch_mask, text_ids, text_atts):
        '''
        batch_node shape = [B, N, D]
        batch_mask shape = [B, N]
        text_ids shape = [B, N]
        text_atts shape = [B, N]
        '''
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1) # shape = [B, Nq, D]
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            batch_node.device
        ) # shape = [B, Nq]
        attention_mask = torch.cat([query_atts, text_atts], dim=1) # shape = [B, Nq + N]
        output_gtm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask,
            return_dict=True,
        )
        gl_embeddings = output_gtm.last_hidden_state[:, : query_tokens.size(1), :] # shape = [B, Nq, D]
        gtm_logit = self.gtm_head(gl_embeddings).mean(dim=1) # shape = [B, Nq, 2]
        # gtm_logit = F.softmax(gtm_logit, dim=-1)[:, 1] # select the axis of the positive class
        gtm_logit = gtm_logit[:, 1] # select the axis of the positive class
        return gtm_logit

