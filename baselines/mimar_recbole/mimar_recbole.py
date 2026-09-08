# -*- coding: utf-8 -*-
"""RecBole/EffiPOI-integrated MIMAR-style baseline.

This model deliberately keeps the SASRec data path, prediction head, Trainer,
Evaluator, and full-sort metrics unchanged. It only adds multi-granularity
intent representations before the final next-POI scoring layer.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec


def _parse_windows(value):
    if value is None:
        return [5, 20, 50]
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


class MIMARRecBole(SASRec):
    """SASRec backbone with multi-granularity intent residual fusion.

    The important part for a fair baseline is that `calculate_loss` and
    `full_sort_predict` use RecBole's original interaction fields and full-sort
    evaluation. The intent module enriches the final sequence representation
    instead of replacing the standard next-POI prediction head.
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.intent_windows = _parse_windows(config["intent_windows"] if "intent_windows" in config else None)
        self.num_intents = int(config["num_intents"]) if "num_intents" in config else 4
        self.intent_residual_weight = (
            float(config["intent_residual_weight"]) if "intent_residual_weight" in config else 1.0
        )
        self.contrastive_temp = float(config["contrastive_temp"]) if "contrastive_temp" in config else 0.2
        self.proto_weight = float(config["proto_weight"]) if "proto_weight" in config else 0.0
        self.seqcl_weight = float(config["seqcl_weight"]) if "seqcl_weight" in config else 0.0
        self.adv_weight = float(config["adv_weight"]) if "adv_weight" in config else 0.0
        self.adv_eps = float(config["adv_eps"]) if "adv_eps" in config else 0.03

        self.intent_queries = nn.Parameter(torch.empty(self.num_intents, self.hidden_size))
        nn.init.xavier_uniform_(self.intent_queries)

        gate_inputs = self.hidden_size * (len(self.intent_windows) + 1)
        self.granularity_gate = nn.Sequential(
            nn.Linear(gate_inputs, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.hidden_dropout_prob),
            nn.Linear(self.hidden_size, len(self.intent_windows) + 1),
        )

        self.intent_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.intent_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    def _encode_sequence_states(self, item_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        input_emb = self.item_embedding(item_seq) + self.position_embedding(position_ids)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb,
            extended_attention_mask,
            output_all_encoded_layers=True,
        )
        return trm_output[-1]

    def _window_mask(self, item_seq, item_seq_len, window_size):
        seq_len = item_seq.size(1)
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)
        end = item_seq_len.unsqueeze(1)
        start = (item_seq_len - window_size).clamp_min(0).unsqueeze(1)
        return item_seq.gt(0) & positions.ge(start) & positions.lt(end)

    def _pool_window_intent(self, sequence_states, item_seq, item_seq_len, window_size, query):
        mask = self._window_mask(item_seq, item_seq_len, window_size)
        logits = torch.einsum("blh,kh->blk", sequence_states, self.intent_queries)
        logits = logits / math.sqrt(self.hidden_size)
        logits = logits.masked_fill(~mask.unsqueeze(-1), -1e9)

        weights = torch.softmax(logits, dim=1)
        weights = weights.masked_fill(~mask.unsqueeze(-1), 0.0)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        intent_vectors = torch.einsum("blk,blh->bkh", weights / denom, sequence_states)

        intent_scores = torch.einsum("bkh,bh->bk", intent_vectors, query)
        intent_scores = intent_scores / math.sqrt(self.hidden_size)
        intent_alpha = torch.softmax(intent_scores, dim=-1)
        return torch.einsum("bk,bkh->bh", intent_alpha, intent_vectors)

    def _multi_granularity_intent(self, sequence_states, item_seq, item_seq_len, base_output):
        window_outputs = [
            self._pool_window_intent(sequence_states, item_seq, item_seq_len, window, base_output)
            for window in self.intent_windows
        ]
        candidates = [base_output] + window_outputs
        gate_logits = self.granularity_gate(torch.cat(candidates, dim=-1))
        gate = torch.softmax(gate_logits, dim=-1)
        stacked = torch.stack(candidates, dim=1)
        return torch.einsum("bg,bgh->bh", gate, stacked)

    def forward(self, item_seq, item_seq_len):
        sequence_states = self._encode_sequence_states(item_seq)
        base_output = self.gather_indexes(sequence_states, item_seq_len - 1)
        intent_output = self._multi_granularity_intent(
            sequence_states,
            item_seq,
            item_seq_len,
            base_output,
        )
        fused_delta = self.intent_fusion(torch.cat([base_output, intent_output], dim=-1))
        return self.intent_norm(base_output + self.intent_residual_weight * fused_delta)

    def _ce_loss(self, seq_output, pos_items):
        logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
        return self.loss_fct(logits, pos_items), logits

    def _prototype_loss(self, seq_output):
        if self.proto_weight <= 0:
            return seq_output.new_tensor(0.0)
        seq_norm = F.normalize(seq_output, dim=-1)
        proto_norm = F.normalize(self.intent_queries, dim=-1)
        logits = torch.matmul(seq_norm, proto_norm.transpose(0, 1)) / self.contrastive_temp
        # Confident but balanced assignment regularization.
        assign = torch.softmax(logits, dim=-1)
        entropy = -(assign * torch.log(assign.clamp_min(1e-12))).sum(dim=-1).mean()
        return entropy

    def _seq_contrastive_loss(self, item_seq, item_seq_len, seq_output):
        if self.seqcl_weight <= 0 or item_seq.size(0) < 2:
            return seq_output.new_tensor(0.0)
        seq_output_2 = self.forward(item_seq, item_seq_len)
        z1 = F.normalize(seq_output, dim=-1)
        z2 = F.normalize(seq_output_2, dim=-1)
        logits = torch.matmul(z1, z2.transpose(0, 1)) / self.contrastive_temp
        labels = torch.arange(item_seq.size(0), device=item_seq.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))

    def _adv_loss(self, seq_output, clean_logits):
        if self.adv_weight <= 0:
            return seq_output.new_tensor(0.0)
        noise = torch.randn_like(seq_output)
        noise = self.adv_eps * F.normalize(noise, dim=-1)
        adv_output = seq_output + noise
        adv_logits = torch.matmul(adv_output, self.item_embedding.weight.transpose(0, 1))
        return F.kl_div(
            F.log_softmax(adv_logits, dim=-1),
            F.softmax(clean_logits.detach(), dim=-1),
            reduction="batchmean",
        )

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)
        ce_loss, logits = self._ce_loss(seq_output, pos_items)

        loss = ce_loss
        loss = loss + self.proto_weight * self._prototype_loss(seq_output)
        loss = loss + self.seqcl_weight * self._seq_contrastive_loss(item_seq, item_seq_len, seq_output)
        loss = loss + self.adv_weight * self._adv_loss(seq_output, logits)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        return torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
