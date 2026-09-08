# -*- coding: utf-8 -*-
"""Protocol-aligned S2HyRec baseline for the EffiPOI RecBole pipeline.

The implementation preserves the four mechanisms of S2HyRec while keeping
the benchmark split, SASRec prediction head, and RecBole full-sort evaluator
identical to the other formal baselines:

1. learnable hypergraph prototypes model global intent tendency;
2. ordered sequence segments model temporal contextual intent;
3. a causal Transformer models sequence dependency;
4. cross-view contrastive learning aligns global and temporal intent.

The local benchmark files contain ordered POI sequences but no absolute
timestamps. Temporal contexts are therefore defined by relative chronological
segments instead of calendar-time buckets.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec


class S2HyRecRecBole(SASRec):
    """S2HyRec-style sequential recommender under a fixed RecBole protocol."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.hyper_num = int(config["hyper_num"])
        self.time_slices = int(config["time_slices"])
        self.hyper_temperature = float(config["hyper_temperature"])
        self.global_intent_weight = float(config["global_intent_weight"])
        self.intent_residual_weight = float(config["intent_residual_weight"])
        self.ssl_weight = float(config["ssl_weight"])

        self.hyperedge_embeddings = nn.Parameter(
            torch.empty(self.hyper_num, self.hidden_size)
        )
        nn.init.xavier_uniform_(self.hyperedge_embeddings)

        self.hyper_mapper = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.temporal_mapper = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.intent_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.intent_norm = nn.LayerNorm(
            self.hidden_size,
            eps=self.layer_norm_eps,
        )

        self.apply(self._init_weights)

    def _encode_sequence_states(self, item_seq):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        input_emb = self.item_embedding(item_seq) + self.position_embedding(position_ids)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        attention_mask = self.get_attention_mask(item_seq)
        transformer_output = self.trm_encoder(
            input_emb,
            attention_mask,
            output_all_encoded_layers=True,
        )
        return transformer_output[-1]

    def _masked_attention_pool(self, values, query, mask):
        scores = torch.einsum("blh,bh->bl", values, query)
        scores = scores / math.sqrt(self.hidden_size)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = weights.masked_fill(~mask, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.einsum("bl,blh->bh", weights, values)

    def _global_hypergraph_intent(self, item_seq, base_output):
        """Map visited POIs to shared hyperedges and aggregate global intent."""
        sequence_items = self.item_embedding(item_seq)
        item_norm = F.normalize(sequence_items, dim=-1)
        edge_norm = F.normalize(self.hyperedge_embeddings, dim=-1)
        assignment_logits = torch.einsum("blh,mh->blm", item_norm, edge_norm)
        assignment_logits = assignment_logits / self.hyper_temperature
        assignment = torch.softmax(assignment_logits, dim=-1)

        hypergraph_items = torch.einsum(
            "blm,mh->blh", assignment, self.hyperedge_embeddings
        )
        hypergraph_items = hypergraph_items + self.hyper_mapper(hypergraph_items)
        return self._masked_attention_pool(
            hypergraph_items,
            base_output,
            item_seq.gt(0),
        )

    def _temporal_contextual_intent(
        self,
        sequence_states,
        item_seq,
        item_seq_len,
        base_output,
    ):
        """Aggregate ordered relative-time segments into contextual intent."""
        batch_size, sequence_length, _ = sequence_states.shape
        positions = torch.arange(sequence_length, device=item_seq.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        lengths = item_seq_len.clamp_min(1).unsqueeze(1)
        segment_ids = torch.div(
            positions * self.time_slices,
            lengths,
            rounding_mode="floor",
        ).clamp(max=self.time_slices - 1)

        valid_positions = item_seq.gt(0) & positions.lt(lengths)
        segment_mask = F.one_hot(
            segment_ids,
            num_classes=self.time_slices,
        ).to(sequence_states.dtype)
        segment_mask = segment_mask * valid_positions.unsqueeze(-1)

        segment_sum = torch.einsum("bls,blh->bsh", segment_mask, sequence_states)
        segment_count = segment_mask.sum(dim=1).unsqueeze(-1).clamp_min(1.0)
        segment_states = segment_sum / segment_count
        segment_states = segment_states + self.temporal_mapper(segment_states)

        active_segments = segment_mask.sum(dim=1).gt(0)
        return self._masked_attention_pool(
            segment_states,
            base_output,
            active_segments,
        )

    def _encode_views(self, item_seq, item_seq_len):
        sequence_states = self._encode_sequence_states(item_seq)
        base_output = self.gather_indexes(sequence_states, item_seq_len - 1)
        global_intent = self._global_hypergraph_intent(item_seq, base_output)
        temporal_intent = self._temporal_contextual_intent(
            sequence_states,
            item_seq,
            item_seq_len,
            base_output,
        )
        return base_output, global_intent, temporal_intent

    def forward(self, item_seq, item_seq_len):
        base_output, global_intent, temporal_intent = self._encode_views(
            item_seq,
            item_seq_len,
        )
        intent_output = (
            self.global_intent_weight * global_intent
            + (1.0 - self.global_intent_weight) * temporal_intent
        )
        intent_delta = self.intent_fusion(
            torch.cat([base_output, intent_output], dim=-1)
        )
        return self.intent_norm(
            base_output + self.intent_residual_weight * intent_delta
        )

    def _cross_view_ssl(self, global_intent, temporal_intent):
        if self.ssl_weight <= 0 or global_intent.size(0) < 2:
            return global_intent.new_tensor(0.0)
        global_view = F.normalize(global_intent, dim=-1)
        temporal_view = F.normalize(temporal_intent, dim=-1)
        logits = torch.matmul(global_view, temporal_view.transpose(0, 1))
        logits = logits / self.temperature
        labels = torch.arange(global_intent.size(0), device=global_intent.device)
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.transpose(0, 1), labels)
        )

    @property
    def temperature(self):
        return max(self.hyper_temperature, 1e-6)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        base_output, global_intent, temporal_intent = self._encode_views(
            item_seq,
            item_seq_len,
        )
        intent_output = (
            self.global_intent_weight * global_intent
            + (1.0 - self.global_intent_weight) * temporal_intent
        )
        seq_output = self.intent_norm(
            base_output
            + self.intent_residual_weight
            * self.intent_fusion(torch.cat([base_output, intent_output], dim=-1))
        )

        logits = torch.matmul(
            seq_output,
            self.item_embedding.weight.transpose(0, 1),
        )
        recommendation_loss = self.loss_fct(logits, pos_items)
        ssl_loss = self._cross_view_ssl(global_intent, temporal_intent)
        return recommendation_loss + self.ssl_weight * ssl_loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        return torch.matmul(
            seq_output,
            self.item_embedding.weight.transpose(0, 1),
        )
