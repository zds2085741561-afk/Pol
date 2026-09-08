import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec

class MoEAdaptorLayer(nn.Module):
    """混合专家适配层"""
    def __init__(self, n_exps, layers, dropout=0.0, noise=True, stable_moe=False):
        super(MoEAdaptorLayer, self).__init__()
        self.n_exps = n_exps
        self.stable_moe = stable_moe
        self.noisy_gating = False if stable_moe else noise

        self.experts = nn.ModuleList([nn.Linear(layers[0], layers[1]) for i in range(n_exps)])
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.5)

        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

        if self.stable_moe:
            nn.init.normal_(self.w_gate, mean=0.0, std=1e-3)
            self.layer_norm = nn.LayerNorm(layers[1])

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        return F.softmax(logits, dim=-1)

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        output = multiple_outputs.sum(dim=-2)

        if self.stable_moe:
            output = self.layer_norm(output)
            if not self.training:
                return output

        return self.dropout(output)


class StableDynamicGating(nn.Module):
    """抗衰减残差门控"""
    def __init__(self, hidden_size, dropout_rate=0.3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.graph_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, seq_emb, graph_emb):
        graph_emb = self.graph_transform(graph_emb)
        concat_features = torch.cat([seq_emb, graph_emb], dim=-1)
        g = self.gate(concat_features)
        return self.norm(seq_emb + g * graph_emb)


class TeaRec(SASRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.train_stage = config['train_stage']
        self.temperature = config['temperature']
        self.lam = config['lambda'] if 'lambda' in config else 1.0

        self.target_aware_pool = config['target_aware_graph_pooling'] if 'target_aware_graph_pooling' in config else False
        self.stable_moe = config['stable_moe'] if 'stable_moe' in config else False

        # --- E1 GTF 全局轨迹流图参数 ---
        # E1 同时启用图特征融合和 graph logit；训练期 flow 加分与 MIFG rerank 保持关闭。
        self.use_graph_feature = config['use_graph_feature'] if 'use_graph_feature' in config else True
        self.use_graph_logit = config['use_graph_logit'] if 'use_graph_logit' in config else True
        self.graph_alpha = config['graph_alpha'] if 'graph_alpha' in config else 0.004
        self.graph_score_clip = config['graph_score_clip'] if 'graph_score_clip' in config else 2.0
        self.use_flow_in_train = config['use_flow_in_train'] if 'use_flow_in_train' in config else False
        self.orth_weight = config['orth_weight'] if 'orth_weight' in config else 0.0
        self.use_multi_interest = config['use_multi_interest'] if 'use_multi_interest' in config else False
        self.num_interests = int(config['num_interests'] if 'num_interests' in config else 4)
        self.mic_aggregation = config['mic_aggregation'] if 'mic_aggregation' in config else 'max'

        # 读取 Local Rerank 专属参数。E1 中这些参数保持关闭。
        self.flow_multi_beta = config['flow_multi_beta'] if 'flow_multi_beta' in config else 0.0
        self.use_flow_multi_rerank = config['use_flow_multi_rerank'] if 'use_flow_multi_rerank' in config else False
        self.flow_multi_topk = config['flow_multi_topk'] if 'flow_multi_topk' in config else 20
        self.flow_rerank_recent = int(config['flow_rerank_recent'] if 'flow_rerank_recent' in config else 1)
        self.flow_rerank_scale = float(config['flow_rerank_scale'] if 'flow_rerank_scale' in config else 100.0)
        self.flow_rerank_keep = int(config['flow_rerank_keep'] if 'flow_rerank_keep' in config else 1)
        self.flow_rerank_confidence = float(config['flow_rerank_confidence'] if 'flow_rerank_confidence' in config else 0.30)
        self.flow_rerank_primary_bonus = float(config['flow_rerank_primary_bonus'] if 'flow_rerank_primary_bonus' in config else 0.0)

        print(f"\n==========================================")
        print(
            f"[E4-FULL-TEACHER] use_graph_feature={self.use_graph_feature}, "
            f"use_graph_logit={self.use_graph_logit}, "
            f"use_multi_interest={self.use_multi_interest}, "
            f"num_interests={self.num_interests}, "
            f"graph_alpha={self.graph_alpha}, "
            f"flow_multi_beta={self.flow_multi_beta}, "
            f"rerank={self.use_flow_multi_rerank}, "
            f"rerank_topk={self.flow_multi_topk}, "
            f"rerank_keep={self.flow_rerank_keep}, "
            f"confidence={self.flow_rerank_confidence}, "
            f"primary_bonus={self.flow_rerank_primary_bonus}, "
            f"recent_anchors={self.flow_rerank_recent}"
        )
        print(f"==========================================\n")

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        assert self.train_stage in ['inductive_ft', 'transductive_ft']

        if self.train_stage in ['inductive_ft']:
            self.item_embedding = None

        if self.train_stage in ['transductive_ft']:
            raw_plm = copy.deepcopy(dataset.plm_embedding)
            aligned_plm = torch.zeros((self.n_items, raw_plm.size(1)), dtype=raw_plm.dtype, device=raw_plm.device)
            token2id = dataset.field2token_id['item_id']
            for raw_id_str, internal_id in token2id.items():
                if raw_id_str == '[PAD]': continue
                try:
                    original_idx = int(raw_id_str) - 1
                    if 0 <= original_idx < raw_plm.size(0):
                        aligned_plm[internal_id] = raw_plm[original_idx]
                except ValueError:
                    continue
            self.register_buffer("plm_embedding", aligned_plm.float())

        adaptor_drop = config['adaptor_dropout_prob'] if 'adaptor_dropout_prob' in config else 0.10
        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            adaptor_drop,
            noise=not self.stable_moe,
            stable_moe=self.stable_moe
        )

        if self.use_multi_interest:
            self.interest_queries = nn.Parameter(
                torch.empty(self.num_interests, self.hidden_size)
            )
            nn.init.normal_(self.interest_queries, mean=0.0, std=0.02)
            self.mic_context_norm = nn.LayerNorm(self.hidden_size)
        else:
            self.interest_queries = None
            self.mic_context_norm = None

        if hasattr(dataset, 'graph_embedding') and dataset.graph_embedding is not None:
            self.register_buffer('graph_matrix', dataset.graph_embedding.clone().detach().float())
            self.graph_proj = nn.Linear(self.graph_matrix.size(1), self.hidden_size)

            self.interest_attention = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.Tanh(),
                nn.Linear(self.hidden_size // 2, 1),
            )

            if self.target_aware_pool:
                self.target_aware_attention = nn.Sequential(
                    nn.Linear(self.hidden_size * 2, self.hidden_size // 2),
                    nn.Tanh(),
                    nn.Linear(self.hidden_size // 2, 1)
                )

            self.gating_layer = StableDynamicGating(self.hidden_size, dropout_rate=0.3)
        else:
            self.graph_matrix = None

        if hasattr(dataset, 'graph_indices') and dataset.graph_indices is not None:
            self.register_buffer('graph_indices', dataset.graph_indices.clone().detach().long())
        else:
            self.graph_indices = None

        if hasattr(dataset, 'graph_values') and dataset.graph_values is not None:
            self.register_buffer('graph_values', dataset.graph_values.clone().detach().float())
        else:
            self.graph_values = None

    def _encode_sequence_states(self, item_seq, item_emb):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        if self.train_stage == 'transductive_ft':
            input_emb = input_emb + self.item_embedding(item_seq)

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        return trm_output[-1]

    def forward(self, item_seq, item_emb, item_seq_len):
        sequence_states = self._encode_sequence_states(item_seq, item_emb)
        return self.gather_indexes(sequence_states, item_seq_len - 1)

    def _extract_multi_interest(self, sequence_states, item_seq, item_seq_len):
        """Extract K interests and couple each interest with its GTF trajectory view."""
        if not self.use_multi_interest or self.interest_queries is None:
            seq_output = self.gather_indexes(sequence_states, item_seq_len - 1)
            fused_output, _ = self._fuse_graph_feature(seq_output, item_seq, item_seq_len)
            return fused_output, None

        mask = item_seq.gt(0)
        attention_logits = torch.einsum(
            'blh,kh->bkl', sequence_states, self.interest_queries
        ) / math.sqrt(self.hidden_size)
        attention_logits = attention_logits.masked_fill(~mask.unsqueeze(1), -1e9)
        attention_weights = torch.softmax(attention_logits, dim=-1)

        interest_vectors = torch.einsum(
            'bkl,blh->bkh', attention_weights, sequence_states
        )
        last_output = self.gather_indexes(sequence_states, item_seq_len - 1)
        interest_vectors = self.mic_context_norm(
            interest_vectors + last_output.unsqueeze(1)
        )

        if self.use_graph_feature and getattr(self, 'graph_matrix', None) is not None:
            graph_sequence = self.graph_matrix[item_seq]
            graph_sequence = self.graph_proj(graph_sequence)
            graph_interests = torch.einsum(
                'bkl,blh->bkh', attention_weights, graph_sequence
            )
            interest_vectors = self.gating_layer(interest_vectors, graph_interests)

        return interest_vectors, attention_weights

    def _encode_user_representations(self, item_seq, item_emb, item_seq_len):
        sequence_states = self._encode_sequence_states(item_seq, item_emb)
        user_repr, attention_weights = self._extract_multi_interest(
            sequence_states, item_seq, item_seq_len
        )
        if user_repr.dim() == 3:
            summary = user_repr.mean(dim=1)
        else:
            summary = user_repr
        return user_repr, summary, attention_weights

    def _score_user_representations(self, user_repr, item_embeddings):
        item_embeddings = F.normalize(item_embeddings, dim=-1)
        if user_repr.dim() == 3:
            interest_scores = torch.einsum(
                'bkh,nh->bkn', F.normalize(user_repr, dim=-1), item_embeddings
            )
            if self.mic_aggregation == 'mean':
                return interest_scores.mean(dim=1)
            return interest_scores.max(dim=1).values
        return torch.matmul(
            F.normalize(user_repr, dim=-1), item_embeddings.transpose(0, 1)
        )

    def _interest_orthogonality_loss(self, user_repr):
        if user_repr.dim() != 3 or self.orth_weight <= 0:
            return user_repr.new_tensor(0.0)
        normalized = F.normalize(user_repr, dim=-1)
        gram = torch.matmul(normalized, normalized.transpose(1, 2))
        identity = torch.eye(
            gram.size(1), dtype=gram.dtype, device=gram.device
        ).unsqueeze(0)
        return ((gram - identity) ** 2).mean()

    def _fuse_graph_feature(self, seq_output, item_seq, item_seq_len):
        if (not getattr(self, 'use_graph_feature', True)) or getattr(self, 'graph_matrix', None) is None:
            return seq_output, None

        seq_graph_vec = self.graph_matrix[item_seq]
        seq_graph_emb = self.graph_proj(seq_graph_vec)
        mask = (item_seq > 0).to(seq_output.device)

        B, L, H = seq_graph_emb.shape

        if getattr(self, 'target_aware_pool', False):
            seq_out_expanded = seq_output.unsqueeze(1).expand(-1, L, -1)
            concat_feat = torch.cat([seq_graph_emb, seq_out_expanded], dim=-1)
            scores = self.target_aware_attention(concat_feat.view(B * L, H * 2)).squeeze(-1)
        else:
            combined = seq_graph_emb.view(B * L, H)
            scores = self.interest_attention(combined).squeeze(-1)

        scores = scores.view(B, L)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled_graph = torch.bmm(weights.unsqueeze(1), seq_graph_emb).squeeze(1)

        fused = self.gating_layer(seq_output, pooled_graph)
        return fused, pooled_graph

    # =========================================================================
    # TopK 局部重排引擎。E1 默认关闭，保留给后续消融版本。
    # =========================================================================
    def _legacy_apply_local_flow_multi_rerank(self, scores, flow_multi_score):
        beta = float(getattr(self, "flow_multi_beta", 0.0))
        use_rerank = getattr(self, "use_flow_multi_rerank", False)
        topk = int(getattr(self, "flow_multi_topk", 100))

        if (not use_rerank) or beta <= 0:
            return scores

        topk = min(topk, scores.size(1))

        # 只取 Teacher 原始 topK 候选
        topk_values, topk_indices = torch.topk(scores, topk, dim=-1)

        # 只在 topK 内取多兴趣流图分数
        flow_delta = flow_multi_score.gather(1, topk_indices)

        # 每个用户 topK 内部 z-score，防止全局漂移污染边界
        flow_delta = (
            flow_delta - flow_delta.mean(dim=1, keepdim=True)
        ) / (flow_delta.std(dim=1, keepdim=True) + 1e-8)

        new_topk_values = topk_values + beta * flow_delta

        # 原始 scores 不动，只替换 topK 位置
        new_scores = scores.clone()
        new_scores.scatter_(1, topk_indices, new_topk_values)

        # Debug 探针打印
        if not hasattr(self, "_mifg_local_debug_printed"):
            print("\n" + "="*50)
            print(f"🚨 [LOCAL RERANK DEBUG] Mapped!")
            print(f"TopK isolated: {topk}")
            print(f"Delta mean: {flow_delta.mean().item():.4f}, std: {flow_delta.std().item():.4f}")
            print("="*50 + "\n")
            self._mifg_local_debug_printed = True

        return new_scores

    # =========================================================================
    # 多兴趣流图分数。E1 默认关闭，保留给后续消融版本。
    # =========================================================================
    def _legacy_add_flow_multi_score(self, scores, item_seq, item_seq_len, test_items_emb=None):
        if (
            self.flow_multi_beta <= 0.0
            or (not getattr(self, 'use_graph_feature', True))
            or getattr(self, 'graph_matrix', None) is None
        ):
            return scores

        try:
            B, N = scores.shape
            device = scores.device

            # 提取历史概率 [Batch_size, Seq_Len, Num_Items]
            hist_prob = self.graph_matrix[item_seq].to(device).float()

            mask = (item_seq > 0).unsqueeze(-1).expand(-1, -1, N).to(device)
            hist_prob = hist_prob.masked_fill(~mask, 0.0)

            # Max-Pooling 获取全序列最强兴趣流
            multi_prob, _ = hist_prob.max(dim=1)
            raw_multi_score = torch.log(multi_prob + 1e-8)

            # 如果启用了局部重排，则切入独立沙盒处理
            if getattr(self, "use_flow_multi_rerank", False):
                return self._apply_local_flow_multi_rerank(scores, raw_multi_score)
            else:
                # 兼容老的全局加分 (如果后续你想验证)
                clip_val = float(getattr(self, 'graph_score_clip', 2.0))
                multi_score = (raw_multi_score - raw_multi_score.mean(dim=-1, keepdim=True)) / (raw_multi_score.std(dim=-1, keepdim=True) + 1e-8)
                multi_score = torch.clamp(multi_score, -clip_val, clip_val)
                return scores + self.flow_multi_beta * multi_score

        except Exception as e:
            print(f"\n[Warning] Flow Multi Score bypassed due to: {e}")
            return scores


    # 原始的 graph logit prior (Alpha 分支 - Last item)
    def _apply_local_flow_multi_rerank(self, scores, topk_indices, flow_prob):
        beta = float(getattr(self, "flow_multi_beta", 0.0))
        if (not getattr(self, "use_flow_multi_rerank", False)) or beta <= 0:
            return scores

        topk_values = scores.gather(1, topk_indices)
        flow_delta = torch.log1p(
            float(getattr(self, "flow_rerank_scale", 100.0)) * flow_prob.clamp_min(0.0)
        )
        flow_delta = flow_delta / flow_delta.max(dim=1, keepdim=True).values.clamp_min(1e-8)
        keep = min(max(1, int(getattr(self, "flow_rerank_keep", 1))), flow_delta.size(1))
        keep_indices = flow_delta.topk(keep, dim=1).indices
        keep_mask = torch.zeros_like(flow_delta)
        keep_mask.scatter_(1, keep_indices, 1.0)
        positive = flow_prob.gt(0)
        strongest = flow_delta.topk(min(2, flow_delta.size(1)), dim=1)
        confidence_threshold = float(getattr(self, "flow_rerank_confidence", 0.0))
        confidence_gate = torch.ones_like(flow_delta[:, :1])
        if strongest.values.size(1) > 1 and confidence_threshold > 0:
            confidence_gate = (
                strongest.values[:, :1] - strongest.values[:, 1:2]
            ).ge(confidence_threshold)
        tail_delta = flow_delta * keep_mask * positive * confidence_gate

        primary_bonus = float(getattr(self, "flow_rerank_primary_bonus", 0.0))
        primary_delta = torch.zeros_like(flow_delta)
        if primary_bonus > 0:
            primary_mask = torch.zeros_like(flow_delta)
            primary_mask.scatter_(1, strongest.indices[:, :1], 1.0)
            primary_mask = primary_mask * confidence_gate
            primary_delta = flow_delta * primary_mask * positive

        effective_delta = beta * tail_delta + primary_bonus * primary_delta
        new_topk_values = topk_values + effective_delta

        # Preserve the E2 candidate set: only original topK positions may change.
        new_scores = scores.clone()
        new_scores.scatter_(1, topk_indices, new_topk_values)

        if not hasattr(self, "_mifg_local_debug_printed"):
            coverage = effective_delta.gt(0).float().mean().item()
            print("\n" + "=" * 50)
            print("[E4-MIFG] Local trajectory-flow rerank enabled")
            print(f"TopK isolated: {topk_indices.size(1)}")
            print(f"Flow coverage: {coverage:.4f}")
            print(
                f"Delta mean: {effective_delta.mean().item():.4f}, "
                f"max: {effective_delta.max().item():.4f}"
            )
            print("=" * 50 + "\n")
            self._mifg_local_debug_printed = True

        return new_scores

    def _select_mifg_anchors(self, item_seq, item_seq_len, attention_weights=None):
        """Combine recent trajectory anchors with one anchor from each MIC interest."""
        valid_last = torch.clamp(item_seq_len - 1, min=0, max=item_seq.size(1) - 1)
        recent_count = max(1, int(getattr(self, "flow_rerank_recent", 3)))
        recent_offsets = torch.arange(recent_count, device=item_seq.device).view(1, -1)
        recent_pos = (valid_last.unsqueeze(1) - recent_offsets).clamp_min(0)
        recent_items = item_seq.gather(1, recent_pos)
        recent_weights = torch.linspace(
            1.0, 0.6, recent_count, device=item_seq.device
        ).view(1, -1).expand(item_seq.size(0), -1)

        anchor_items = recent_items
        anchor_weights = recent_weights
        if attention_weights is not None and attention_weights.dim() == 3:
            interest_pos = attention_weights.argmax(dim=-1)
            interest_items = item_seq.gather(1, interest_pos)
            interest_weights = attention_weights.max(dim=-1).values.clamp_min(0.25)
            anchor_items = torch.cat([anchor_items, interest_items], dim=1)
            anchor_weights = torch.cat([anchor_weights, interest_weights], dim=1)

        valid = anchor_items.gt(0).float()
        return anchor_items, anchor_weights * valid

    def _candidate_flow_probability(self, scores, topk_indices, anchor_items, anchor_weights):
        """Return MIFG flow probability for original topK candidates."""
        device = scores.device

        if (
            getattr(self, "graph_indices", None) is not None
            and getattr(self, "graph_values", None) is not None
        ):
            graph_indices = self.graph_indices[anchor_items].to(device)
            graph_values = self.graph_values[anchor_items].to(device).float()
            valid = (
                graph_indices.gt(0)
                & graph_indices.lt(scores.size(1))
                & graph_values.gt(0)
            )
            weighted_values = (
                graph_values * anchor_weights.to(device).unsqueeze(-1) * valid
            )

            # Avoid materializing [batch, anchors, topK, graph_topK] on NYC_STEPS.
            sparse_prior = torch.zeros_like(scores)
            sparse_prior.scatter_add_(
                1,
                graph_indices.clamp(min=0, max=scores.size(1) - 1).flatten(1),
                weighted_values.flatten(1),
            )
            return sparse_prior.gather(1, topk_indices)

        if (
            getattr(self, "graph_matrix", None) is not None
            and self.graph_matrix.size(1) == scores.size(1)
        ):
            graph_rows = self.graph_matrix[anchor_items].to(device).float()
            candidate_index = topk_indices.unsqueeze(1).expand(
                -1, anchor_items.size(1), -1
            )
            candidate_flow = graph_rows.gather(2, candidate_index)
            candidate_flow = candidate_flow * anchor_weights.to(device).unsqueeze(-1)
            return candidate_flow.max(dim=1).values

        return None

    def _add_flow_multi_score(
        self,
        scores,
        item_seq,
        item_seq_len,
        test_items_emb=None,
        attention_weights=None,
    ):
        if (
            self.flow_multi_beta <= 0.0
            or (not getattr(self, "use_graph_feature", True))
            or (not getattr(self, "use_flow_multi_rerank", False))
        ):
            return scores

        try:
            topk = min(
                max(1, int(getattr(self, "flow_multi_topk", 100))),
                scores.size(1),
            )
            topk_indices = torch.topk(scores, topk, dim=-1).indices
            anchor_items, anchor_weights = self._select_mifg_anchors(
                item_seq, item_seq_len, attention_weights
            )
            flow_prob = self._candidate_flow_probability(
                scores, topk_indices, anchor_items, anchor_weights
            )
            if flow_prob is None:
                return scores
            return self._apply_local_flow_multi_rerank(scores, topk_indices, flow_prob)
        except Exception as e:
            print(f"\n[Warning] E4 MIFG rerank bypassed due to: {e}")
            return scores

    def _add_graph_logit_prior(self, scores, item_seq, item_seq_len):
        if (
            not getattr(self, 'use_graph_logit', False)
            or (not getattr(self, 'use_graph_feature', True))
        ):
            return scores
        try:
            valid_len = torch.clamp(item_seq_len - 1, min=0, max=item_seq.size(1) - 1)
            last_item_idx = item_seq.gather(1, valid_len.unsqueeze(1)).squeeze(1)

            # NYC_STEPS / Massive 等大数据集使用 sparse topK GTF 先验。
            if (
                getattr(self, 'graph_indices', None) is not None
                and getattr(self, 'graph_values', None) is not None
            ):
                graph_indices = self.graph_indices[last_item_idx].to(scores.device)
                graph_values = self.graph_values[last_item_idx].to(scores.device).float()
                valid = (
                    (graph_indices > 0)
                    & (graph_indices < scores.size(1))
                    & (graph_values > 0)
                )

                raw_score = torch.log(graph_values.clamp_min(1e-8))
                count = valid.sum(dim=1, keepdim=True).clamp_min(1)
                mean = (raw_score * valid).sum(dim=1, keepdim=True) / count
                centered = (raw_score - mean) * valid
                std = torch.sqrt(
                    (centered.pow(2).sum(dim=1, keepdim=True) / count).clamp_min(1e-8)
                )
                graph_delta = centered / std
                clip_val = float(getattr(self, 'graph_score_clip', 2.0))
                graph_delta = torch.clamp(graph_delta, -clip_val, clip_val) * valid

                prior = torch.zeros_like(scores)
                prior.scatter_add_(1, graph_indices.clamp(min=0, max=scores.size(1) - 1), graph_delta)
                alpha = float(getattr(self, 'graph_alpha', 0.004))
                return scores + alpha * prior

            # 小数据集使用完整 dense 一/二跳 GTF 先验。
            if (
                getattr(self, 'graph_matrix', None) is None
                or self.graph_matrix.size(1) != scores.size(1)
            ):
                return scores

            graph_prob = self.graph_matrix[last_item_idx].to(scores.device).float()
            graph_score = torch.log(graph_prob + 1e-8)
            mean = graph_score.mean(dim=-1, keepdim=True)
            std = graph_score.std(dim=-1, keepdim=True) + 1e-8
            graph_score = (graph_score - mean) / std
            clip_val = float(getattr(self, 'graph_score_clip', 2.0))
            graph_score = torch.clamp(graph_score, -clip_val, clip_val)
            alpha = float(getattr(self, 'graph_alpha', 0.004))
            return scores + alpha * graph_score
        except Exception as e:
            return scores


    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        item_emb_list = self.moe_adaptor(self.plm_embedding[item_seq])
        user_repr, _, attention_weights = self._encode_user_representations(
            item_seq, item_emb_list, item_seq_len
        )

        test_item_emb = self.moe_adaptor(self.plm_embedding)
        if self.train_stage == 'transductive_ft':
            test_item_emb = test_item_emb + self.item_embedding.weight

        logits = self._score_user_representations(user_repr, test_item_emb) / self.temperature

        if getattr(self, 'use_flow_in_train', False):
            logits = self._add_graph_logit_prior(logits, item_seq, item_seq_len)
            logits = self._add_flow_multi_score(
                logits, item_seq, item_seq_len, test_item_emb, attention_weights
            )

        pos_items = interaction[self.POS_ITEM_ID]
        recommendation_loss = self.loss_fct(logits, pos_items)
        orthogonality_loss = self._interest_orthogonality_loss(user_repr)
        return recommendation_loss + self.orth_weight * orthogonality_loss

    def get_candidate_embeddings(self):
        """Return normalized teacher-side POI embeddings for TF-PQ relation distillation."""
        item_emb = self.moe_adaptor(self.plm_embedding)
        if self.train_stage == 'transductive_ft':
            item_emb = item_emb + self.item_embedding.weight
        return F.normalize(item_emb, dim=-1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        item_emb_list = self.moe_adaptor(self.plm_embedding[item_seq])
        user_repr, _, attention_weights = self._encode_user_representations(
            item_seq, item_emb_list, item_seq_len
        )

        test_items_emb = self.moe_adaptor(self.plm_embedding)
        if self.train_stage == 'transductive_ft':
            test_items_emb = test_items_emb + self.item_embedding.weight

        scores = self._score_user_representations(user_repr, test_items_emb)

        # E1 开启 graph logit；flow_multi/MIFG 分支仍由配置关闭。
        scores = self._add_graph_logit_prior(scores, item_seq, item_seq_len)
        scores = self._add_flow_multi_score(
            scores, item_seq, item_seq_len, test_items_emb, attention_weights
        )

        return scores

    def get_teacher_outputs(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        item_emb_list = self.moe_adaptor(self.plm_embedding[item_seq])
        user_repr, summary, attention_weights = self._encode_user_representations(
            item_seq, item_emb_list, item_seq_len
        )

        test_items_emb = self.moe_adaptor(self.plm_embedding)
        if self.train_stage == 'transductive_ft':
            test_items_emb = test_items_emb + self.item_embedding.weight

        scores = self._score_user_representations(user_repr, test_items_emb)
        scores = self._add_graph_logit_prior(scores, item_seq, item_seq_len)
        scores = self._add_flow_multi_score(
            scores, item_seq, item_seq_len, test_items_emb, attention_weights
        )

        return scores, F.normalize(summary, dim=-1)
