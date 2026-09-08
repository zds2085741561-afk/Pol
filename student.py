import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.utils.enum_type import InputType


def cfg_get(config, key, default=None):
    """Safe config getter for RecBole Config / dict."""
    try:
        value = config[key]
    except Exception:
        return default
    return default if value is None else value


def cfg_bool(config, key, default=False):
    """Parse bool safely, avoiding bool('False') == True."""
    value = cfg_get(config, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ["true", "1", "yes", "y"]
    return bool(value)


class StableDynamicGating(nn.Module):
    """Optional graph feature fusion gate. Default config keeps it off."""

    def __init__(self, hidden_size, dropout_rate=0.1):
        super().__init__()
        self.graph_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_rate),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, seq_emb, graph_emb):
        graph_emb = self.graph_transform(graph_emb)
        gate = self.gate(torch.cat([seq_emb, graph_emb], dim=-1))
        return self.norm(seq_emb + gate * graph_emb)


class TrajectoryFlowContextAdapter(nn.Module):
    """Inject a compact GTF neighborhood context into the student state."""

    def __init__(
        self,
        hidden_size,
        dropout_rate=0.1,
        gate_bias=-2.0,
        residual_init=0.0,
    ):
        super().__init__()
        self.context_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.gate_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.gate_out = nn.Linear(hidden_size, hidden_size)
        self.gate_bias = float(gate_bias)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def reset_gate(self):
        # Start close to the S2 representation and learn graph injection gradually.
        nn.init.zeros_(self.gate_out.weight)
        nn.init.constant_(self.gate_out.bias, self.gate_bias)

    def forward(self, seq_emb, graph_context, valid_context):
        context = self.context_transform(graph_context)
        gate_hidden = F.gelu(
            self.gate_hidden(torch.cat([seq_emb, context], dim=-1))
        )
        gate = torch.sigmoid(self.gate_out(gate_hidden))
        # tanh(0)=0 makes a warm-started adapter exactly preserve S1 outputs.
        residual_weight = torch.tanh(self.residual_scale)
        fused = seq_emb + residual_weight * gate * context
        fused = torch.where(valid_context.unsqueeze(-1), fused, seq_emb)
        return fused, gate, residual_weight


class StuRec(SequentialRecommender):
    """
    StuRec E4 full student.

    E0 仅保留 PQ + Transformer + 主排序分支 KD。
    Item residual、multi-interest、graph、rerank 和辅助损失全部关闭。
    增强模块的实现仍保留在完整代码中，便于后续消融版本复用。
    """

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # ---------- PQ / base config ----------
        self.code_dim = int(cfg_get(config, "code_dim", 64))
        self.code_cap = int(cfg_get(config, "code_cap", 256))
        self.temperature = float(cfg_get(config, "temperature", 0.07))
        self.ce_temperature = float(cfg_get(config, "ce_temperature", 0.10))
        self.train_stage = cfg_get(config, "train_stage", "transductive_ft")

        # ---------- Transformer ----------
        self.n_layers = int(cfg_get(config, "n_layers", 2))
        self.n_heads = int(cfg_get(config, "n_heads", 2))
        self.hidden_size = int(cfg_get(config, "hidden_size", 64))
        self.inner_size = int(cfg_get(config, "inner_size", 256))
        self.hidden_dropout_prob = float(cfg_get(config, "hidden_dropout_prob", 0.30))
        self.attn_dropout_prob = float(cfg_get(config, "attn_dropout_prob", 0.30))
        self.hidden_act = cfg_get(config, "hidden_act", "gelu")
        self.layer_norm_eps = float(cfg_get(config, "layer_norm_eps", 1e-12))
        self.initializer_range = float(cfg_get(config, "initializer_range", 0.02))

        # ---------- Multi-interest ----------
        self.use_cmid = cfg_bool(config, "use_cmid", False)
        self.use_multi_interest = (
            cfg_bool(config, "use_multi_interest", False) or self.use_cmid
        )
        self.num_interests = int(cfg_get(config, "num_interests", 1))
        self.cmid_loss_weight = float(cfg_get(config, "cmid_loss_weight", 5.0))
        self.cmid_diversity_weight = float(
            cfg_get(config, "cmid_diversity_weight", 0.01)
        )
        self.cmid_max_fusion_weight = float(
            cfg_get(config, "cmid_max_fusion_weight", 0.20)
        )
        self.cmid_fusion_init = float(cfg_get(config, "cmid_fusion_init", 0.05))

        # ---------- Graph / rerank ----------
        self.use_graph_feature = cfg_bool(config, "use_graph_feature", False)
        self.use_graph_logit = cfg_bool(config, "use_graph_logit", False)
        self.use_catf = cfg_bool(config, "use_catf", False)
        self.catf_alpha = float(cfg_get(config, "catf_alpha", 0.0))
        self.catf_uncertainty_topk = int(
            cfg_get(config, "catf_uncertainty_topk", 100)
        )
        self.catf_temperature = float(cfg_get(config, "catf_temperature", 1.0))
        self.graph_alpha = float(
            cfg_get(
                config,
                "student_graph_alpha",
                cfg_get(config, "graph_alpha", cfg_get(config, "graph_logit_alpha", 0.0)),
            )
        )
        self.graph_score_clip = float(cfg_get(config, "graph_score_clip", 2.0))
        self.graph_logit_alpha = self.graph_alpha
        self.graph_logit_scale = float(cfg_get(config, "graph_logit_scale", 80.0))

        self.use_local_rerank = cfg_bool(config, "use_local_rerank", False)
        self.local_rerank_topk = int(cfg_get(config, "local_rerank_topk", 100))
        self.local_graph_beta = float(cfg_get(config, "local_graph_beta", 0.0))
        self.local_graph_2hop_ratio = float(cfg_get(config, "local_graph_2hop_ratio", 0.0))
        self.local_repeat_beta = float(cfg_get(config, "local_repeat_beta", 0.0))
        self.local_interest_beta = float(cfg_get(config, "local_interest_beta", 0.0))

        # ---------- Item residual ----------
        self.use_item_residual = cfg_bool(config, "use_item_residual", False)
        self.residual_weight = float(cfg_get(config, "residual_weight", 0.0))

        # ---------- E4 full loss weights ----------
        self.ce_loss_weight = float(cfg_get(config, "ce_loss_weight", 1.0))
        self.distill_loss_weight = float(cfg_get(config, "distill_loss_weight", 1.0))
        self.rank_loss_weight = float(cfg_get(config, "rank_loss_weight", 0.0))
        self.graph_rank_loss_weight = float(cfg_get(config, "graph_rank_loss_weight", 0.0))
        self.mse_loss_weight = float(cfg_get(config, "mse_loss_weight", 0.0))
        self.use_feature_kd = cfg_bool(config, "use_feature_kd", False)
        self.feature_kd_loss_weight = float(
            cfg_get(config, "feature_kd_loss_weight", 0.0)
        )
        self.contrastive_loss_weight = float(cfg_get(config, "contrastive_loss_weight", 0.0))

        self.distill_temperature = float(cfg_get(config, "distill_temperature", 3.0))
        self.distill_topk = int(cfg_get(config, "distill_topk", 100))
        self.use_tfkd = cfg_bool(config, "use_tfkd", False)
        self.tfkd_topk = int(cfg_get(config, "tfkd_topk", self.distill_topk))
        self.tfkd_graph_topk = int(cfg_get(config, "tfkd_graph_topk", 50))
        self.use_tfca = cfg_bool(config, "use_tfca", False)
        self.tfca_topk = int(cfg_get(config, "tfca_topk", 20))
        self.tfca_gate_bias = float(cfg_get(config, "tfca_gate_bias", -2.0))
        self.tfca_residual_init = float(cfg_get(config, "tfca_residual_init", 0.0))
        self.use_tfpq = cfg_bool(config, "use_tfpq", False)
        self.tfpq_loss_weight = float(cfg_get(config, "tfpq_loss_weight", 0.5))
        self.tfpq_graph_topk = int(cfg_get(config, "tfpq_graph_topk", 10))
        self.use_ndcg_rank = cfg_bool(config, "use_ndcg_rank", False)
        self.ndcg_rank_loss_weight = float(
            cfg_get(config, "ndcg_rank_loss_weight", 0.0)
        )
        self.ndcg_rank_candidate_topk = int(
            cfg_get(config, "ndcg_rank_candidate_topk", 100)
        )
        self.ndcg_rank_hard_neg_num = int(
            cfg_get(config, "ndcg_rank_hard_neg_num", 10)
        )
        self.ndcg_rank_cutoff = int(cfg_get(config, "ndcg_rank_cutoff", 10))
        self.ndcg_rank_margin = float(cfg_get(config, "ndcg_rank_margin", 0.0))
        self.rank_topk = int(cfg_get(config, "rank_topk", 30))
        self.graph_rank_topk = int(cfg_get(config, "graph_rank_topk", 10))
        self.rank_margin = float(cfg_get(config, "rank_margin", 0.05))
        self.teacher_hidden_size = int(cfg_get(config, "teacher_hidden_size", 300))
        self.user_id_field = cfg_get(config, "USER_ID_FIELD", "user_id")

        # ---------- Buffers ----------
        self.register_buffer("pq_codes", self._load_pq_codes(dataset))
        max_code = int(self.pq_codes.max().item()) if self.pq_codes.numel() > 0 else 0
        self.codebook_size = max(self.code_dim * (self.code_cap + 1), max_code + 1)

        self._init_graph(dataset)
        self._init_user_seq_bank(dataset)

        # ---------- Modules ----------
        self.pq_code_embedding = nn.Embedding(
            self.codebook_size, self.hidden_size, padding_idx=0
        )

        self.item_residual_embedding = None
        if self.use_item_residual:
            self.item_residual_embedding = nn.Embedding(
                self.n_items, self.hidden_size, padding_idx=0
            )

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.multi_interest_proj = None
        if self.use_multi_interest:
            self.multi_interest_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size * self.num_interests),
                nn.Tanh(),
            )
        self.cmid_fusion_logit = None
        if self.use_cmid:
            max_weight = max(self.cmid_max_fusion_weight, 1e-6)
            init_ratio = min(max(self.cmid_fusion_init / max_weight, 1e-4), 1 - 1e-4)
            self.cmid_fusion_logit = nn.Parameter(
                torch.tensor(float(math.log(init_ratio / (1.0 - init_ratio))))
            )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()
        self.last_loss_components = {}
        self.last_tfca_gate_mean = 0.0
        self.last_tfca_coverage = 0.0
        self.last_tfca_residual_weight = 0.0
        self.last_catf_uncertainty_mean = 0.0
        self.last_cmid_fusion_weight = 0.0
        self.last_ndcg_positive_rank = 0.0
        self.last_ndcg_active_pairs = 0.0
        self.last_ndcg_score_margin = 0.0
        self.align_layer = None
        if self.mse_loss_weight > 0 or (
            self.use_feature_kd and self.feature_kd_loss_weight > 0
        ):
            self.align_layer = nn.Linear(self.hidden_size, self.teacher_hidden_size)

        self.graph_proj = None
        self.gating_layer = None
        self.interest_attention = None
        if self.graph_matrix is not None and self.use_graph_feature:
            self.graph_proj = nn.Linear(self.graph_matrix.size(1), self.hidden_size)
            attn_hidden = max(1, self.hidden_size // 2)
            self.interest_attention = nn.Sequential(
                nn.Linear(self.hidden_size, attn_hidden),
                nn.Tanh(),
                nn.Linear(attn_hidden, 1),
            )
            self.gating_layer = StableDynamicGating(
                self.hidden_size, dropout_rate=self.hidden_dropout_prob
            )

        self.tfca_adapter = None
        if self.use_tfca:
            self.tfca_adapter = TrajectoryFlowContextAdapter(
                self.hidden_size,
                dropout_rate=self.hidden_dropout_prob,
                gate_bias=self.tfca_gate_bias,
                residual_init=self.tfca_residual_init,
            )

        self.apply(self._init_weights)
        if self.tfca_adapter is not None:
            self.tfca_adapter.reset_gate()

    # =========================================================
    # Init helpers
    # =========================================================
    def _load_pq_codes(self, dataset):
        if hasattr(dataset, "pq_codes") and dataset.pq_codes is not None:
            codes_tensor = torch.as_tensor(dataset.pq_codes, dtype=torch.long)

            if codes_tensor.size(0) == self.n_items - 1:
                pad_code = torch.zeros((1, codes_tensor.size(1)), dtype=torch.long)
                codes_tensor = torch.cat([pad_code, codes_tensor], dim=0)
            elif codes_tensor.size(0) < self.n_items:
                pad_len = self.n_items - codes_tensor.size(0)
                pad_code = torch.zeros((pad_len, codes_tensor.size(1)), dtype=torch.long)
                codes_tensor = torch.cat([codes_tensor, pad_code], dim=0)
            elif codes_tensor.size(0) > self.n_items:
                codes_tensor = codes_tensor[: self.n_items]
        else:
            codes_tensor = torch.zeros((self.n_items, self.code_dim), dtype=torch.long)

        if codes_tensor.size(1) > self.code_dim:
            codes_tensor = codes_tensor[:, : self.code_dim]
        elif codes_tensor.size(1) < self.code_dim:
            pad_dim = self.code_dim - codes_tensor.size(1)
            codes_tensor = torch.cat(
                [
                    codes_tensor,
                    torch.zeros((codes_tensor.size(0), pad_dim), dtype=torch.long),
                ],
                dim=1,
            )

        codes_tensor = codes_tensor.clamp(min=0, max=self.code_cap)
        base_id = torch.arange(self.code_dim, dtype=torch.long) * (self.code_cap + 1)
        codes_tensor = codes_tensor + base_id.unsqueeze(0) + 1
        codes_tensor[0].zero_()
        return codes_tensor

    def _init_graph(self, dataset):
        # Keep the graph for TF-KD candidate construction even when the pure
        # student disables graph feature/logit scoring.
        if (
            not self.use_graph_feature
            and not self.use_graph_logit
            and not self.use_catf
            and not self.use_tfkd
            and not self.use_tfca
            and not self.use_tfpq
        ):
            self.graph_matrix = None
            self.graph_indices = None
            self.graph_values = None
            self.graph_matrix_2hop = None
            return

        if hasattr(dataset, "graph_embedding") and dataset.graph_embedding is not None:
            graph_tensor = dataset.graph_embedding.clone().detach().float()
            self.register_buffer("graph_matrix", graph_tensor)
        else:
            self.graph_matrix = None

        if hasattr(dataset, "graph_indices") and dataset.graph_indices is not None:
            self.register_buffer(
                "graph_indices", dataset.graph_indices.clone().detach().long()
            )
        else:
            self.graph_indices = None

        if hasattr(dataset, "graph_values") and dataset.graph_values is not None:
            self.register_buffer(
                "graph_values", dataset.graph_values.clone().detach().float()
            )
        else:
            self.graph_values = None

        self.graph_matrix_2hop = None

    def _init_user_seq_bank(self, dataset):
        user_num = int(getattr(dataset, "user_num", getattr(self, "n_users", 0)))
        if user_num <= 0:
            user_num = int(getattr(self, "n_users", 1))

        seq_bank = torch.zeros((user_num, self.max_seq_length), dtype=torch.long)
        len_bank = torch.ones(user_num, dtype=torch.long)
        inter_feat = getattr(dataset, "inter_feat", None)

        if inter_feat is not None:
            user_ids = self._field(inter_feat, self.user_id_field)
            item_seq = self._field(inter_feat, self.ITEM_SEQ)
            item_seq_len = self._field(inter_feat, self.ITEM_SEQ_LEN)

            if user_ids is not None and item_seq is not None:
                if item_seq_len is None:
                    item_seq_len = item_seq.ne(0).sum(dim=1)

                for uid, seq, seq_len in zip(
                    user_ids.long(), item_seq.long(), item_seq_len.long()
                ):
                    uid_value = int(uid.item())
                    if uid_value < 0 or uid_value >= user_num:
                        continue
                    length = int(seq_len.item())
                    if length <= 0:
                        continue
                    seq = seq[:length][-self.max_seq_length :]
                    seq_bank[uid_value, -seq.numel() :] = seq
                    len_bank[uid_value] = max(1, seq.numel())

        self.register_buffer("user_seq_bank", seq_bank)
        self.register_buffer("user_seq_len_bank", len_bank)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # =========================================================
    # Embeddings / forward
    # =========================================================
    def get_raw_pq_item_embeddings(self):
        return self.pq_code_embedding(self.pq_codes).mean(dim=-2)

    def calculate_item_emb(self):
        base_emb = self.get_raw_pq_item_embeddings()
        if self.use_item_residual and self.item_residual_embedding is not None:
            item_ids = torch.arange(self.n_items, device=base_emb.device)
            return base_emb + self.residual_weight * self.item_residual_embedding(item_ids)
        return base_emb

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        all_item_emb = self.calculate_item_emb()
        input_emb = F.embedding(item_seq, all_item_emb, padding_idx=0)
        input_emb = self.LayerNorm(input_emb + self.position_embedding(position_ids))
        input_emb = self.dropout(input_emb)

        trm_output = self.trm_encoder(
            input_emb,
            self.get_attention_mask(item_seq),
            output_all_encoded_layers=True,
        )
        output = trm_output[-1]
        gather_index = (item_seq_len - 1).clamp(min=0)
        seq_output = self.gather_indexes(output, gather_index)

        if self.use_tfca and self.tfca_adapter is not None:
            graph_context, valid_context = self._tfca_context(
                item_seq, item_seq_len, all_item_emb
            )
            seq_output, tfca_gate, tfca_residual_weight = self.tfca_adapter(
                seq_output, graph_context, valid_context
            )
            with torch.no_grad():
                self.last_tfca_coverage = float(valid_context.float().mean().item())
                if valid_context.any():
                    self.last_tfca_gate_mean = float(
                        tfca_gate[valid_context].mean().item()
                    )
                else:
                    self.last_tfca_gate_mean = 0.0
                self.last_tfca_residual_weight = float(
                    tfca_residual_weight.item()
                )

        # E1 graph feature fusion.
        if (
            self.use_graph_feature
            and self.graph_matrix is not None
            and self.graph_proj is not None
            and self.gating_layer is not None
            and self.interest_attention is not None
        ):
            seq_output = self._fuse_graph_feature(seq_output, item_seq)

        return seq_output

    def _last_item(self, item_seq, item_seq_len):
        gather_index = (item_seq_len - 1).clamp(min=0)
        return item_seq[torch.arange(item_seq.size(0), device=item_seq.device), gather_index]

    def _tfca_context(self, item_seq, item_seq_len, item_emb):
        """Aggregate Top-K trajectory-flow neighbors using compact POI embeddings."""
        last_item = self._last_item(item_seq, item_seq_len)
        batch_size = last_item.size(0)
        zero_context = item_emb.new_zeros((batch_size, self.hidden_size))
        zero_valid = torch.zeros(batch_size, dtype=torch.bool, device=item_emb.device)

        if self.tfca_topk <= 0:
            return zero_context, zero_valid

        if self.graph_indices is not None and self.graph_values is not None:
            graph_indices = self.graph_indices[
                last_item.to(self.graph_indices.device)
            ].to(item_emb.device)
            graph_values = self.graph_values[
                last_item.to(self.graph_values.device)
            ].to(item_emb.device)
            valid = (
                graph_indices.gt(0)
                & graph_indices.lt(self.n_items)
                & graph_values.gt(0)
            )
            ranked_values = graph_values.masked_fill(~valid, -1.0)
            k = min(self.tfca_topk, ranked_values.size(1))
            if k <= 0:
                return zero_context, zero_valid
            flow_weight, positions = ranked_values.topk(k, dim=1)
            neighbor_ids = graph_indices.gather(1, positions)
            valid = flow_weight.gt(0)
            flow_weight = flow_weight.clamp_min(0.0)
        elif self.graph_matrix is not None and self.graph_matrix.size(1) == self.n_items:
            graph_prob = self.graph_matrix[
                last_item.to(self.graph_matrix.device)
            ].to(item_emb.device).clone()
            graph_prob[:, 0] = -1.0
            k = min(self.tfca_topk, max(0, graph_prob.size(1) - 1))
            if k <= 0:
                return zero_context, zero_valid
            flow_weight, neighbor_ids = graph_prob.topk(k, dim=1)
            valid = flow_weight.gt(0) & neighbor_ids.gt(0)
            flow_weight = flow_weight.clamp_min(0.0)
        else:
            return zero_context, zero_valid

        flow_weight = flow_weight * valid.float()
        row_sum = flow_weight.sum(dim=1, keepdim=True)
        valid_rows = row_sum.squeeze(1).gt(0)
        flow_weight = flow_weight / row_sum.clamp_min(1e-8)
        safe_ids = neighbor_ids.clamp(min=0, max=self.n_items - 1)
        neighbor_emb = F.embedding(safe_ids, item_emb, padding_idx=0)
        context = (flow_weight.unsqueeze(-1) * neighbor_emb).sum(dim=1)
        context = torch.where(valid_rows.unsqueeze(-1), context, zero_context)
        return context, valid_rows

    def _fuse_graph_feature(self, seq_output, item_seq):
        graph_vec = self.graph_matrix[item_seq.to(self.graph_matrix.device)].to(seq_output.device)
        graph_emb = self.graph_proj(graph_vec)
        mask = item_seq.gt(0).to(seq_output.device)

        batch_size, seq_len, hidden_size = graph_emb.shape
        attn_score = self.interest_attention(
            graph_emb.reshape(batch_size * seq_len, hidden_size)
        ).view(batch_size, seq_len)
        attn_score = attn_score.masked_fill(~mask, -1e9)
        attn_weight = torch.softmax(attn_score, dim=1)
        pooled_graph = torch.bmm(attn_weight.unsqueeze(1), graph_emb).squeeze(1)
        return self.gating_layer(seq_output, pooled_graph)

    # =========================================================
    # Scoring
    # =========================================================
    def _compact_multi_interest_scores(self, seq_output, item_emb=None):
        if not self.use_multi_interest or self.multi_interest_proj is None:
            return None, None
        if item_emb is None:
            item_emb = F.normalize(self.calculate_item_emb(), dim=-1)
        interest_vecs = self.multi_interest_proj(seq_output).view(
            -1, self.num_interests, self.hidden_size
        )
        interest_vecs = F.normalize(interest_vecs, dim=-1)
        scores = torch.einsum("bmh,nh->bmn", interest_vecs, item_emb).max(dim=1).values
        scores[:, 0] = -1e4
        return scores, interest_vecs

    def _cmid_weight(self):
        if not self.use_cmid or self.cmid_fusion_logit is None:
            return self.pq_code_embedding.weight.new_tensor(0.0)
        return self.cmid_max_fusion_weight * torch.sigmoid(self.cmid_fusion_logit)

    def _rank_scores(self, item_seq, item_seq_len):
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = F.normalize(seq_output, dim=-1)
        item_emb = F.normalize(self.calculate_item_emb(), dim=-1)

        scores = torch.matmul(seq_output, item_emb.transpose(0, 1))
        scores[:, 0] = -1e4

        if self.use_cmid:
            cmid_scores, _ = self._compact_multi_interest_scores(seq_output, item_emb)
            if cmid_scores is not None:
                cmid_weight = self._cmid_weight()
                scores = scores + cmid_weight * cmid_scores
                scores[:, 0] = -1e4
                self.last_cmid_fusion_weight = float(cmid_weight.detach().item())

        if self.use_graph_logit and self.graph_alpha > 0:
            scores = self._add_graph_logit_prior(scores, item_seq, item_seq_len)

        if self.use_catf and self.catf_alpha > 0:
            scores = self._add_catf_prior(scores, item_seq, item_seq_len)

        return scores, seq_output

    def _add_catf_prior(self, scores, item_seq, item_seq_len):
        """Calibrate uncertain rankings with a sparse trajectory-flow prior."""
        last_item = self._last_item(item_seq, item_seq_len)
        candidate_count = max(1, scores.size(1) - 1)
        k = min(max(2, self.catf_uncertainty_topk), candidate_count)
        top_scores = scores[:, 1:].topk(k, dim=1).values
        prob = torch.softmax(
            top_scores / max(self.catf_temperature, 1e-6), dim=1
        )
        entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum(dim=1)
        uncertainty = entropy / max(float(torch.log(scores.new_tensor(float(k)))), 1e-8)
        uncertainty = uncertainty.clamp(0.0, 1.0)
        self.last_catf_uncertainty_mean = float(uncertainty.detach().mean().item())

        if self.graph_indices is not None and self.graph_values is not None:
            graph_indices = self.graph_indices[
                last_item.to(self.graph_indices.device)
            ].to(scores.device)
            graph_values = self.graph_values[
                last_item.to(self.graph_values.device)
            ].to(scores.device)
            valid = (
                graph_indices.gt(0)
                & graph_indices.lt(scores.size(1))
                & graph_values.gt(0)
            )
            log_flow = torch.log(graph_values.clamp_min(1e-8))
            count = valid.sum(dim=1, keepdim=True).clamp_min(1)
            mean = (log_flow * valid).sum(dim=1, keepdim=True) / count
            centered = (log_flow - mean) * valid
            std = torch.sqrt(
                (centered.pow(2).sum(dim=1, keepdim=True) / count).clamp_min(1e-8)
            )
            graph_delta = (centered / std).clamp(
                -self.graph_score_clip, self.graph_score_clip
            ) * valid
            graph_delta = graph_delta * uncertainty.unsqueeze(1)
            prior = torch.zeros_like(scores)
            prior.scatter_add_(
                1,
                graph_indices.clamp(min=0, max=scores.size(1) - 1),
                graph_delta,
            )
            return scores + self.catf_alpha * prior

        if self.graph_matrix is None or self.graph_matrix.size(1) != scores.size(1):
            return scores
        graph_prob = self.graph_matrix[
            last_item.to(self.graph_matrix.device)
        ].to(scores.device)
        graph_delta = torch.log(graph_prob.clamp_min(1e-8))
        graph_delta = (
            graph_delta - graph_delta.mean(dim=1, keepdim=True)
        ) / graph_delta.std(dim=1, keepdim=True).clamp_min(1e-8)
        graph_delta = graph_delta.clamp(
            -self.graph_score_clip, self.graph_score_clip
        )
        graph_delta[:, 0] = 0.0
        return scores + self.catf_alpha * uncertainty.unsqueeze(1) * graph_delta

    def _add_graph_logit_prior(self, scores, item_seq, item_seq_len):
        try:
            last_item = self._last_item(item_seq, item_seq_len)

            if self.graph_indices is not None and self.graph_values is not None:
                graph_indices = self.graph_indices[
                    last_item.to(self.graph_indices.device)
                ].to(scores.device)
                graph_values = self.graph_values[
                    last_item.to(self.graph_values.device)
                ].to(scores.device)
                valid = (
                    graph_indices.gt(0)
                    & graph_indices.lt(scores.size(1))
                    & graph_values.gt(0)
                )

                raw_score = torch.log(graph_values.clamp_min(1e-8))
                count = valid.sum(dim=1, keepdim=True).clamp_min(1)
                mean = (raw_score * valid).sum(dim=1, keepdim=True) / count
                centered = (raw_score - mean) * valid
                std = torch.sqrt(
                    (centered.pow(2).sum(dim=1, keepdim=True) / count).clamp_min(1e-8)
                )
                graph_delta = centered / std
                graph_delta = torch.clamp(
                    graph_delta, -self.graph_score_clip, self.graph_score_clip
                ) * valid

                prior = torch.zeros_like(scores)
                prior.scatter_add_(
                    1,
                    graph_indices.clamp(min=0, max=scores.size(1) - 1),
                    graph_delta,
                )
                return scores + self.graph_alpha * prior

            if self.graph_matrix is None or self.graph_matrix.size(1) != scores.size(1):
                return scores

            graph_prob = self.graph_matrix[
                last_item.to(self.graph_matrix.device)
            ].to(scores.device)
            graph_score = torch.log(graph_prob.clamp_min(1e-8))
            graph_score = (
                graph_score - graph_score.mean(dim=-1, keepdim=True)
            ) / (graph_score.std(dim=-1, keepdim=True) + 1e-8)
            graph_score = torch.clamp(
                graph_score, -self.graph_score_clip, self.graph_score_clip
            )
            graph_score[:, 0] = 0.0
            return scores + self.graph_alpha * graph_score
        except Exception:
            return scores

    # =========================================================
    # Losses
    # =========================================================
    def _trajectory_candidate_indices(self, item_seq, item_seq_len, score_size, device):
        if (
            not self.use_tfkd
            or item_seq is None
            or item_seq_len is None
            or self.tfkd_graph_topk <= 0
            or (self.graph_indices is None and self.graph_matrix is None)
        ):
            return None

        last_item = self._last_item(item_seq, item_seq_len)
        if self.graph_indices is not None and self.graph_values is not None:
            graph_idx = self.graph_indices[last_item.to(self.graph_indices.device)].to(device)
            graph_val = self.graph_values[last_item.to(self.graph_values.device)].to(device)
            valid = graph_idx.gt(0) & graph_idx.lt(score_size) & graph_val.gt(0)
            graph_val = graph_val.masked_fill(~valid, -1.0)
            k = min(self.tfkd_graph_topk, graph_idx.size(1))
            if k <= 0:
                return None
            _, positions = graph_val.topk(k, dim=1)
            return graph_idx.gather(1, positions).clamp(min=0, max=score_size - 1)

        if self.graph_matrix is not None and self.graph_matrix.size(1) == score_size:
            graph_prob = self.graph_matrix[last_item.to(self.graph_matrix.device)].to(device)
            graph_prob = graph_prob.clone()
            graph_prob[:, 0] = -1.0
            k = min(self.tfkd_graph_topk, graph_prob.size(1) - 1)
            if k <= 0:
                return None
            return graph_prob.topk(k, dim=1).indices

        return None

    def _dedupe_candidates(self, cand_idx, max_size):
        rows = []
        for row in cand_idx.detach().long().cpu():
            seen = set()
            uniq = []
            for value in row.tolist():
                if 0 <= value < max_size and value not in seen:
                    seen.add(value)
                    uniq.append(value)
            rows.append(uniq)

        width = max(1, max(len(row) for row in rows))
        padded = cand_idx.new_zeros((len(rows), width))
        valid = torch.zeros((len(rows), width), dtype=torch.bool, device=cand_idx.device)
        for i, row in enumerate(rows):
            if row:
                values = cand_idx.new_tensor(row)
                padded[i, : len(row)] = values
                valid[i, : len(row)] = True
        return padded, valid

    def _topk_distillation_loss(
        self,
        student_scores,
        teacher_scores,
        pos_items,
        item_seq=None,
        item_seq_len=None,
    ):
        # Handle possible item-number mismatch safely.
        min_cols = min(student_scores.size(1), teacher_scores.size(1))
        if min_cols <= 1:
            return student_scores.new_tensor(0.0)

        student_scores = student_scores[:, :min_cols]
        teacher_scores = teacher_scores[:, :min_cols]

        valid_mask = pos_items.lt(min_cols)
        if valid_mask.sum() <= 0:
            return student_scores.new_tensor(0.0)

        student_scores = student_scores[valid_mask]
        teacher_scores = teacher_scores[valid_mask]
        pos_items = pos_items[valid_mask]

        k = min(
            self.tfkd_topk if self.use_tfkd else self.distill_topk,
            teacher_scores.size(1) - 1,
        )
        with torch.no_grad():
            masked_teacher = teacher_scores.detach().clone()
            masked_teacher[:, 0] = -1e4
            masked_teacher.scatter_(1, pos_items.unsqueeze(1), -1e4)
            topk_idx = masked_teacher.topk(k, dim=1).indices
            cand_idx = torch.cat([pos_items.unsqueeze(1), topk_idx], dim=1)
            if self.use_tfkd:
                graph_idx = self._trajectory_candidate_indices(
                    item_seq,
                    item_seq_len,
                    teacher_scores.size(1),
                    teacher_scores.device,
                )
                if graph_idx is not None:
                    cand_idx = torch.cat([cand_idx, graph_idx[valid_mask]], dim=1)
            cand_idx, cand_valid = self._dedupe_candidates(cand_idx, teacher_scores.size(1))
            teacher_part = torch.gather(teacher_scores, 1, cand_idx)
            teacher_part = teacher_part.masked_fill(~cand_valid, -1e4)

        student_part = torch.gather(student_scores, 1, cand_idx)
        student_part = student_part.masked_fill(~cand_valid, -1e4)

        return F.kl_div(
            F.log_softmax(student_part / self.distill_temperature, dim=1),
            F.softmax(teacher_part / self.distill_temperature, dim=1),
            reduction="batchmean",
        ) * (self.distill_temperature ** 2)

    def _hard_negative_rank_loss(self, student_scores, pos_items, teacher_scores=None):
        # Use teacher only if dimensions match, otherwise fall back to student.
        if teacher_scores is not None and teacher_scores.size(1) == student_scores.size(1):
            source_scores = teacher_scores.detach()
        else:
            source_scores = student_scores.detach()

        k = min(self.rank_topk, student_scores.size(1) - 1)
        with torch.no_grad():
            masked_scores = source_scores.clone()
            masked_scores[:, 0] = -1e4
            masked_scores.scatter_(1, pos_items.unsqueeze(1), -1e4)
            neg_idx = masked_scores.topk(k, dim=1).indices

        pos_score = torch.gather(student_scores, 1, pos_items.unsqueeze(1))
        neg_score = torch.gather(student_scores, 1, neg_idx)
        return F.softplus(neg_score - pos_score + self.rank_margin).mean()

    def _ndcg_hard_negative_rank_loss(
        self,
        student_scores,
        pos_items,
        teacher_scores,
        item_seq,
        item_seq_len,
    ):
        """Prioritize trajectory-relevant errors that most reduce NDCG."""
        zero = student_scores.new_tensor(0.0)
        score_size = student_scores.size(1)
        if score_size <= 1:
            return zero

        valid_rows = pos_items.gt(0) & pos_items.lt(score_size)
        if valid_rows.sum() <= 0:
            return zero

        if teacher_scores is not None and teacher_scores.size(1) == score_size:
            source_scores = teacher_scores.detach()
        else:
            source_scores = student_scores.detach()

        candidate_topk = min(self.ndcg_rank_candidate_topk, score_size - 1)
        hard_neg_num = min(self.ndcg_rank_hard_neg_num, candidate_topk)
        if candidate_topk <= 0 or hard_neg_num <= 0:
            return zero

        with torch.no_grad():
            masked_source = source_scores.clone()
            masked_source[:, 0] = -1e4
            masked_source.scatter_(1, pos_items.unsqueeze(1), -1e4)
            teacher_idx = masked_source.topk(candidate_topk, dim=1).indices
            candidate_idx = torch.cat([pos_items.unsqueeze(1), teacher_idx], dim=1)

            graph_idx = self._trajectory_candidate_indices(
                item_seq,
                item_seq_len,
                score_size,
                student_scores.device,
            )
            if graph_idx is not None:
                candidate_idx = torch.cat([candidate_idx, graph_idx], dim=1)

            candidate_idx, candidate_valid = self._dedupe_candidates(
                candidate_idx, score_size
            )
            candidate_valid = (
                candidate_valid
                & candidate_idx.gt(0)
                & candidate_idx.ne(pos_items.unsqueeze(1))
            )

            detached_candidate_scores = torch.gather(
                student_scores.detach(), 1, candidate_idx
            ).masked_fill(~candidate_valid, -1e4)
            available = candidate_valid.sum(dim=1)
            selected_num = min(hard_neg_num, candidate_idx.size(1))
            hard_position = detached_candidate_scores.topk(
                selected_num, dim=1
            ).indices
            hard_idx = candidate_idx.gather(1, hard_position)
            hard_valid = candidate_valid.gather(1, hard_position)
            hard_valid = hard_valid & (
                torch.arange(selected_num, device=student_scores.device)
                .unsqueeze(0)
                .lt(available.unsqueeze(1))
            )

        pos_score = torch.gather(student_scores, 1, pos_items.unsqueeze(1))
        neg_score = torch.gather(student_scores, 1, hard_idx)

        with torch.no_grad():
            ranking_scores = torch.cat(
                [pos_score.detach(), neg_score.detach()], dim=1
            )
            ranking_valid = torch.cat(
                [
                    valid_rows.unsqueeze(1),
                    hard_valid,
                ],
                dim=1,
            )
            ranking_scores = ranking_scores.masked_fill(~ranking_valid, -1e4)

            pos_rank = 1 + (
                ranking_scores[:, 1:] > ranking_scores[:, :1]
            ).sum(dim=1)
            neg_rank = 1 + (
                ranking_scores.unsqueeze(1)
                > neg_score.detach().unsqueeze(2)
            ).sum(dim=2)

            cutoff = max(1, self.ndcg_rank_cutoff)
            pos_discount = torch.where(
                pos_rank.le(cutoff),
                1.0 / torch.log2(pos_rank.float() + 1.0),
                torch.zeros_like(pos_rank, dtype=student_scores.dtype),
            )
            neg_discount = torch.where(
                neg_rank.le(cutoff),
                1.0 / torch.log2(neg_rank.float() + 1.0),
                torch.zeros_like(neg_rank, dtype=student_scores.dtype),
            )
            delta_ndcg = (neg_discount - pos_discount.unsqueeze(1)).abs()
            violating = neg_score.detach() + self.ndcg_rank_margin > pos_score.detach()
            pair_valid = hard_valid & valid_rows.unsqueeze(1) & violating
            pair_weight = delta_ndcg * pair_valid.float()
            weight_sum = pair_weight.sum()

            self.last_ndcg_positive_rank = float(
                pos_rank[valid_rows].float().mean().item()
            )
            self.last_ndcg_active_pairs = float(
                pair_valid.float().sum(dim=1)[valid_rows].mean().item()
            )
            if pair_valid.any():
                expanded_pos = pos_score.detach().expand_as(neg_score)
                self.last_ndcg_score_margin = float(
                    (expanded_pos - neg_score.detach())[pair_valid].mean().item()
                )
            else:
                self.last_ndcg_score_margin = 0.0

        if weight_sum <= 0:
            return zero

        pair_loss = F.softplus(
            neg_score - pos_score + self.ndcg_rank_margin
        )
        # Keep the loss scale stable when the number of active pairs changes.
        normalized_weight = pair_weight / weight_sum.clamp_min(1e-8)
        return (pair_loss * normalized_weight).sum()

    def _graph_hard_negative_loss(self, student_scores, item_seq, item_seq_len, pos_items):
        if self.graph_matrix is None and self.graph_indices is None:
            return student_scores.new_tensor(0.0)

        k = min(self.graph_rank_topk, student_scores.size(1) - 1)
        last_item = self._last_item(item_seq, item_seq_len)

        with torch.no_grad():
            if self.graph_indices is not None and self.graph_values is not None:
                sparse_idx = self.graph_indices[last_item.to(self.graph_indices.device)].to(
                    student_scores.device
                )
                sparse_val = self.graph_values[last_item.to(self.graph_values.device)].to(
                    student_scores.device
                )
                valid = (
                    sparse_idx.gt(0)
                    & sparse_idx.lt(student_scores.size(1))
                    & sparse_idx.ne(pos_items.unsqueeze(1))
                    & sparse_val.gt(0)
                )
                sparse_val = sparse_val.masked_fill(~valid, -1.0)
                k_sparse = min(k, sparse_val.size(1))
                neg_prob, positions = sparse_val.topk(k_sparse, dim=1)
                neg_idx = sparse_idx.gather(1, positions)
                valid_mask = neg_prob.gt(0).float()
            elif self.graph_matrix is not None and self.graph_matrix.size(1) == student_scores.size(1):
                graph_prob = self.graph_matrix[last_item.to(self.graph_matrix.device)].to(
                    student_scores.device
                ).clone()
                graph_prob[:, 0] = -1.0
                graph_prob.scatter_(1, pos_items.unsqueeze(1), -1.0)
                neg_prob, neg_idx = graph_prob.topk(k, dim=1)
                valid_mask = neg_prob.gt(0).float()
            else:
                return student_scores.new_tensor(0.0)

        if valid_mask.sum() <= 0:
            return student_scores.new_tensor(0.0)

        pos_score = torch.gather(student_scores, 1, pos_items.unsqueeze(1))
        neg_score = torch.gather(student_scores, 1, neg_idx)
        return (
            F.softplus(neg_score - pos_score + self.rank_margin) * valid_mask
        ).sum() / valid_mask.sum().clamp_min(1.0)

    def _contrastive_loss(self, seq_output, pos_items):
        seq_emb = F.normalize(seq_output, dim=-1)
        pos_emb = F.normalize(self.calculate_item_emb()[pos_items], dim=-1)
        sim_matrix = torch.matmul(seq_emb, pos_emb.transpose(0, 1)) / max(
            self.temperature, 1e-6
        )
        labels = torch.arange(seq_emb.size(0), device=seq_emb.device)
        return F.cross_entropy(sim_matrix, labels)

    def _tfpq_relation_loss(self, teacher, item_seq, item_seq_len):
        """Preserve teacher-side POI relations after PQ, weighted by GTF flow."""
        zero = self.pq_code_embedding.weight.new_tensor(0.0)
        if (
            not self.use_tfpq
            or self.tfpq_loss_weight <= 0
            or teacher is None
            or not hasattr(teacher, "get_candidate_embeddings")
            or (self.graph_indices is None and self.graph_matrix is None)
        ):
            return zero

        last_item = self._last_item(item_seq, item_seq_len)
        score_size = self.n_items

        if self.graph_indices is not None and self.graph_values is not None:
            neighbor_idx = self.graph_indices[
                last_item.to(self.graph_indices.device)
            ].to(last_item.device)
            flow_weight = self.graph_values[
                last_item.to(self.graph_values.device)
            ].to(last_item.device)
            valid = (
                neighbor_idx.gt(0)
                & neighbor_idx.lt(score_size)
                & flow_weight.gt(0)
            )
            ranked_weight = flow_weight.masked_fill(~valid, -1.0)
            k = min(self.tfpq_graph_topk, ranked_weight.size(1))
            if k <= 0:
                return zero
            top_weight, positions = ranked_weight.topk(k, dim=1)
            neighbor_idx = neighbor_idx.gather(1, positions)
            valid = top_weight.gt(0)
            flow_weight = top_weight.clamp_min(0.0)
        elif self.graph_matrix is not None and self.graph_matrix.size(1) == score_size:
            graph_prob = self.graph_matrix[
                last_item.to(self.graph_matrix.device)
            ].to(last_item.device).clone()
            graph_prob[:, 0] = -1.0
            k = min(self.tfpq_graph_topk, graph_prob.size(1) - 1)
            if k <= 0:
                return zero
            flow_weight, neighbor_idx = graph_prob.topk(k, dim=1)
            valid = flow_weight.gt(0)
            flow_weight = flow_weight.clamp_min(0.0)
        else:
            return zero

        with torch.no_grad():
            teacher_item_emb = teacher.get_candidate_embeddings()
            common_items = min(score_size, teacher_item_emb.size(0))
            valid = (
                valid
                & last_item.unsqueeze(1).lt(common_items)
                & neighbor_idx.lt(common_items)
            )
            safe_anchor = last_item.clamp(min=0, max=common_items - 1)
            safe_neighbor = neighbor_idx.clamp(min=0, max=common_items - 1)
            teacher_anchor = teacher_item_emb[safe_anchor]
            teacher_neighbor = teacher_item_emb[safe_neighbor]
            teacher_relation = (
                teacher_anchor.unsqueeze(1) * teacher_neighbor
            ).sum(dim=-1)

        pq_item_emb = F.normalize(self.get_raw_pq_item_embeddings(), dim=-1)
        student_anchor = pq_item_emb[last_item]
        student_neighbor = pq_item_emb[
            neighbor_idx.clamp(min=0, max=score_size - 1)
        ]
        student_relation = (
            student_anchor.unsqueeze(1) * student_neighbor
        ).sum(dim=-1)

        valid_weight = flow_weight * valid.float()
        if valid_weight.sum() <= 0:
            return zero
        valid_weight = valid_weight / valid_weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        per_pair = (student_relation - teacher_relation.detach()).pow(2)
        valid_rows = valid.any(dim=1)
        if valid_rows.sum() <= 0:
            return zero
        return (per_pair * valid_weight).sum(dim=1)[valid_rows].mean()

    def calculate_loss(self, interaction, teacher=None):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        student_scores, seq_output = self._rank_scores(item_seq, item_seq_len)
        ce_loss = self.loss_fct(
            student_scores / max(self.ce_temperature, 1e-6), pos_items
        )

        mi_scores = student_scores
        mi_vecs = None
        if self.use_multi_interest and self.multi_interest_proj is not None:
            item_emb = F.normalize(self.calculate_item_emb(), dim=-1)
            mi_scores, mi_vecs = self._compact_multi_interest_scores(
                seq_output, item_emb
            )

        zero = student_scores.new_tensor(0.0)
        kd_loss = zero
        cmid_kd_loss = zero
        cmid_diversity_loss = zero
        rank_loss = zero
        graph_rank_loss = zero
        mse_loss = zero
        feature_kd_loss = zero
        contrastive_loss = zero
        tfpq_loss = zero
        ndcg_rank_loss = zero
        teacher_scores = None
        teacher_seq_output = None
        teacher_pos_mean = zero
        teacher_top10_mean = zero
        teacher_score_std = zero

        if teacher is not None and (
            self.distill_loss_weight > 0
            or self.mse_loss_weight > 0
            or (self.use_feature_kd and self.feature_kd_loss_weight > 0)
            or (self.use_ndcg_rank and self.ndcg_rank_loss_weight > 0)
        ):
            with torch.no_grad():
                teacher_scores, teacher_seq_output = teacher.get_teacher_outputs(interaction)
                common_cols = min(student_scores.size(1), teacher_scores.size(1))
                valid_diag = pos_items.lt(common_cols)
                if common_cols > 1 and valid_diag.sum() > 0:
                    diag_scores = teacher_scores[valid_diag, :common_cols].detach()
                    diag_pos = pos_items[valid_diag].clamp(min=0, max=common_cols - 1)
                    teacher_pos_mean = torch.gather(
                        diag_scores, 1, diag_pos.unsqueeze(1)
                    ).mean()
                    masked_diag = diag_scores.clone()
                    masked_diag[:, 0] = -1e4
                    diag_k = min(10, common_cols - 1)
                    teacher_top10_mean = masked_diag.topk(diag_k, dim=1).values.mean()
                    teacher_score_std = diag_scores[:, 1:].std()

            if self.distill_loss_weight > 0:
                kd_main = self._topk_distillation_loss(
                    student_scores, teacher_scores, pos_items, item_seq, item_seq_len
                )
                if self.use_cmid:
                    kd_loss = kd_main
                    cmid_kd_loss = self._topk_distillation_loss(
                        mi_scores,
                        teacher_scores,
                        pos_items,
                        item_seq,
                        item_seq_len,
                    )
                else:
                    kd_mi = self._topk_distillation_loss(
                        mi_scores, teacher_scores, pos_items, item_seq, item_seq_len
                    )
                    kd_loss = 0.8 * kd_main + 0.2 * kd_mi

            if (
                self.mse_loss_weight > 0
                and teacher_seq_output is not None
                and self.align_layer is not None
            ):
                mse_loss = F.mse_loss(
                    self.align_layer(seq_output), teacher_seq_output.detach()
                )

            if (
                self.use_feature_kd
                and self.feature_kd_loss_weight > 0
                and teacher_seq_output is not None
                and self.align_layer is not None
            ):
                student_feature = F.normalize(
                    self.align_layer(seq_output), dim=-1
                )
                teacher_feature = F.normalize(
                    teacher_seq_output.detach(), dim=-1
                )
                feature_kd_loss = (
                    1.0 - F.cosine_similarity(
                        student_feature, teacher_feature, dim=-1
                    )
                ).mean()

        if self.use_cmid and mi_vecs is not None and self.num_interests > 1:
            similarity = torch.bmm(mi_vecs, mi_vecs.transpose(1, 2))
            identity = torch.eye(
                self.num_interests,
                device=similarity.device,
                dtype=similarity.dtype,
            ).unsqueeze(0)
            off_diagonal = similarity * (1.0 - identity)
            cmid_diversity_loss = off_diagonal.pow(2).sum() / (
                similarity.size(0) * self.num_interests * (self.num_interests - 1)
            )

        if self.rank_loss_weight > 0:
            rank_loss = self._hard_negative_rank_loss(
                student_scores, pos_items, teacher_scores
            )
        if self.use_ndcg_rank and self.ndcg_rank_loss_weight > 0:
            ndcg_rank_loss = self._ndcg_hard_negative_rank_loss(
                student_scores,
                pos_items,
                teacher_scores,
                item_seq,
                item_seq_len,
            )
        if self.graph_rank_loss_weight > 0:
            graph_rank_loss = self._graph_hard_negative_loss(
                student_scores, item_seq, item_seq_len, pos_items
            )
        if self.contrastive_loss_weight > 0:
            contrastive_loss = self._contrastive_loss(seq_output, pos_items)
        if self.use_tfpq and self.tfpq_loss_weight > 0:
            tfpq_loss = self._tfpq_relation_loss(
                teacher, item_seq, item_seq_len
            )

        total_loss = (
            self.ce_loss_weight * ce_loss
            + self.distill_loss_weight * kd_loss
            + self.cmid_loss_weight * cmid_kd_loss
            + self.cmid_diversity_weight * cmid_diversity_loss
            + self.rank_loss_weight * rank_loss
            + self.graph_rank_loss_weight * graph_rank_loss
            + self.mse_loss_weight * mse_loss
            + self.feature_kd_loss_weight * feature_kd_loss
            + self.contrastive_loss_weight * contrastive_loss
            + self.tfpq_loss_weight * tfpq_loss
            + self.ndcg_rank_loss_weight * ndcg_rank_loss
        )
        self.last_loss_components = {
            "ce": float(ce_loss.detach().cpu()),
            "kd_raw": float(kd_loss.detach().cpu()),
            "kd_weighted": float((self.distill_loss_weight * kd_loss).detach().cpu()),
            "cmid_kd_raw": float(cmid_kd_loss.detach().cpu()),
            "cmid_kd_weighted": float(
                (self.cmid_loss_weight * cmid_kd_loss).detach().cpu()
            ),
            "cmid_diversity_raw": float(cmid_diversity_loss.detach().cpu()),
            "cmid_diversity_weighted": float(
                (self.cmid_diversity_weight * cmid_diversity_loss).detach().cpu()
            ),
            "cmid_fusion_weight": float(self.last_cmid_fusion_weight),
            "tfpq_raw": float(tfpq_loss.detach().cpu()),
            "tfpq_weighted": float((self.tfpq_loss_weight * tfpq_loss).detach().cpu()),
            "feature_kd_raw": float(feature_kd_loss.detach().cpu()),
            "feature_kd_weighted": float(
                (self.feature_kd_loss_weight * feature_kd_loss).detach().cpu()
            ),
            "ndcg_rank_raw": float(ndcg_rank_loss.detach().cpu()),
            "ndcg_rank_weighted": float(
                (self.ndcg_rank_loss_weight * ndcg_rank_loss).detach().cpu()
            ),
            "ndcg_positive_rank": float(self.last_ndcg_positive_rank),
            "ndcg_active_pairs": float(self.last_ndcg_active_pairs),
            "ndcg_score_margin": float(self.last_ndcg_score_margin),
            "teacher_pos_mean": float(teacher_pos_mean.detach().cpu()),
            "teacher_top10_mean": float(teacher_top10_mean.detach().cpu()),
            "teacher_score_std": float(teacher_score_std.detach().cpu()),
            "tfca_gate_mean": float(self.last_tfca_gate_mean),
            "tfca_coverage": float(self.last_tfca_coverage),
            "tfca_residual_weight": float(self.last_tfca_residual_weight),
            "catf_uncertainty_mean": float(self.last_catf_uncertainty_mean),
            "total": float(total_loss.detach().cpu()),
        }
        return total_loss

    # =========================================================
    # Prediction helpers
    # =========================================================
    def _field(self, inter_feat, field):
        try:
            return inter_feat[field]
        except Exception:
            return None

    def _interaction_has(self, interaction, field):
        return hasattr(interaction, "interaction") and field in interaction.interaction

    def _seq_from_interaction(self, interaction):
        if self._interaction_has(interaction, self.ITEM_SEQ) and self._interaction_has(
            interaction, self.ITEM_SEQ_LEN
        ):
            return interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN]

        user_ids = interaction[self.user_id_field].long().to(self.user_seq_bank.device)
        item_seq = torch.zeros(
            (user_ids.size(0), self.max_seq_length),
            dtype=torch.long,
            device=self.user_seq_bank.device,
        )
        item_seq_len = torch.ones(
            user_ids.size(0), dtype=torch.long, device=self.user_seq_bank.device
        )

        valid = user_ids.lt(self.user_seq_bank.size(0))
        if valid.any():
            item_seq[valid] = self.user_seq_bank[user_ids[valid]]
            item_seq_len[valid] = self.user_seq_len_bank[user_ids[valid]]

        device = next(self.parameters()).device
        return item_seq.to(device), item_seq_len.to(device)

    def predict(self, interaction):
        scores = self.full_sort_predict(interaction)
        return torch.gather(
            scores, 1, interaction[self.ITEM_ID].long().unsqueeze(1)
        ).squeeze(1)

    def _local_graph_candidate_prob(self, scores, topk_indices, item_seq, item_seq_len):
        """Return recent-trajectory graph probabilities for dense or sparse GTF."""
        batch_idx = torch.arange(item_seq.size(0), device=item_seq.device)
        gather_index = (item_seq_len - 1).clamp(min=0)
        anchors = torch.stack(
            [
                item_seq[batch_idx, gather_index],
                item_seq[batch_idx, (gather_index - 1).clamp(min=0)],
                item_seq[batch_idx, (gather_index - 2).clamp(min=0)],
            ],
            dim=1,
        )
        weights = scores.new_tensor([0.6, 0.3, 0.1]).view(1, 3, 1)

        if self.graph_indices is not None and self.graph_values is not None:
            graph_idx = self.graph_indices[anchors.to(self.graph_indices.device)].to(
                scores.device
            )
            graph_val = self.graph_values[anchors.to(self.graph_values.device)].to(
                scores.device
            )
            valid = (
                graph_idx.gt(0)
                & graph_idx.lt(scores.size(1))
                & graph_val.gt(0)
            )
            prior = torch.zeros_like(scores)
            prior.scatter_add_(
                1,
                graph_idx.clamp(min=0, max=scores.size(1) - 1).flatten(1),
                (graph_val * weights * valid).flatten(1),
            )
            return prior.gather(1, topk_indices)

        if self.graph_matrix is not None and self.graph_matrix.size(1) == scores.size(1):
            graph_rows = self.graph_matrix[anchors.to(self.graph_matrix.device)].to(
                scores.device
            )
            candidate_idx = topk_indices.unsqueeze(1).expand(-1, 3, -1)
            return (graph_rows.gather(2, candidate_idx) * weights).sum(dim=1)

        return None

    def full_sort_predict(self, interaction):
        """Return full-sort scores with optional local rerank."""
        item_seq, item_seq_len = self._seq_from_interaction(interaction)
        scores, seq_output = self._rank_scores(item_seq, item_seq_len)

        if self.use_local_rerank and not self.training:
            k = min(self.local_rerank_topk, scores.size(1) - 1)
            if k > 0:
                topk_scores, topk_indices = torch.topk(scores, k, dim=1)
                reranked_scores = topk_scores.clone()

                if self.local_graph_beta != 0:
                    graph_prob = self._local_graph_candidate_prob(
                        scores, topk_indices, item_seq, item_seq_len
                    )
                    if graph_prob is not None:
                        graph_score = torch.log1p(self.graph_logit_scale * graph_prob)
                        reranked_scores = (
                            reranked_scores + self.local_graph_beta * graph_score
                        )

                # Repeat rerank. Default beta = 0.02.
                if self.local_repeat_beta != 0:
                    repeat_count = (
                        item_seq.unsqueeze(1) == topk_indices.unsqueeze(2)
                    ).sum(dim=-1).float()
                    repeat_score = repeat_count.clamp(max=1.0)
                    reranked_scores = reranked_scores + self.local_repeat_beta * repeat_score

                # Local interest rerank, default beta = 0.0.
                if (
                    getattr(self, "local_interest_beta", 0.0) != 0
                    and self.use_multi_interest
                    and self.multi_interest_proj is not None
                ):
                    item_emb = F.normalize(self.calculate_item_emb(), dim=-1)
                    mi_vecs = self.multi_interest_proj(seq_output).view(
                        -1, self.num_interests, self.hidden_size
                    )
                    topk_item_emb = item_emb[topk_indices]
                    mi_score = torch.einsum(
                        "bmh,bkh->bkm",
                        F.normalize(mi_vecs, dim=-1),
                        F.normalize(topk_item_emb, dim=-1),
                    ).max(dim=-1).values
                    reranked_scores = reranked_scores + self.local_interest_beta * mi_score

                scores.scatter_(1, topk_indices, reranked_scores)

        return scores
