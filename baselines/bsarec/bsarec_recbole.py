# -*- coding: utf-8 -*-
"""BSARec under the formal EffiPOI/RecBole evaluation protocol.

The encoder follows the official AAAI 2024 implementation: each block mixes
a Fourier attentive-inductive-bias branch with causal self-attention, then
passes the fused states through a point-wise feed-forward network.
"""

import copy

import torch
import torch.nn as nn
from recbole.model.layers import FeedForward, MultiHeadAttention
from recbole.model.sequential_recommender.sasrec import SASRec


class FrequencyLayer(nn.Module):
    """Official BSARec low/high-frequency decomposition."""

    def __init__(self, hidden_size, hidden_dropout_prob, layer_norm_eps, cutoff):
        super().__init__()
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.cutoff = int(cutoff) // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):
        sequence_length = input_tensor.size(1)
        spectrum = torch.fft.rfft(input_tensor, dim=1, norm="ortho")
        low_pass_spectrum = spectrum.clone()
        low_pass_spectrum[:, self.cutoff :, :] = 0
        low_pass = torch.fft.irfft(
            low_pass_spectrum,
            n=sequence_length,
            dim=1,
            norm="ortho",
        )
        high_pass = input_tensor - low_pass
        filtered = low_pass + self.sqrt_beta.square() * high_pass
        return self.layer_norm(self.out_dropout(filtered) + input_tensor)


class BSARecLayer(nn.Module):
    """Fuse the attentive inductive bias and causal self-attention."""

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
        alpha,
        cutoff,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.filter_layer = FrequencyLayer(
            hidden_size,
            hidden_dropout_prob,
            layer_norm_eps,
            cutoff,
        )
        self.attention_layer = MultiHeadAttention(
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
        )

    def forward(self, input_tensor, attention_mask):
        inductive_states = self.filter_layer(input_tensor)
        attentive_states = self.attention_layer(input_tensor, attention_mask)
        return self.alpha * inductive_states + (1.0 - self.alpha) * attentive_states


class BSARecBlock(nn.Module):
    def __init__(
        self,
        n_heads,
        hidden_size,
        inner_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        alpha,
        cutoff,
    ):
        super().__init__()
        self.layer = BSARecLayer(
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
            alpha,
            cutoff,
        )
        self.feed_forward = FeedForward(
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask):
        mixed_states = self.layer(hidden_states, attention_mask)
        return self.feed_forward(mixed_states)


class BSARecEncoder(nn.Module):
    def __init__(
        self,
        n_layers,
        n_heads,
        hidden_size,
        inner_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        alpha,
        cutoff,
    ):
        super().__init__()
        block = BSARecBlock(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
            alpha,
            cutoff,
        )
        self.blocks = nn.ModuleList(
            [copy.deepcopy(block) for _ in range(int(n_layers))]
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_all_encoded_layers=False,
    ):
        all_encoder_layers = []
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BSARecRecBole(SASRec):
    """BSARec with RecBole input, loss, prediction, and full-sort interfaces."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.alpha = float(config["bsarec_alpha"])
        self.cutoff = int(config["bsarec_cutoff"])
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("bsarec_alpha must lie in [0, 1].")
        if self.cutoff < 1:
            raise ValueError("bsarec_cutoff must be a positive integer.")

        self.trm_encoder = BSARecEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            alpha=self.alpha,
            cutoff=self.cutoff,
        )
        self.trm_encoder.apply(self._init_weights)
