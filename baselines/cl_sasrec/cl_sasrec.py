# -*- coding: utf-8 -*-
"""RecBole/EffiPOI-integrated CL-SASRec baseline.

The baseline keeps the original RecBole data pipeline, Trainer, full-sort
Evaluator, and SASRec next-POI prediction head. It only adds sequence
augmentation based contrastive learning during training.
"""

import random

import torch
import torch.nn.functional as F
from recbole.model.sequential_recommender.sasrec import SASRec


def _parse_aug_types(value):
    if value is None:
        return ["crop", "mask", "reorder"]
    if isinstance(value, str):
        return [x.strip().lower() for x in value.split(",") if x.strip()]
    return [str(x).lower() for x in value]


class CLSASRec(SASRec):
    """SASRec with CL4SRec-style sequence contrastive regularization."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.cl_weight = float(config["cl_weight"]) if "cl_weight" in config else 0.1
        self.cl_temperature = float(config["cl_temperature"]) if "cl_temperature" in config else 0.2
        self.aug_types = _parse_aug_types(config["aug_types"] if "aug_types" in config else None)
        self.crop_ratio = float(config["crop_ratio"]) if "crop_ratio" in config else 0.7
        self.mask_ratio = float(config["mask_ratio"]) if "mask_ratio" in config else 0.2
        self.reorder_ratio = float(config["reorder_ratio"]) if "reorder_ratio" in config else 0.2

    def _crop(self, tokens):
        length = tokens.size(0)
        if length <= 1:
            return tokens
        crop_len = max(1, int(round(length * self.crop_ratio)))
        crop_len = min(crop_len, length)
        start = random.randint(0, length - crop_len)
        return tokens[start : start + crop_len]

    def _mask(self, tokens):
        length = tokens.size(0)
        if length <= 1:
            return tokens
        output = tokens.clone()
        mask_num = max(1, int(round(length * self.mask_ratio)))
        mask_num = min(mask_num, length - 1)
        index = torch.randperm(length, device=tokens.device)[:mask_num]
        output[index] = 0
        return output

    def _reorder(self, tokens):
        length = tokens.size(0)
        if length <= 2:
            return tokens
        output = tokens.clone()
        reorder_len = max(2, int(round(length * self.reorder_ratio)))
        reorder_len = min(reorder_len, length)
        start = random.randint(0, length - reorder_len)
        segment = output[start : start + reorder_len]
        perm = torch.randperm(reorder_len, device=tokens.device)
        output[start : start + reorder_len] = segment[perm]
        return output

    def _augment_one(self, tokens):
        aug_type = random.choice(self.aug_types)
        if aug_type == "crop":
            return self._crop(tokens)
        if aug_type == "mask":
            return self._mask(tokens)
        if aug_type == "reorder":
            return self._reorder(tokens)
        return tokens

    def _augment_batch(self, item_seq, item_seq_len):
        aug_seq = torch.zeros_like(item_seq)
        aug_len = torch.ones_like(item_seq_len)
        max_len = item_seq.size(1)

        for row in range(item_seq.size(0)):
            length = int(item_seq_len[row].item())
            length = max(1, min(length, max_len))
            tokens = item_seq[row, :length]
            view = self._augment_one(tokens)
            view = view[view.gt(0)]
            if view.numel() == 0:
                view = tokens[-1:].clone()
            new_len = min(view.numel(), max_len)
            aug_seq[row, :new_len] = view[:new_len]
            aug_len[row] = new_len

        return aug_seq, aug_len

    def _contrastive_loss(self, item_seq, item_seq_len):
        if self.cl_weight <= 0 or item_seq.size(0) < 2:
            return item_seq.new_tensor(0.0, dtype=torch.float)

        view1, len1 = self._augment_batch(item_seq, item_seq_len)
        view2, len2 = self._augment_batch(item_seq, item_seq_len)

        z1 = F.normalize(self.forward(view1, len1), dim=-1)
        z2 = F.normalize(self.forward(view2, len2), dim=-1)
        logits = torch.matmul(z1, z2.transpose(0, 1)) / self.cl_temperature
        labels = torch.arange(item_seq.size(0), device=item_seq.device)
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.transpose(0, 1), labels)
        )

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)
        logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
        ce_loss = self.loss_fct(logits, pos_items)
        return ce_loss + self.cl_weight * self._contrastive_loss(item_seq, item_seq_len)
