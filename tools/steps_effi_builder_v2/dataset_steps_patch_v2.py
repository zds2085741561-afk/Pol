# -*- coding: utf-8 -*-
"""
dataset_steps_patch_v2.py

动态数据集适配版：
- 自动根据 dataset 名读取：
  data/<dataset>/<dataset>.feat1CLS
  data/<dataset>/<dataset>.feat_token_to_row.json
  data/<dataset>/<dataset>.OPQ128,IVF1,PQ128x4.strict.index
- 根据 RecBole 内部 item token 映射重排 feat1CLS，避免 item 特征错位。
"""

import json
import os

import faiss
import numpy as np
import torch
from recbole.data.dataset import SequentialDataset


class TeaRecDataset(SequentialDataset):
    def __init__(self, config):
        config["plm_size"] = 1664
        super().__init__(config)
        self.plm_size = 1664
        self.dataset_name = config["dataset"]
        self.data_path = config["data_path"]
        self.plm_embedding = self.load_plm_embedding()

    def _get_item_tokens_by_internal_id(self):
        # RecBole commonly uses field2id_token[field] = numpy array/list.
        iid_field = self.iid_field if hasattr(self, "iid_field") else "item_id"
        tokens = None

        if hasattr(self, "field2id_token") and iid_field in self.field2id_token:
            tokens = self.field2id_token[iid_field]
        elif hasattr(self, "_field2id_token") and iid_field in self._field2id_token:
            tokens = self._field2id_token[iid_field]

        if tokens is None:
            return None

        return [str(x) for x in list(tokens)]

    def load_plm_embedding(self):
        feat_path = os.path.join(self.data_path, f"{self.dataset_name}.feat1CLS")
        token_map_path = os.path.join(self.data_path, f"{self.dataset_name}.feat_token_to_row.json")

        print(f"[PLM] Loading feat: {feat_path}")

        if not os.path.exists(feat_path):
            print("[PLM] feat missing, using random fallback")
            return torch.randn(self.item_num, self.plm_size)

        loaded_feat = np.fromfile(feat_path, dtype=np.float32)
        n = loaded_feat.shape[0] // self.plm_size
        raw_feat = loaded_feat[: n * self.plm_size].reshape(-1, self.plm_size).astype(np.float32)
        print(f"[PLM] raw feat rows={raw_feat.shape[0]}, item_num={self.item_num}")

        aligned = np.zeros((self.item_num, self.plm_size), dtype=np.float32)

        tokens = self._get_item_tokens_by_internal_id()
        if tokens is not None and os.path.exists(token_map_path):
            with open(token_map_path, "r", encoding="utf-8") as f:
                token_to_row = json.load(f)

            hit = 0
            for internal_id, token in enumerate(tokens):
                # token 0 / [PAD] stays zero
                if token in token_to_row:
                    row = int(token_to_row[token])
                    if 0 <= row < raw_feat.shape[0]:
                        aligned[internal_id] = raw_feat[row]
                        hit += 1

            print(f"[PLM] aligned by RecBole token map. hit={hit}/{self.item_num}")
            return torch.from_numpy(aligned).float()

        print("[PLM] token map not found, fallback to sequential alignment")
        if raw_feat.shape[0] == self.item_num - 1:
            aligned[1:] = raw_feat
        elif raw_feat.shape[0] >= self.item_num:
            aligned[:] = raw_feat[: self.item_num]
        else:
            aligned[: raw_feat.shape[0]] = raw_feat
            aligned[raw_feat.shape[0]:] = np.random.randn(
                self.item_num - raw_feat.shape[0], self.plm_size
            ).astype(np.float32) * 0.01

        return torch.from_numpy(aligned).float()


class StuRecDataset(TeaRecDataset):
    def __init__(self, config):
        super().__init__(config)
        self.pq_codes = self.load_index()

    def load_index(self):
        index_path = os.path.join(
            self.data_path,
            f"{self.dataset_name}.OPQ128,IVF1,PQ128x4.strict.index",
        )
        print(f"\n[Index] Loading Faiss index: {index_path}")

        if not os.path.exists(index_path):
            print("[Index] index missing, using zero PQ codes")
            return torch.zeros((self.item_num, 128)).long()

        try:
            index = faiss.read_index(index_path)
            print(f"[Index] loaded. ntotal={index.ntotal}, d={index.d}")

            # Encode aligned item embeddings.
            # Skip padding row 0 when index was built without padding.
            x = self.plm_embedding.detach().cpu().numpy().astype(np.float32)
            x_encode = x[1:] if index.ntotal == self.item_num - 1 and x.shape[0] == self.item_num else x

            if hasattr(index, "sa_encode"):
                codes = index.sa_encode(x_encode)
            else:
                if isinstance(index, faiss.IndexPreTransform):
                    vt = faiss.downcast_VectorTransform(index.chain.at(0))
                    x_tmp = vt.apply_py(x_encode)
                    inner = faiss.downcast_index(index.index)
                else:
                    x_tmp = x_encode
                    inner = index

                if hasattr(inner, "pq"):
                    codes = inner.pq.compute_codes(x_tmp)
                else:
                    codes = np.zeros((x_encode.shape[0], 128), dtype=np.uint8)

            codes = torch.from_numpy(codes).long()

            if codes.size(0) == self.item_num - 1:
                pad = torch.zeros((1, codes.size(1)), dtype=torch.long)
                codes = torch.cat([pad, codes], dim=0)

            print(f"[Index] PQ codes shape={tuple(codes.shape)}")
            return codes

        except Exception as e:
            print(f"[Index] extraction failed: {e}")
            return torch.zeros((self.item_num, 128)).long()
