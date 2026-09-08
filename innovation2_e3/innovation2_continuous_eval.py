# -*- coding: utf-8 -*-
"""
TF-SID-CPR continuous POI recommendation built on Research-Point-1 Best Student.

This script does not train or modify the Best Student. It evaluates:
  C0: greedy autoregressive rollout
  C1: beam-search rollout
  C2: beam-search + GTF candidate expansion + path-level GTF rerank
  C3: beam-search + GTF + TF-SID consistency score
  C4: beam-search + GTF + TF-SID + repeat penalty

Continuous labels are constructed only from each selected evaluation row:
    full_sequence = item_id_list + item_id
    history = full_sequence[:-horizon]
    future = full_sequence[-horizon:]

The GTF graph is still built exclusively from the training split. Dataset files
and all innovation-point-1 model/config files are protected by read-only guards.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

def _bootstrap_import_root():
    """Find the unchanged EffiPOI root before project-module imports."""
    candidates = []
    for index, value in enumerate(sys.argv):
        if value == "--project_root" and index + 1 < len(sys.argv):
            candidates.append(sys.argv[index + 1])
        elif value.startswith("--project_root="):
            candidates.append(value.split("=", 1)[1])
    if os.environ.get("EFFIPOI_PROJECT_ROOT"):
        candidates.append(os.environ["EFFIPOI_PROJECT_ROOT"])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.extend([script_dir, os.path.dirname(script_dir)])
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(os.path.join(candidate, "teacher.py")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate
    return ""


_BOOTSTRAP_PROJECT_ROOT = _bootstrap_import_root()

import numpy as np
import torch
from recbole.data import data_preparation
from recbole.data.interaction import Interaction
from recbole.utils import init_seed

from config import Config
from data.dataset import StuRecDataset
from student import StuRec
from student_main import (
    build_config_and_data,
    filter_state_dict_for_model,
    safe_torch_load,
)
from teacher import TeaRec
from teacher_main import (
    attach_graph,
    choose_device,
    force_e4_teacher_config,
    get_or_build_e1_graph,
)


GRAPH_BUFFER_KEYS = {
    "graph_matrix",
    "graph_indices",
    "graph_values",
    "graph_matrix_2hop",
}
METRIC_KS = (1, 5, 10)
METHOD_NAMES = {
    "greedy": "C0-Greedy",
    "beam": "C1-Beam",
    "beam_gtf": "C2-Beam+GTF",
    "beam_sid": "C3-Beam+GTF+TF-SID",
    "full": "C4-TF-SID-CPR-Full",
}
DECODER_VERSION = "v6-strict-tf-sid-cpr"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate TF-SID-CPR on 2-step or 3-step continuous POI recommendation."
    )
    parser.add_argument("-d", "--dataset", default="NYC")
    parser.add_argument(
        "--base_model",
        choices=["teacher", "student"],
        default="student",
        help="formal research-point-2 base is Research-Point-1 Ours-Best Student",
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="test",
        help="use valid for parameter selection and test only for final reporting",
    )
    parser.add_argument(
        "-p",
        "--checkpoint",
        default="",
        help="trained Ours-Best StuRec checkpoint; safe defaults are provided for NYC and NYC_STEPS",
    )
    parser.add_argument("--test_file", default="", help="optional explicit test .inter file")
    parser.add_argument("--horizon", type=int, choices=[2, 3], default=2)
    parser.add_argument(
        "--method",
        choices=["greedy", "beam", "beam_gtf", "beam_sid", "full", "all"],
        default="all",
    )
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--expand_topk", type=int, default=30)
    parser.add_argument(
        "--flow_candidate_topk",
        type=int,
        default=10,
        help="extra train-derived GTF neighbors injected into C2/C3/C4 beam candidates",
    )
    parser.add_argument(
        "--flow_candidate_margin",
        type=float,
        default=2.5,
        help="inject a GTF neighbor only if its log-prob is within this margin of the local top candidate",
    )
    parser.add_argument(
        "--base_preserve_k",
        type=int,
        default=0,
        help="diagnostic only: preserve this many candidates from the current best beam history",
    )
    parser.add_argument(
        "--rank_replace_k",
        type=int,
        default=0,
        help="diagnostic only: maximum tail slots in HIT@10 ranking that GTF/TF-SID may replace",
    )
    parser.add_argument(
        "--gtf_beta",
        type=float,
        default=0.05,
        help="path-level GTF weight used only by C2",
    )
    parser.add_argument(
        "--repeat_penalty",
        type=float,
        default=0.10,
        help="path repeat penalty used only by formal C4",
    )
    parser.add_argument(
        "--sid_beta",
        type=float,
        default=0.05,
        help="TF-SID consistency weight used by C3/C4",
    )
    parser.add_argument(
        "--sid_recent",
        type=int,
        default=3,
        help="number of recent route POIs used by TF-SID consistency",
    )
    parser.add_argument(
        "--community_topk",
        type=int,
        default=5,
        help="strongest outgoing GTF edges per POI used for Louvain communities",
    )
    parser.add_argument(
        "--community_resolution",
        type=float,
        default=1.0,
        help="Louvain resolution for Flow-Community Tokens",
    )
    parser.add_argument(
        "--transition_topk",
        type=int,
        default=20,
        help="top-k train-derived GTF neighbors used by Transition Validity",
    )
    parser.add_argument(
        "--transition_eps",
        type=float,
        default=1e-6,
        help="minimum GTF edge value counted by Transition Validity",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 evaluates every eligible test sample",
    )
    parser.add_argument("--save_examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument(
        "--project_root",
        default="",
        help="EffiPOI project root; defaults to the directory containing this script",
    )
    parser.add_argument("--output_dir", default="innovation2_e3_results")
    parser.add_argument("--force_cuda", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--no_base_local_rerank",
        action="store_true",
        help="student-only diagnostic: disable the student's original local rerank",
    )
    return parser.parse_args()


def snapshot_dataset_files(dataset_dir):
    """Capture metadata so innovation-point-2 runs cannot silently change data."""
    snapshot = {}
    for root, _, files in os.walk(dataset_dir):
        for name in files:
            path = os.path.join(root, name)
            stat = os.stat(path)
            snapshot[os.path.relpath(path, dataset_dir)] = (
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
    return snapshot


@contextmanager
def dataset_immutability_guard(dataset_dir):
    before = snapshot_dataset_files(dataset_dir)
    print(
        f"[Data Guard] Read-only contract active for {dataset_dir}; "
        f"tracking {len(before)} files."
    )
    try:
        yield
    finally:
        after = snapshot_dataset_files(dataset_dir)
        if before != after:
            created = sorted(set(after) - set(before))
            deleted = sorted(set(before) - set(after))
            changed = sorted(
                path
                for path in set(before) & set(after)
                if before[path] != after[path]
            )
            raise RuntimeError(
                "Dataset immutability check failed. "
                f"created={created[:5]}, deleted={deleted[:5]}, changed={changed[:5]}"
            )
        print("[Data Guard] Passed: no dataset file was created, deleted, or modified.")


def snapshot_innovation1_files(project_root):
    relative_paths = (
        "teacher.py",
        "teacher_main.py",
        os.path.join("props", "TeaRec.yaml"),
        "student.py",
        "student_main.py",
        os.path.join("props", "StuRec.yaml"),
    )
    snapshot = {}
    for relative_path in relative_paths:
        path = os.path.join(project_root, relative_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Protected innovation-point-1 file missing: {path}")
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        snapshot[relative_path] = digest.hexdigest()
    return snapshot


@contextmanager
def innovation1_immutability_guard(project_root):
    before = snapshot_innovation1_files(project_root)
    print("[Innovation-1 Guard] Six model/config files are protected.")
    try:
        yield
    finally:
        after = snapshot_innovation1_files(project_root)
        if before != after:
            changed = sorted(path for path in before if before[path] != after[path])
            raise RuntimeError(
                "Innovation-point-1 immutability check failed. "
                f"changed={changed}"
            )
        print("[Innovation-1 Guard] Passed: no protected file was modified.")


def resolve_project_root(explicit_path):
    if explicit_path:
        project_root = os.path.abspath(explicit_path)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = (
            script_dir
            if os.path.exists(os.path.join(script_dir, "teacher.py"))
            else os.path.dirname(script_dir)
        )
    if not os.path.exists(os.path.join(project_root, "teacher.py")):
        raise FileNotFoundError(f"Invalid EffiPOI project root: {project_root}")
    return project_root


def resolve_checkpoint(project_root, dataset_name, explicit_path, base_model="teacher"):
    if explicit_path:
        path = explicit_path
        if not os.path.isabs(path):
            path = os.path.join(project_root, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{base_model.title()} checkpoint not found: {path}")
        return os.path.abspath(path)

    known = {}
    if base_model == "teacher":
        known["NYC"] = os.path.join(
            project_root, "saved", "TeaRec-Jun-10-2026_19-08-07.pth"
        )
    else:
        known["NYC"] = os.path.join(
            project_root, "saved", "StuRec-Jun-02-2026_11-08-16.pth"
        )
        known["NYC_STEPS"] = os.path.join(
            project_root, "saved", "StuRec-Jun-14-2026_10-39-45.pth"
        )
    path = known.get(str(dataset_name).upper(), "")
    if path and os.path.exists(path):
        return path
    raise ValueError(
        f"No safe default {base_model} checkpoint is known for this dataset. "
        "Pass it explicitly with -p."
    )


def resolve_interaction_file(project_root, dataset_name, split, explicit_path):
    if explicit_path:
        path = explicit_path
        if not os.path.isabs(path):
            path = os.path.join(project_root, path)
    else:
        path = os.path.join(
            project_root, "data", dataset_name, f"{dataset_name}.{split}.inter"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Interaction file not found: {path}")
    return os.path.abspath(path)


_MISSING = object()


def checkpoint_config_get(config, key, default=_MISSING):
    if config is None:
        return default
    try:
        return config[key]
    except Exception:
        return default


def first_checkpoint_value(config, keys, default=None):
    for key in keys:
        value = checkpoint_config_get(config, key, _MISSING)
        if value is not _MISSING and value is not None:
            return value
    return default


def as_bool(value, default=False):
    if value is _MISSING or value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")
    return bool(value)


def apply_eval_device(kwargs, args):
    if args.cpu:
        kwargs["use_gpu"] = False
        kwargs["gpu_id"] = -1
        kwargs["device"] = "cpu"
    elif args.force_cuda:
        kwargs["use_gpu"] = True
        kwargs["gpu_id"] = 0
        kwargs["device"] = "cuda"
    elif torch.cuda.is_available():
        kwargs["use_gpu"] = True
        kwargs["gpu_id"] = 0
        kwargs["device"] = "cuda"
    else:
        kwargs["use_gpu"] = False
        kwargs["gpu_id"] = -1
        kwargs["device"] = "cpu"
    return kwargs


def infer_num_interests(state_dict, hidden_size, default_value):
    weight = state_dict.get("multi_interest_proj.0.weight")
    if weight is None or not hasattr(weight, "shape") or int(hidden_size) <= 0:
        return int(default_value or 1)
    return max(1, int(weight.shape[0]) // int(hidden_size))


def student_kwargs_from_checkpoint(base_kwargs, checkpoint, state_dict, args):
    """Rebuild the Best Student with the architecture stored in its checkpoint.

    The research-point-2 scorer must be the already trained Best Student. Using
    the current StuRec.yaml blindly can turn on new modules and leave them
    randomly initialized, which makes C0-C4 comparisons noisy and unfair.
    """
    kwargs = dict(base_kwargs)
    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None

    copied_keys = [
        "code_dim",
        "code_cap",
        "temperature",
        "ce_temperature",
        "n_layers",
        "n_heads",
        "hidden_size",
        "inner_size",
        "hidden_dropout_prob",
        "attn_dropout_prob",
        "hidden_act",
        "layer_norm_eps",
        "initializer_range",
        "train_stage",
        "residual_weight",
        "graph_logit_scale",
        "local_rerank_topk",
        "local_graph_beta",
        "local_graph_2hop_ratio",
        "local_repeat_beta",
        "local_interest_beta",
        "mse_loss_weight",
        "teacher_hidden_size",
        "distill_loss_weight",
        "rank_loss_weight",
        "graph_rank_loss_weight",
        "contrastive_loss_weight",
        "distill_temperature",
        "distill_topk",
        "rank_topk",
        "graph_rank_topk",
        "rank_margin",
        "flow_inner_weight",
        "flow_target_weight",
        "flow_smoothing",
        "flow_topk",
        "flow_2hop_weight",
        "graph_score_clip",
    ]
    for key in copied_keys:
        value = checkpoint_config_get(checkpoint_config, key, _MISSING)
        if value is not _MISSING and value is not None:
            kwargs[key] = value

    hidden_size = int(first_checkpoint_value(checkpoint_config, ["hidden_size"], 64))
    has_item_residual = "item_residual_embedding.weight" in state_dict
    has_multi_interest = any(key.startswith("multi_interest_proj.") for key in state_dict)
    has_graph_feature = "graph_proj.weight" in state_dict
    has_graph_buffer = any(key in state_dict for key in GRAPH_BUFFER_KEYS)

    kwargs["use_item_residual"] = has_item_residual
    kwargs["use_multi_interest"] = has_multi_interest
    kwargs["num_interests"] = infer_num_interests(
        state_dict,
        hidden_size,
        first_checkpoint_value(checkpoint_config, ["num_interests"], 1),
    )
    kwargs["use_graph_feature"] = has_graph_feature
    kwargs["use_graph_logit"] = as_bool(
        first_checkpoint_value(checkpoint_config, ["use_graph_logit"], None),
        default=has_graph_feature or has_graph_buffer,
    )
    kwargs["use_local_rerank"] = as_bool(
        first_checkpoint_value(checkpoint_config, ["use_local_rerank"], False)
    )

    graph_alpha = first_checkpoint_value(
        checkpoint_config,
        ["student_graph_alpha", "graph_alpha", "graph_logit_alpha"],
        0.0,
    )
    kwargs["student_graph_alpha"] = graph_alpha
    kwargs["graph_alpha"] = graph_alpha
    kwargs["graph_logit_alpha"] = graph_alpha

    if "graph_score_clip" not in kwargs:
        kwargs["graph_score_clip"] = 2.0
    if "flow_topk" not in kwargs:
        kwargs["flow_topk"] = 0
    if "use_2hop_flow" not in kwargs:
        kwargs["use_2hop_flow"] = as_bool(
            first_checkpoint_value(checkpoint_config, ["use_2hop_flow"], True),
            default=True,
        )

    return apply_eval_device(kwargs, args)


def graph_state_from_checkpoint(state_dict):
    graph_state = {}
    if "graph_matrix" in state_dict:
        graph_state["graph_embedding"] = state_dict["graph_matrix"].detach().cpu().float()
    if "graph_indices" in state_dict:
        graph_state["graph_indices"] = state_dict["graph_indices"].detach().cpu().long()
    if "graph_values" in state_dict:
        graph_state["graph_values"] = state_dict["graph_values"].detach().cpu().float()
    return graph_state


def build_student_data_without_rebuilding_graph(project_root, dataset_name, kwargs, graph_state):
    props = [
        os.path.join(project_root, "props", "StuRec.yaml"),
        os.path.join(project_root, "props", "fintune.yaml"),
    ]
    config = Config(
        model=StuRec,
        dataset=dataset_name,
        config_file_list=props,
        config_dict=kwargs,
    )
    init_seed(config["seed"], config["reproducibility"])
    dataset_obj = StuRecDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)
    for target in (dataset_obj, train_data, valid_data, test_data):
        attach_graph(target, graph_state)
    return config, dataset_obj, train_data, valid_data, test_data


def load_student_model(project_root, dataset_name, checkpoint_path, args):
    base_kwargs = {
        "seed": args.seed,
        "show_progress": False,
    }
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    kwargs = student_kwargs_from_checkpoint(base_kwargs, checkpoint, state_dict, args)
    graph_state = graph_state_from_checkpoint(state_dict)

    if graph_state:
        config, dataset_obj, train_data, valid_data, test_data = (
            build_student_data_without_rebuilding_graph(
                project_root, dataset_name, kwargs, graph_state
            )
        )
        print("[Graph] Reusing GTF graph buffers stored in the Best Student checkpoint.")
    else:
        (
            config,
            dataset_obj,
            train_data,
            valid_data,
            test_data,
            graph_state,
        ) = build_config_and_data(dataset_name, kwargs)
        for target in (dataset_obj, train_data, valid_data, test_data):
            attach_graph(target, graph_state)
        print("[Graph] No checkpoint graph buffer found; using train-derived runtime graph.")

    model = StuRec(config, train_data.dataset)
    filtered, skipped = filter_state_dict_for_model(model, state_dict)
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    del checkpoint
    del state_dict
    gc.collect()

    device = config.device
    model = model.to(device)
    model.eval()
    if args.no_base_local_rerank:
        model.use_local_rerank = False

    print(
        "[Model] Loaded Research-Point-1 Best Student checkpoint. "
        f"matched={len(filtered)}, skipped={len(skipped)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}, device={device}"
    )
    print(
        "[Model] Base student settings: "
        f"graph_feature={model.use_graph_feature}, "
        f"graph_logit={model.use_graph_logit}, "
        f"MIC={model.use_multi_interest}, "
        f"local_rerank={model.use_local_rerank}"
    )
    if skipped:
        print(f"[Model] Skipped student keys example: {skipped[:8]}")
    if missing:
        print(f"[Model] Missing runtime keys example: {missing[:8]}")
    print(
        "[Graph] "
        f"feature_shape={tuple(graph_state['graph_embedding'].shape) if 'graph_embedding' in graph_state else None}, "
        f"sparse_logit={'graph_indices' in graph_state}"
    )
    return model, dataset_obj, config


def load_e3_teacher_model(project_root, dataset_name, checkpoint_path, args):
    kwargs = {
        "seed": args.seed,
        "show_progress": False,
    }
    if args.force_cuda:
        kwargs["force_cuda"] = True
    if args.cpu:
        kwargs["force_cpu"] = True

    # The function name is retained by the original runner, but its enabled
    # modules are exactly the completed E3 Teacher: GTF + MIC + MIFG.
    kwargs = force_e4_teacher_config(kwargs, dataset_name=dataset_name)
    kwargs = choose_device(kwargs)
    props = [
        os.path.join(project_root, "props", "TeaRec.yaml"),
        os.path.join(project_root, "props", "fintune.yaml"),
    ]
    config = Config(
        model=TeaRec,
        dataset=dataset_name,
        config_file_list=props,
        config_dict=kwargs,
    )
    init_seed(config["seed"], config["reproducibility"])

    # StuRecDataset is used only as a read-only loader for aligned PQ codes.
    # TeaRec ignores these codes; TF-SID consumes them during path decoding.
    dataset_obj = StuRecDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)
    graph_state = get_or_build_e1_graph(
        project_root, dataset_name, dataset_obj, train_data, config
    )
    for target in (dataset_obj, train_data, valid_data, test_data):
        attach_graph(target, graph_state)

    model = TeaRec(config, train_data.dataset)
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint
        else checkpoint
    )
    state_dict = {
        key: value for key, value in state_dict.items() if key not in GRAPH_BUFFER_KEYS
    }
    filtered, skipped = filter_state_dict_for_model(model, state_dict)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    missing = [key for key in missing if key not in GRAPH_BUFFER_KEYS]

    del checkpoint
    del state_dict
    gc.collect()

    device = config["device"]
    model = model.to(device)
    model.pq_codes = dataset_obj.pq_codes.long().to(device)
    model.eval()

    print(
        "[Model] Loaded completed E3 Teacher checkpoint. "
        f"matched={len(filtered)}, skipped={len(skipped)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}, device={device}"
    )
    print(
        "[Model] Base E3 settings: "
        f"GTF={model.use_graph_feature and model.use_graph_logit}, "
        f"MIC={model.use_multi_interest}, "
        f"MIFG={model.use_flow_multi_rerank}, "
        f"PQ_shape={tuple(model.pq_codes.shape)}"
    )
    print(
        "[Graph] "
        f"feature_shape={tuple(graph_state['graph_embedding'].shape)}, "
        f"sparse_logit={'graph_indices' in graph_state}"
    )
    return model, dataset_obj, config


def load_base_model(project_root, dataset_name, checkpoint_path, args):
    if args.base_model == "teacher":
        return load_e3_teacher_model(project_root, dataset_name, checkpoint_path, args)
    return load_student_model(project_root, dataset_name, checkpoint_path, args)


def map_item_tokens(dataset_obj, tokens):
    if not tokens:
        return []
    try:
        ids = dataset_obj.token2id("item_id", np.asarray(tokens))
    except Exception:
        ids = [dataset_obj.token2id("item_id", token) for token in tokens]
    ids = np.asarray(ids).reshape(-1).tolist()
    return [int(item_id) for item_id in ids]


def load_continuous_samples(test_file, dataset_obj, horizon, max_samples=0):
    samples = []
    skipped_short = 0
    skipped_unknown = 0

    with open(test_file, "r", encoding="utf-8") as file:
        header = file.readline().rstrip("\n").split("\t")
        try:
            user_col = header.index("user_id:token")
            history_col = header.index("item_id_list:token_seq")
            target_col = header.index("item_id:token")
        except ValueError as error:
            raise ValueError(
                f"Unexpected test header in {test_file}: {header}"
            ) from error

        for row_index, line in enumerate(file, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(user_col, history_col, target_col):
                continue

            history_tokens = parts[history_col].split()
            target_token = parts[target_col].strip()
            full_tokens = history_tokens + ([target_token] if target_token else [])
            if len(full_tokens) <= horizon:
                skipped_short += 1
                continue

            full_ids = map_item_tokens(dataset_obj, full_tokens)
            if len(full_ids) != len(full_tokens) or any(item_id <= 0 for item_id in full_ids):
                skipped_unknown += 1
                continue

            samples.append(
                {
                    "row": row_index,
                    "user": parts[user_col],
                    "history": full_ids[:-horizon],
                    "truth": full_ids[-horizon:],
                }
            )
            if max_samples > 0 and len(samples) >= max_samples:
                break

    print(
        f"[Data] eligible_samples={len(samples)}, skipped_short={skipped_short}, "
        f"skipped_unknown={skipped_unknown}, horizon={horizon}"
    )
    return samples


def histories_to_tensors(histories, max_seq_length, device):
    batch_size = len(histories)
    item_seq = torch.zeros(
        (batch_size, max_seq_length), dtype=torch.long, device=device
    )
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

    for row, history in enumerate(histories):
        truncated = history[-max_seq_length:]
        length = len(truncated)
        if length <= 0:
            raise ValueError("Continuous evaluation received an empty history.")
        item_seq[row, :length] = torch.as_tensor(
            truncated, dtype=torch.long, device=device
        )
        lengths[row] = length
    return item_seq, lengths


@torch.no_grad()
def score_histories(model, histories):
    device = next(model.parameters()).device
    item_seq, lengths = histories_to_tensors(histories, model.max_seq_length, device)
    interaction = Interaction(
        {
            model.ITEM_SEQ: item_seq,
            model.ITEM_SEQ_LEN: lengths,
        }
    )
    scores = model.full_sort_predict(interaction)
    scores[:, 0] = -1e4
    return scores


def unique_items_in_order(items, limit):
    result = []
    seen = set()
    for item in items:
        item = int(item)
        if item not in seen:
            result.append(item)
            seen.add(item)
            if len(result) >= limit:
                break
    return result


@torch.no_grad()
def graph_candidate_items(model, last_item, topk):
    """Return strong train-derived GTF neighbors for candidate augmentation."""
    if topk <= 0:
        return []
    last_item = int(last_item)
    if model.graph_indices is not None and model.graph_values is not None:
        k = min(int(topk), model.graph_indices.size(1))
        items = model.graph_indices[last_item, :k].detach().cpu().tolist()
        values = model.graph_values[last_item, :k].detach().cpu().tolist()
        pairs = [
            (int(item), float(value))
            for item, value in zip(items, values)
            if int(item) > 0 and float(value) > 0.0
        ]
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        return [item for item, _ in pairs[:topk]]

    if (
        model.graph_matrix is not None
        and model.graph_matrix.ndim == 2
        and model.graph_matrix.size(1) == model.n_items
    ):
        row = model.graph_matrix[last_item]
        k = min(int(topk) + 1, row.numel())
        values, items = torch.topk(row, k=k)
        result = []
        for item, value in zip(items.detach().cpu().tolist(), values.detach().cpu().tolist()):
            item = int(item)
            if item <= 0 or item == last_item or float(value) <= 0.0:
                continue
            result.append(item)
            if len(result) >= topk:
                break
        return result

    return []


def select_endpoint_diverse_beams(expanded, beam_size):
    """
    Preserve the strongest path for distinct current endpoints first.

    Ordinary beam search can fill all slots with paths ending at the same POI,
    which collapses both future exploration and the effective HIT@10 candidate
    count. Remaining slots are filled by score after endpoint diversity.
    """
    selected = []
    selected_paths = set()
    seen_endpoints = set()

    for beam in expanded:
        endpoint = int(beam["path"][-1])
        path_key = tuple(beam["path"])
        if endpoint in seen_endpoints:
            continue
        selected.append(beam)
        selected_paths.add(path_key)
        seen_endpoints.add(endpoint)
        if len(selected) >= beam_size:
            return selected

    for beam in expanded:
        path_key = tuple(beam["path"])
        if path_key in selected_paths:
            continue
        selected.append(beam)
        if len(selected) >= beam_size:
            break
    return selected


def rank_unique_endpoints(expanded, limit, score_key="score"):
    """Rank current-step POI candidates by their best path-prefix endpoint score."""
    best_by_item = {}
    for beam in expanded:
        item = int(beam["path"][-1])
        score = float(beam.get(score_key, beam["score"]))
        if item not in best_by_item or score > best_by_item[item]:
            best_by_item[item] = score
    ranked = sorted(best_by_item.items(), key=lambda pair: pair[1], reverse=True)
    return [item for item, _ in ranked[:limit]]


def merge_base_preserving_ranking(
    base_ranking,
    enhanced_ranking,
    limit,
    preserve_k=3,
    replace_k=4,
):
    """
    Keep the Best Student's greedy Top-K as an accuracy floor, while allowing
    path-aware candidates to replace only tail positions.

    This makes C2/C3/C4 a base-preserving decoder instead of a free beam search
    that can accidentally destroy the already strong single-step ranking.
    """
    limit = int(limit)
    if limit <= 0:
        return []
    base = unique_items_in_order(base_ranking, limit)
    if replace_k <= 0 or not enhanced_ranking:
        return base[:limit]

    preserve_k = max(0, min(int(preserve_k), limit, len(base)))
    replace_k = max(0, min(int(replace_k), limit - preserve_k))
    ranking = list(base[:preserve_k])

    for item in enhanced_ranking:
        item = int(item)
        if item <= 0 or item in ranking:
            continue
        ranking.append(item)
        if len(ranking) >= preserve_k + replace_k:
            break

    for item in base:
        if item not in ranking:
            ranking.append(item)
        if len(ranking) >= limit:
            break

    for item in enhanced_ranking:
        item = int(item)
        if item > 0 and item not in ranking:
            ranking.append(item)
        if len(ranking) >= limit:
            break

    return ranking[:limit]


@torch.no_grad()
def graph_raw_values(model, last_item, candidates):
    device = next(model.parameters()).device
    candidate_tensor = torch.as_tensor(candidates, dtype=torch.long, device=device)

    if model.graph_indices is not None and model.graph_values is not None:
        indices = model.graph_indices[int(last_item)].to(device)
        values = model.graph_values[int(last_item)].to(device)
        matches = candidate_tensor.unsqueeze(1).eq(indices.unsqueeze(0))
        raw = torch.where(
            matches,
            values.unsqueeze(0).expand(candidate_tensor.size(0), -1),
            torch.zeros((), dtype=values.dtype, device=device),
        ).max(dim=1).values
        return raw

    if (
        model.graph_matrix is not None
        and model.graph_matrix.ndim == 2
        and model.graph_matrix.size(1) == model.n_items
    ):
        return model.graph_matrix[int(last_item), candidate_tensor].to(device)

    return torch.zeros(candidate_tensor.size(0), dtype=torch.float, device=device)


@torch.no_grad()
def graph_path_bonus(model, last_item, candidates):
    raw = graph_raw_values(model, last_item, candidates)
    scale = float(
        getattr(model, "graph_logit_scale", getattr(model, "flow_rerank_scale", 100.0))
    )
    return torch.log1p(scale * raw.clamp_min(0.0))


def _community_cache_path(output_dir, dataset_name, model, args):
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(
        output_dir,
        (
            f"tf_sid_flow_communities_{dataset_name}_n{model.n_items}"
            f"_top{args.community_topk}_res{args.community_resolution}.pt"
        ),
    )


@torch.no_grad()
def _strong_flow_edges(model, topk):
    """Return a small CPU edge list from the train-derived GTF graph."""
    topk = max(1, int(topk))
    if model.graph_indices is not None and model.graph_values is not None:
        k = min(topk, model.graph_values.size(1))
        edge_values, positions = torch.topk(model.graph_values, k=k, dim=1)
        edge_items = torch.gather(model.graph_indices, 1, positions)
        return edge_items.cpu(), edge_values.cpu()

    if (
        model.graph_matrix is not None
        and model.graph_matrix.ndim == 2
        and model.graph_matrix.size(1) == model.n_items
    ):
        graph = model.graph_matrix
        k = min(topk + 1, graph.size(1))
        edge_values, edge_items = torch.topk(graph, k=k, dim=1)
        row_items = torch.arange(graph.size(0), device=graph.device).unsqueeze(1)
        edge_values = edge_values.masked_fill(edge_items.eq(row_items), 0.0)
        edge_values, positions = torch.topk(edge_values, k=min(topk, k), dim=1)
        edge_items = torch.gather(edge_items, 1, positions)
        return edge_items.cpu(), edge_values.cpu()

    raise ValueError("TF-SID requires a dense or sparse train-derived GTF graph.")


def build_flow_community_tokens(model, dataset_name, args, output_dir):
    """
    Build Flow-Community Tokens with Louvain on a compact train-derived GTF graph.

    Only the strongest outgoing edges are used. This preserves the dominant
    movement structure without materializing the full large graph in NetworkX.
    """
    cache_path = _community_cache_path(output_dir, dataset_name, model, args)
    if os.path.exists(cache_path):
        state = safe_torch_load(cache_path, map_location="cpu")
        tokens = state["community_tokens"] if isinstance(state, dict) else state
        if int(tokens.numel()) == model.n_items:
            print(f"[TF-SID] Loaded Flow-Community Tokens: {cache_path}")
            return tokens.long().to(next(model.parameters()).device)

    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
    except ImportError as error:
        raise ImportError(
            "TF-SID Flow-Community Tokens require networkx with Louvain support."
        ) from error

    edge_items, edge_values = _strong_flow_edges(model, args.community_topk)
    graph = nx.Graph()
    graph.add_nodes_from(range(1, model.n_items))

    for source in range(1, model.n_items):
        for target, weight in zip(
            edge_items[source].tolist(), edge_values[source].tolist()
        ):
            target = int(target)
            weight = float(weight)
            if target <= 0 or target == source or weight <= 0.0:
                continue
            if graph.has_edge(source, target):
                graph[source][target]["weight"] += weight
            else:
                graph.add_edge(source, target, weight=weight)

    print(
        "[TF-SID] Running Louvain on train-derived flow graph: "
        f"nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}"
    )
    communities = louvain_communities(
        graph,
        weight="weight",
        resolution=args.community_resolution,
        seed=args.seed,
    )
    tokens = torch.zeros(model.n_items, dtype=torch.long)
    for community_id, members in enumerate(communities, start=1):
        member_ids = list(members)
        if member_ids:
            tokens[member_ids] = community_id

    sizes = sorted((len(members) for members in communities), reverse=True)
    largest = sizes[0] if sizes else 0
    print(
        "[TF-SID] Built Flow-Community Tokens: "
        f"communities={len(communities)}, largest={largest}, "
        f"largest_ratio={largest / max(1, model.n_items - 1):.4f}"
    )
    torch.save(
        {
            "community_tokens": tokens,
            "community_count": len(communities),
            "community_topk": args.community_topk,
            "community_resolution": args.community_resolution,
            "source": "train-derived GTF Louvain",
        },
        cache_path,
    )
    print(f"[TF-SID] Saved Flow-Community Tokens: {cache_path}")
    return tokens.to(next(model.parameters()).device)


@torch.no_grad()
def sid_consistency_bonus(
    model,
    candidate_items,
    context_items,
):
    """
    Score candidate consistency with the recent route's outgoing TF-SID profile.

    TF-SID = Flow-Community Token + Semantic PQ Tokens + Unique POI Token.
    Instead of rewarding similarity to already visited POIs, this score compares
    candidates with the strongest train-derived flow destinations of recent
    route POIs. This avoids turning semantic consistency into a repeat bonus.
    """
    if not hasattr(model, "tf_sid_communities"):
        raise ValueError("Flow-Community Tokens have not been attached to the model.")

    candidate_items = candidate_items.long()
    context_items = torch.as_tensor(
        context_items, dtype=torch.long, device=candidate_items.device
    )
    context_count = int(context_items.numel())
    if context_count <= 0:
        return torch.zeros(candidate_items.numel(), device=candidate_items.device)

    flow_topk = 5
    if model.graph_indices is not None and model.graph_values is not None:
        k = min(flow_topk, model.graph_indices.size(1))
        target_items = model.graph_indices[context_items, :k]
        target_weights = model.graph_values[context_items, :k].float()
    elif (
        model.graph_matrix is not None
        and model.graph_matrix.ndim == 2
        and model.graph_matrix.size(1) == model.n_items
    ):
        k = min(flow_topk, model.graph_matrix.size(1))
        target_weights, target_items = torch.topk(
            model.graph_matrix[context_items], k=k, dim=1
        )
        target_weights = target_weights.float()
    else:
        return torch.zeros(candidate_items.numel(), device=candidate_items.device)

    valid_targets = target_items.gt(0).float()
    target_weights = target_weights.clamp_min(0.0) * valid_targets
    target_weights = target_weights / target_weights.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-9)
    recency = torch.softmax(
        torch.arange(
            1, context_count + 1, dtype=torch.float, device=candidate_items.device
        ),
        dim=0,
    )
    target_weights = target_weights * recency.unsqueeze(1)

    candidate_communities = model.tf_sid_communities[candidate_items]
    target_communities = model.tf_sid_communities[target_items]
    community_match = candidate_communities[:, None, None].eq(
        target_communities[None, :, :]
    ).float()

    candidate_codes = model.pq_codes[candidate_items]
    target_codes = model.pq_codes[target_items]
    semantic_match = candidate_codes[:, None, None, :].eq(
        target_codes[None, :, :, :]
    ).float().mean(dim=-1)

    unique_poi_match = candidate_items[:, None, None].eq(
        target_items[None, :, :]
    ).float()
    hierarchical_match = (
        0.35 * community_match
        + 0.35 * semantic_match
        + 0.30 * unique_poi_match
    )
    return (hierarchical_match * target_weights.unsqueeze(0)).sum(dim=(1, 2))


def decode_greedy(model, history, horizon, metric_max_k):
    current_history = list(history)
    path = []
    step_rankings = []
    path_score = 0.0

    for _ in range(horizon):
        scores = score_histories(model, [current_history])[0]
        k = min(metric_max_k, scores.numel() - 1)
        top_scores, top_items = torch.topk(scores, k=k)
        ranking = unique_items_in_order(top_items.tolist(), metric_max_k)
        step_rankings.append(ranking)

        selected = int(top_items[0].item())
        path.append(selected)
        current_history.append(selected)
        path_score += float(top_scores[0].item())

    return {
        "path": path,
        "score": path_score,
        "step_rankings": step_rankings,
    }


def decode_beam(
    model,
    history,
    horizon,
    beam_size,
    expand_topk,
    metric_max_k,
    gtf_beta=0.0,
    sid_beta=0.0,
    sid_recent=3,
    repeat_penalty=0.0,
    flow_candidate_topk=0,
    flow_candidate_margin=2.5,
    base_preserve_k=0,
    rank_replace_k=0,
):
    beams = [{"path": [], "history": list(history), "score": 0.0}]
    step_rankings = []

    for _ in range(horizon):
        scores = score_histories(model, [beam["history"] for beam in beams])
        # Beam paths must accumulate comparable conditional log-probabilities.
        # Raw logits from different autoregressive histories are not calibrated
        # against each other and caused later steps to be dominated incorrectly.
        log_probs = torch.log_softmax(scores, dim=-1)
        k = min(expand_topk, log_probs.size(1) - 1)
        top_scores, top_items = torch.topk(log_probs, k=k, dim=1)
        expanded = []

        for beam_index, beam in enumerate(beams):
            candidate_list = top_items[beam_index].detach().cpu().tolist()
            if gtf_beta != 0.0 and flow_candidate_topk > 0:
                top_log_prob = float(top_scores[beam_index, 0].item())
                for graph_item in graph_candidate_items(
                    model, beam["history"][-1], flow_candidate_topk
                ):
                    graph_item = int(graph_item)
                    graph_log_prob = float(log_probs[beam_index, graph_item].item())
                    if graph_log_prob >= top_log_prob - flow_candidate_margin:
                        candidate_list.append(graph_item)
            candidate_list = unique_items_in_order(
                candidate_list, expand_topk + max(0, flow_candidate_topk)
            )
            candidate_items = torch.as_tensor(
                candidate_list, dtype=torch.long, device=log_probs.device
            )
            candidate_scores = log_probs[beam_index, candidate_items]
            k_current = int(candidate_items.numel())
            if gtf_beta != 0.0:
                last_item = beam["history"][-1]
                bonuses = graph_path_bonus(model, last_item, candidate_items)
            else:
                bonuses = torch.zeros_like(candidate_scores)
            if sid_beta != 0.0:
                sid_bonuses = sid_consistency_bonus(
                    model,
                    candidate_items,
                    beam["history"][-max(1, sid_recent):],
                )
            else:
                sid_bonuses = torch.zeros_like(candidate_scores)

            for candidate_index in range(k_current):
                item = int(candidate_items[candidate_index].item())
                local_score = (
                    float(candidate_scores[candidate_index].item())
                    + gtf_beta * float(bonuses[candidate_index].item())
                    + sid_beta * float(sid_bonuses[candidate_index].item())
                )
                score = beam["score"] + local_score
                if repeat_penalty != 0.0 and (
                    item in beam["path"] or item == int(beam["history"][-1])
                ):
                    # Confidence-aware repeat control: preserve a highly ranked
                    # repeated destination when the base model strongly supports
                    # it, while suppressing weak repetitive loops.
                    rank_ratio = candidate_index / max(1, k_current - 1)
                    repeat_scale = 0.20 + 0.80 * rank_ratio
                    penalty = repeat_penalty * repeat_scale
                    score -= penalty
                expanded.append(
                    {
                        "path": beam["path"] + [item],
                        "history": beam["history"] + [item],
                        "score": score,
                        "rank_score": local_score,
                    }
                )

        expanded.sort(key=lambda beam: beam["score"], reverse=True)
        # Rank endpoints before beam truncation so HIT@10 can still expose up to
        # ten distinct step candidates, but score them with path-prefix scores
        # rather than isolated local scores to stay faithful to beam decoding.
        enhanced_ranking = rank_unique_endpoints(
            expanded, metric_max_k, score_key="score"
        )
        if (
            (gtf_beta != 0.0 or sid_beta != 0.0)
            and base_preserve_k > 0
            and rank_replace_k > 0
        ):
            base_scores = score_histories(model, [beams[0]["history"]])[0]
            base_k = min(metric_max_k, base_scores.numel() - 1)
            _, base_top_items = torch.topk(base_scores, k=base_k)
            base_ranking = unique_items_in_order(
                base_top_items.detach().cpu().tolist(), metric_max_k
            )
            step_rankings.append(
                merge_base_preserving_ranking(
                    base_ranking,
                    enhanced_ranking,
                    metric_max_k,
                    preserve_k=base_preserve_k,
                    replace_k=rank_replace_k,
                )
            )
        else:
            step_rankings.append(enhanced_ranking)
        beams = select_endpoint_diverse_beams(expanded, beam_size)

    return {
        "path": beams[0]["path"],
        "score": beams[0]["score"],
        "step_rankings": step_rankings,
        "beam_paths": [
            {"path": beam["path"], "score": beam["score"]} for beam in beams
        ],
    }


def reciprocal_discount(rank):
    return 1.0 / math.log2(rank + 1.0)


class MetricAccumulator:
    def __init__(self, horizon):
        self.horizon = horizon
        self.sample_count = 0
        self.sums = defaultdict(float)
        self.step_sums = [defaultdict(float) for _ in range(horizon)]

    def update(self, truth, result, transition_validity, sid_consistency):
        self.sample_count += 1
        path = result["path"]
        rankings = result["step_rankings"]

        for step, target in enumerate(truth):
            ranking = rankings[step]
            self.step_sums[step]["ranking_size"] += len(ranking)
            self.sums["avg_ranking_size"] += len(ranking) / self.horizon
            for k in METRIC_KS:
                metric_hit = f"hit@{k}"
                metric_ndcg = f"ndcg@{k}"
                try:
                    rank = ranking[:k].index(target) + 1
                except ValueError:
                    rank = 0
                hit = 1.0 if rank > 0 else 0.0
                ndcg = reciprocal_discount(rank) if rank > 0 else 0.0
                self.step_sums[step][metric_hit] += hit
                self.step_sums[step][metric_ndcg] += ndcg
                self.sums[f"avg_{metric_hit}"] += hit / self.horizon
                self.sums[f"avg_{metric_ndcg}"] += ndcg / self.horizon

        truth_set = set(truth)
        path_set = set(path)
        self.sums["route_recall"] += len(truth_set & path_set) / max(1, len(truth_set))
        self.sums["exact_route_rate"] += float(path == truth)
        self.sums["transition_validity"] += transition_validity
        self.sums["sid_consistency"] += sid_consistency
        self.sums["repeat_rate"] += 1.0 - (len(path_set) / max(1, len(path)))

    def result(self):
        if self.sample_count <= 0:
            return {}
        output = {
            key: value / self.sample_count for key, value in sorted(self.sums.items())
        }
        output["sample_count"] = self.sample_count
        for step, step_sum in enumerate(self.step_sums, start=1):
            for key, value in sorted(step_sum.items()):
                output[f"step{step}_{key}"] = value / self.sample_count
        return output


@torch.no_grad()
def is_topk_gtf_edge(model, start, end, topk=20, eps=1e-6):
    start = int(start)
    end = int(end)
    if end <= 0 or end == start:
        return False

    if model.graph_indices is not None and model.graph_values is not None:
        k = min(int(topk), model.graph_indices.size(1))
        if k <= 0:
            return False
        items = model.graph_indices[start, :k]
        values = model.graph_values[start, :k]
        mask = items.eq(end)
        return bool(mask.any().item() and values[mask].max().item() > eps)

    if (
        model.graph_matrix is not None
        and model.graph_matrix.ndim == 2
        and model.graph_matrix.size(1) == model.n_items
    ):
        row = model.graph_matrix[start]
        k = min(int(topk) + 1, row.numel())
        if k <= 0:
            return False
        values, items = torch.topk(row, k=k)
        for item, value in zip(items.detach().cpu().tolist(), values.detach().cpu().tolist()):
            if int(item) == end and float(value) > eps:
                return True
    return False


@torch.no_grad()
def calculate_transition_validity(model, history, predicted_path, topk=20, eps=1e-6):
    if not predicted_path:
        return 0.0
    route = [history[-1]] + list(predicted_path)
    valid = 0
    for start, end in zip(route[:-1], route[1:]):
        valid += int(is_topk_gtf_edge(model, start, end, topk=topk, eps=eps))
    return valid / max(1, len(route) - 1)


@torch.no_grad()
def calculate_sid_consistency(model, history, predicted_path, sid_recent=3):
    if not predicted_path or not hasattr(model, "tf_sid_communities"):
        return 0.0
    current_history = list(history)
    scores = []
    for item in predicted_path:
        candidate_items = torch.as_tensor(
            [int(item)], dtype=torch.long, device=model.pq_codes.device
        )
        context_items = current_history[-max(1, int(sid_recent)) :]
        score = float(
            sid_consistency_bonus(model, candidate_items, context_items)[0].item()
        )
        scores.append(score)
        current_history.append(int(item))
    return sum(scores) / max(1, len(scores))


def item_ids_to_tokens(dataset_obj, ids):
    try:
        tokens = dataset_obj.id2token("item_id", np.asarray(ids))
        return np.asarray(tokens).reshape(-1).tolist()
    except Exception:
        return [str(item_id) for item_id in ids]


def selected_methods(method):
    if method == "all":
        return ["greedy", "beam", "beam_gtf", "beam_sid", "full"]
    return [method]


def evaluate(model, dataset_obj, samples, args):
    methods = selected_methods(args.method)
    accumulators = {
        method: MetricAccumulator(args.horizon) for method in methods
    }
    examples = {method: [] for method in methods}
    metric_max_k = max(METRIC_KS)
    started = time.time()

    for sample_index, sample in enumerate(samples, start=1):
        for method in methods:
            if method == "greedy":
                result = decode_greedy(
                    model, sample["history"], args.horizon, metric_max_k
                )
            else:
                # Incremental ablation chain:
                # C1 Beam -> C2 +GTF -> C3 +TF-SID -> C4 +Repeat Penalty.
                use_gtf = method in ("beam_gtf", "beam_sid", "full")
                use_sid = method in ("beam_sid", "full")
                result = decode_beam(
                    model=model,
                    history=sample["history"],
                    horizon=args.horizon,
                    beam_size=args.beam_size,
                    expand_topk=args.expand_topk,
                    metric_max_k=metric_max_k,
                    gtf_beta=args.gtf_beta if use_gtf else 0.0,
                    sid_beta=args.sid_beta if use_sid else 0.0,
                    sid_recent=args.sid_recent,
                    repeat_penalty=args.repeat_penalty if method == "full" else 0.0,
                    flow_candidate_topk=(
                        args.flow_candidate_topk if use_gtf else 0
                    ),
                    flow_candidate_margin=args.flow_candidate_margin,
                    base_preserve_k=args.base_preserve_k,
                    rank_replace_k=args.rank_replace_k,
                )

            validity = calculate_transition_validity(
                model,
                sample["history"],
                result["path"],
                topk=args.transition_topk,
                eps=args.transition_eps,
            )
            sid_consistency = calculate_sid_consistency(
                model, sample["history"], result["path"], sid_recent=args.sid_recent
            )
            accumulators[method].update(
                sample["truth"], result, validity, sid_consistency
            )

            if len(examples[method]) < args.save_examples:
                examples[method].append(
                    {
                        "row": sample["row"],
                        "user": sample["user"],
                        "history_tail": item_ids_to_tokens(
                            dataset_obj, sample["history"][-5:]
                        ),
                        "truth": item_ids_to_tokens(dataset_obj, sample["truth"]),
                        "prediction": item_ids_to_tokens(dataset_obj, result["path"]),
                        "path_score": result["score"],
                    }
                )

        if sample_index % 25 == 0 or sample_index == len(samples):
            elapsed = time.time() - started
            print(
                f"[Progress] {sample_index}/{len(samples)} samples, "
                f"elapsed={elapsed:.1f}s"
            )

    return (
        {METHOD_NAMES[method]: accumulators[method].result() for method in methods},
        {METHOD_NAMES[method]: examples[method] for method in methods},
    )


def save_results(project_root, checkpoint_path, test_file, results, examples, args):
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"{args.dataset}_{args.split}_h{args.horizon}_{args.method}_{timestamp}"
    json_path = os.path.join(output_dir, f"{stem}.json")
    csv_path = os.path.join(output_dir, f"{stem}.csv")

    payload = {
        "experiment": "TF-SID-CPR continuous evaluation on Research-Point-1 Best Student",
        "decoder_version": DECODER_VERSION,
        "base_model": args.base_model,
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": checkpoint_path,
        "test_file": test_file,
        "horizon": args.horizon,
        "beam_size": args.beam_size,
        "expand_topk": args.expand_topk,
        "flow_candidate_topk": args.flow_candidate_topk,
        "flow_candidate_margin": args.flow_candidate_margin,
        "base_preserve_k": args.base_preserve_k,
        "rank_replace_k": args.rank_replace_k,
        "gtf_beta": args.gtf_beta,
        "sid_beta": args.sid_beta,
        "sid_recent": args.sid_recent,
        "repeat_penalty": args.repeat_penalty,
        "community_topk": args.community_topk,
        "community_resolution": args.community_resolution,
        "transition_validity_rule": "top-k train-derived GTF edge",
        "transition_topk": args.transition_topk,
        "transition_eps": args.transition_eps,
        "base_local_rerank": (
            not args.no_base_local_rerank if args.base_model == "student" else None
        ),
        "results": results,
        "examples": examples,
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "metric", "value"])
        for method, metrics in results.items():
            for metric, value in metrics.items():
                writer.writerow([method, metric, value])

    print(f"[Saved] JSON: {json_path}")
    print(f"[Saved] CSV : {csv_path}")
    return json_path, csv_path


def print_summary(results):
    keys = (
        "avg_hit@1",
        "avg_hit@5",
        "avg_hit@10",
        "avg_ndcg@10",
        "route_recall",
        "transition_validity",
        "sid_consistency",
        "repeat_rate",
    )
    print("\n=== Continuous Recommendation Summary ===")
    for method, metrics in results.items():
        values = ", ".join(
            f"{key}={metrics.get(key, 0.0):.6f}" for key in keys
        )
        print(f"{method}: {values}")


def main():
    args = parse_args()
    if args.force_cuda and args.cpu:
        raise ValueError("Choose only one of --force_cuda and --cpu.")
    if args.beam_size < max(METRIC_KS) and args.method != "greedy":
        raise ValueError(
            f"beam_size must be at least {max(METRIC_KS)} for formal HIT@10 evaluation."
        )
    if args.expand_topk < args.beam_size:
        raise ValueError("expand_topk must be greater than or equal to beam_size.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    project_root = resolve_project_root(args.project_root)
    checkpoint_path = resolve_checkpoint(
        project_root, args.dataset, args.checkpoint, args.base_model
    )
    test_file = resolve_interaction_file(
        project_root, args.dataset, args.split, args.test_file
    )

    print(
        f"[Experiment] base_model={args.base_model}, dataset={args.dataset}, split={args.split}, "
        f"horizon={args.horizon}, method={args.method}"
    )
    print(f"[Experiment] checkpoint={checkpoint_path}")
    print(f"[Experiment] test_file={test_file}")
    print(
        "[Experiment] C0-C4 share the same frozen Research-Point-1 Best Student; "
        "C1/C2/C3/C4 only change the continuous path decoder."
    )

    dataset_dir = os.path.join(project_root, "data", args.dataset)
    with innovation1_immutability_guard(project_root), dataset_immutability_guard(
        dataset_dir
    ):
        model, dataset_obj, _ = load_base_model(
            project_root, args.dataset, checkpoint_path, args
        )
        community_cache_dir = args.output_dir
        if not os.path.isabs(community_cache_dir):
            community_cache_dir = os.path.join(project_root, community_cache_dir)
        model.tf_sid_communities = build_flow_community_tokens(
            model, args.dataset, args, community_cache_dir
        )
        samples = load_continuous_samples(
            test_file, dataset_obj, args.horizon, args.max_samples
        )
        if not samples:
            raise ValueError("No eligible continuous test samples were found.")

        results, examples = evaluate(model, dataset_obj, samples, args)
        print_summary(results)
        save_results(
            project_root, checkpoint_path, test_file, results, examples, args
        )


if __name__ == "__main__":
    main()
