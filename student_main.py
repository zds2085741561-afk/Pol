# -*- coding: utf-8 -*-
"""
StuRec runner for full E4 and pure-student ablations.

Default: full GTF + MIC + MIFG teacher -> full PQ student.
--pure_student: plain PQKD student; optional --use_tfkd adds only TF-KD.
"""

import argparse
import copy
import glob
import os
from logging import getLogger

import torch
import torch.nn.functional as F
from recbole.data import data_preparation
from recbole.utils import init_logger, init_seed, set_color

from config import Config
from data.dataset import StuRecDataset
from student import StuRec
from teacher import TeaRec
from teacher_main import attach_graph, get_or_build_e1_graph
from trainer import StuRecTrainer


def is_large_dataset_name(dataset_name):
    name = str(dataset_name).upper()
    return "STEPS" in name or "MASSIVE" in name


def filter_state_dict_for_model(model, state_dict):
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape):
            filtered[key] = value
        else:
            skipped.append(key)
    return filtered, skipped


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def default_teacher_checkpoint(project_root):
    preferred = os.path.join(project_root, "saved", "TeaRec-Jun-08-2026_18-54-09.pth")
    if os.path.exists(preferred):
        return preferred
    candidates = glob.glob(os.path.join(project_root, "saved", "TeaRec*.pth"))
    if not candidates:
        return ""
    return max(candidates, key=os.path.getmtime)


def enable_teacher_cache(teacher):
    cached_item_emb = {}

    def candidate_embeddings():
        if "value" not in cached_item_emb:
            with torch.no_grad():
                item_emb = teacher.moe_adaptor(teacher.plm_embedding)
                if getattr(teacher, "train_stage", "") == "transductive_ft":
                    item_emb = item_emb + teacher.item_embedding.weight
                cached_item_emb["value"] = F.normalize(item_emb, dim=-1).detach()
        return cached_item_emb["value"]

    def cached_get_teacher_outputs(interaction):
        item_seq = interaction[teacher.ITEM_SEQ]
        item_seq_len = interaction[teacher.ITEM_SEQ_LEN]

        device = next(teacher.parameters()).device
        item_seq = item_seq.to(device)
        item_seq_len = item_seq_len.to(device)
        attention_weights = None

        with torch.no_grad():
            item_emb_list = teacher.moe_adaptor(teacher.plm_embedding[item_seq])
            if hasattr(teacher, "_encode_user_representations"):
                user_repr, fused_output, attention_weights = teacher._encode_user_representations(
                    item_seq, item_emb_list, item_seq_len
                )
                scores = teacher._score_user_representations(
                    user_repr, candidate_embeddings()
                )
            else:
                seq_output = teacher.forward(item_seq, item_emb_list, item_seq_len)
                fused_output, _ = teacher._fuse_graph_feature(
                    seq_output, item_seq, item_seq_len
                )
                fused_output = F.normalize(fused_output, dim=-1)
                scores = torch.matmul(
                    fused_output, candidate_embeddings().transpose(0, 1)
                )

            if hasattr(teacher, "_add_graph_logit_prior"):
                scores = teacher._add_graph_logit_prior(scores, item_seq, item_seq_len)

            if hasattr(teacher, "_add_flow_multi_score"):
                scores = teacher._add_flow_multi_score(
                    scores,
                    item_seq,
                    item_seq_len,
                    candidate_embeddings(),
                    attention_weights,
                )

            scores[:, 0] = -1e4

        return scores, fused_output

    teacher.get_teacher_outputs = cached_get_teacher_outputs
    teacher.get_candidate_embeddings = candidate_embeddings


def force_e4_full_config(kwargs, dataset_name="NYC"):
    kwargs = dict(kwargs)
    is_large = is_large_dataset_name(dataset_name)
    pure_student = bool(kwargs.pop("pure_student", False))
    use_feature_kd = bool(kwargs.pop("use_feature_kd", False))
    use_tfca = bool(kwargs.pop("use_tfca", False))
    use_catf = bool(kwargs.pop("use_catf", False))
    use_cmid = bool(kwargs.pop("use_cmid", False))
    teacher_variant = str(kwargs.get("teacher_variant", "T3")).upper()

    force_cpu = kwargs.pop("force_cpu", False)
    force_cuda = kwargs.pop("force_cuda", False)
    if force_cpu:
        kwargs["use_gpu"] = False
        kwargs["gpu_id"] = -1
    elif force_cuda:
        kwargs["use_gpu"] = True
        kwargs["gpu_id"] = 0
    else:
        kwargs["use_gpu"] = True if torch.cuda.is_available() else False
        if not torch.cuda.is_available():
            kwargs["gpu_id"] = -1

    kwargs["hidden_size"] = 64
    kwargs["teacher_hidden_size"] = 300
    kwargs["n_layers"] = 2
    kwargs["n_heads"] = 2
    kwargs["inner_size"] = 256
    kwargs["hidden_dropout_prob"] = 0.30
    kwargs["attn_dropout_prob"] = 0.30
    kwargs.setdefault("hidden_act", "gelu")
    kwargs.setdefault("layer_norm_eps", 1e-12)
    kwargs.setdefault("initializer_range", 0.02)

    kwargs.setdefault("code_dim", 64)
    kwargs.setdefault("code_cap", 256)
    kwargs.setdefault("temperature", 0.07)

    # Full E4 student representation.
    kwargs["use_item_residual"] = True
    kwargs["residual_weight"] = 0.10
    kwargs["use_multi_interest"] = True
    kwargs["num_interests"] = 4
    kwargs["use_graph_feature"] = True
    kwargs["use_graph_logit"] = True
    kwargs["student_graph_alpha"] = 0.010 if is_large else 0.003
    kwargs["graph_logit_alpha"] = kwargs["student_graph_alpha"]

    # Full E4 teacher GTF settings. Teacher and student share deterministic graph
    # data, while student_graph_alpha keeps their scoring strengths independent.
    kwargs["flow_inner_weight"] = 0.2
    kwargs["flow_target_weight"] = 1.0
    kwargs["flow_smoothing"] = 1e-8
    kwargs["flow_topk"] = 500 if is_large else 0
    kwargs["use_2hop_flow"] = True
    kwargs["flow_2hop_weight"] = 0.10 if is_large else 0.05
    kwargs["graph_alpha"] = 0.02 if is_large else 0.004
    kwargs["graph_score_clip"] = 3.0 if is_large else 2.0
    kwargs["graph_logit_scale"] = 80.0
    kwargs["use_flow_in_train"] = False
    kwargs["target_aware_graph_pooling"] = False

    # Full E4 teacher MIC + local MIFG settings.
    kwargs["teacher_variant"] = teacher_variant
    kwargs["teacher_use_multi_interest"] = True
    kwargs["teacher_num_interests"] = 4
    kwargs["teacher_mic_aggregation"] = "max"
    kwargs["teacher_orth_weight"] = 0.01
    kwargs["teacher_use_flow_multi_rerank"] = True
    kwargs["teacher_flow_multi_topk"] = 20
    kwargs["teacher_flow_multi_beta"] = 0.015
    kwargs["teacher_flow_rerank_recent"] = 1
    kwargs["teacher_flow_rerank_scale"] = 100.0
    kwargs["teacher_flow_rerank_keep"] = 1
    kwargs["teacher_flow_rerank_confidence"] = 0.30
    kwargs["teacher_flow_rerank_primary_bonus"] = 0.0

    # Full E4 student local rerank.
    kwargs["use_local_rerank"] = True
    kwargs["local_rerank_topk"] = 20
    kwargs["local_graph_beta"] = 0.005
    kwargs["local_graph_2hop_ratio"] = 0.0
    kwargs["local_repeat_beta"] = 0.005
    kwargs["local_interest_beta"] = 0.010

    # CE remains the main objective; all E4 auxiliary objectives are enabled.
    kwargs["ce_temperature"] = 0.10
    kwargs["ce_loss_weight"] = 1.0
    kwargs["distill_loss_weight"] = kwargs.get("distill_loss_weight", 0.08)
    kwargs["distill_temperature"] = kwargs.get("distill_temperature", 3.0)
    kwargs["distill_topk"] = 100
    kwargs["use_tfkd"] = kwargs.get("use_tfkd", False)
    kwargs["tfkd_topk"] = kwargs.get("tfkd_topk", 100)
    kwargs["tfkd_graph_topk"] = kwargs.get("tfkd_graph_topk", 50 if is_large else 30)
    kwargs["use_tfca"] = use_tfca
    kwargs["tfca_topk"] = kwargs.get("tfca_topk", 20)
    kwargs["tfca_gate_bias"] = kwargs.get("tfca_gate_bias", -2.0)
    kwargs["tfca_residual_init"] = kwargs.get("tfca_residual_init", 0.0)
    kwargs["use_catf"] = use_catf
    kwargs["catf_alpha"] = kwargs.get("catf_alpha", 0.0)
    kwargs["catf_uncertainty_topk"] = kwargs.get("catf_uncertainty_topk", 100)
    kwargs["catf_temperature"] = kwargs.get("catf_temperature", 1.0)
    kwargs["use_cmid"] = use_cmid
    kwargs["cmid_loss_weight"] = kwargs.get("cmid_loss_weight", 5.0)
    kwargs["cmid_diversity_weight"] = kwargs.get(
        "cmid_diversity_weight", 0.01
    )
    kwargs["cmid_fusion_init"] = kwargs.get("cmid_fusion_init", 0.05)
    kwargs["cmid_max_fusion_weight"] = kwargs.get(
        "cmid_max_fusion_weight", 0.20
    )
    kwargs["use_tfpq"] = kwargs.get("use_tfpq", False)
    kwargs["tfpq_loss_weight"] = kwargs.get("tfpq_loss_weight", 0.5)
    kwargs["tfpq_graph_topk"] = kwargs.get("tfpq_graph_topk", 10)
    kwargs["use_ndcg_rank"] = kwargs.get("use_ndcg_rank", False)
    kwargs["ndcg_rank_loss_weight"] = kwargs.get(
        "ndcg_rank_loss_weight", 0.0
    )
    kwargs["ndcg_rank_candidate_topk"] = kwargs.get(
        "ndcg_rank_candidate_topk", 100
    )
    kwargs["ndcg_rank_hard_neg_num"] = kwargs.get(
        "ndcg_rank_hard_neg_num", 10
    )
    kwargs["ndcg_rank_cutoff"] = kwargs.get("ndcg_rank_cutoff", 10)
    kwargs["ndcg_rank_margin"] = kwargs.get("ndcg_rank_margin", 0.0)

    kwargs["rank_loss_weight"] = 0.02
    kwargs["rank_topk"] = 30
    kwargs["self_rank_topk"] = 30
    kwargs["hard_neg_topk"] = 100
    kwargs["hard_neg_num"] = 10
    kwargs["rank_margin"] = 0.05
    kwargs["graph_rank_loss_weight"] = 0.005
    kwargs["graph_rank_topk"] = 10
    kwargs["graph_hard_neg_topk"] = 50
    kwargs["mse_loss_weight"] = 0.02
    kwargs["use_feature_kd"] = use_feature_kd
    kwargs["feature_kd_loss_weight"] = kwargs.get(
        "feature_kd_loss_weight", 0.05
    ) if use_feature_kd else 0.0
    kwargs["contrastive_loss_weight"] = 0.005

    if pure_student:
        # Pure S0-S3 ablations keep the PQKD architecture unchanged. GTF is
        # used only by the explicitly enabled TF-KD or TF-PQ objective.
        kwargs["use_multi_interest"] = False
        kwargs["num_interests"] = 1
        kwargs["use_graph_feature"] = False
        kwargs["use_graph_logit"] = False
        kwargs["student_graph_alpha"] = 0.0
        kwargs["graph_logit_alpha"] = 0.0
        kwargs["use_local_rerank"] = False
        kwargs["local_graph_beta"] = 0.0
        kwargs["local_graph_2hop_ratio"] = 0.0
        kwargs["local_repeat_beta"] = 0.0
        kwargs["local_interest_beta"] = 0.0
        kwargs["rank_loss_weight"] = 0.0
        kwargs["graph_rank_loss_weight"] = 0.0
        kwargs["mse_loss_weight"] = 0.0
        kwargs["use_feature_kd"] = use_feature_kd
        kwargs["contrastive_loss_weight"] = 0.0
        kwargs["pure_student"] = True
        if use_cmid:
            kwargs["use_multi_interest"] = True
            kwargs["num_interests"] = 4

    kwargs["learning_rate"] = kwargs.get("learning_rate", 0.00003)
    kwargs["weight_decay"] = kwargs.get("weight_decay", 0.00001)
    kwargs["train_batch_size"] = kwargs.get("train_batch_size", 64 if is_large else 128)
    kwargs["eval_batch_size"] = kwargs.get("eval_batch_size", 128 if is_large else 256)
    kwargs["epochs"] = kwargs.get("epochs", 1000000)
    kwargs["eval_step"] = 1
    kwargs["stopping_step"] = kwargs.get("stopping_step", 4)

    kwargs["train_stage"] = "transductive_ft"
    kwargs["dataset_class"] = "StuRecDataset"
    kwargs.setdefault("seed", 2020)
    kwargs.setdefault("reproducibility", True)
    kwargs["rebuild_gtf"] = kwargs.get("rebuild_gtf", False)

    return kwargs


def build_config_and_data(dataset, kwargs):
    project_root = os.path.dirname(os.path.abspath(__file__))
    props = [
        os.path.join(project_root, "props", "StuRec.yaml"),
        os.path.join(project_root, "props", "fintune.yaml"),
    ]

    config = Config(model=StuRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset_obj = StuRecDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    graph_state = get_or_build_e1_graph(project_root, dataset, dataset_obj, train_data, config)
    print(
        "[E4] Shared full GTF graph prepared for teacher and student. "
        f"feature_shape={tuple(graph_state['graph_embedding'].shape)}, "
        f"sparse_logit={'graph_indices' in graph_state}"
    )

    return config, dataset_obj, train_data, valid_data, test_data, graph_state


def load_student_checkpoint_if_needed(model, ckpt_path, device, logger):
    if not ckpt_path:
        return
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Student checkpoint not found: {ckpt_path}")

    logger.info(f"Loading student checkpoint for resume fine-tuning: {ckpt_path}")
    checkpoint = safe_torch_load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    filtered, skipped = filter_state_dict_for_model(model, state_dict)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    logger.info(
        f"Student loaded. matched={len(filtered)}, skipped={len(skipped)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if skipped:
        logger.warning(f"Skipped student keys example: {skipped[:10]}")


def load_teacher_if_needed(config, train_dataset, pretrained_file, device, kwargs, logger):
    if not pretrained_file:
        raise ValueError(
            "E4 full Student requires a trained full Teacher checkpoint for KD and MSE alignment."
        )
    if not os.path.exists(pretrained_file):
        raise FileNotFoundError(f"Teacher checkpoint not found: {pretrained_file}")

    teacher_config = copy.deepcopy(config)
    teacher_config["hidden_size"] = 300
    teacher_config["inner_size"] = kwargs.get("teacher_inner_size", 1200)
    teacher_config["n_layers"] = kwargs.get("teacher_n_layers", 4)
    teacher_config["n_heads"] = kwargs.get("teacher_n_heads", 4)
    teacher_config["hidden_dropout_prob"] = kwargs.get("teacher_hidden_dropout_prob", 0.15)
    teacher_config["attn_dropout_prob"] = kwargs.get("teacher_attn_dropout_prob", 0.15)
    # The teacher is frozen and never calls calculate_loss in the student runner.
    teacher_config["loss_type"] = "BPR"
    teacher_config["n_exps"] = kwargs.get("teacher_n_exps", kwargs.get("n_exps", 4))
    teacher_config["adaptor_dropout_prob"] = kwargs.get("teacher_adaptor_dropout_prob", 0.10)
    teacher_config["plm_size"] = kwargs.get("teacher_plm_size", 1664)
    teacher_config["adaptor_layers"] = [teacher_config["plm_size"], teacher_config["hidden_size"]]
    teacher_config["lambda"] = kwargs.get("teacher_lambda", kwargs.get("lambda", 1e-3))
    teacher_config["train_stage"] = "transductive_ft"
    teacher_variant = str(kwargs.get("teacher_variant", "T3")).upper()
    if teacher_variant in {"E0", "BASE"}:
        teacher_variant = "T0"
    elif teacher_variant in {"E1", "GTF"}:
        teacher_variant = "T1"
    elif teacher_variant in {"E2", "GTF_MIC"}:
        teacher_variant = "T2"
    elif teacher_variant in {"E3", "E4", "FULL", "GTF_MIC_MIFG"}:
        teacher_variant = "T3"
    if teacher_variant not in {"T0", "T1", "T2", "T3"}:
        raise ValueError(f"Unknown teacher_variant={teacher_variant}. Use T0/T1/T2/T3.")

    teacher_uses_gtf = teacher_variant in {"T1", "T2", "T3"}
    teacher_uses_mic = teacher_variant in {"T2", "T3"}
    teacher_uses_mifg = teacher_variant == "T3"

    teacher_config["teacher_variant"] = teacher_variant
    teacher_config["use_graph_feature"] = teacher_uses_gtf
    teacher_config["use_graph_logit"] = teacher_uses_gtf
    teacher_config["graph_alpha"] = kwargs["graph_alpha"]
    teacher_config["graph_score_clip"] = kwargs["graph_score_clip"]
    teacher_config["use_flow_in_train"] = False
    teacher_config["target_aware_graph_pooling"] = False
    teacher_config["use_multi_interest"] = teacher_uses_mic
    teacher_config["num_interests"] = kwargs.get("teacher_num_interests", 4)
    teacher_config["mic_aggregation"] = kwargs.get("teacher_mic_aggregation", "max")
    teacher_config["orth_weight"] = kwargs.get("teacher_orth_weight", 0.01)
    teacher_config["flow_multi_beta"] = (
        kwargs.get("teacher_flow_multi_beta", 0.015) if teacher_uses_mifg else 0.0
    )
    teacher_config["use_flow_multi_rerank"] = (
        kwargs.get("teacher_use_flow_multi_rerank", True) if teacher_uses_mifg else False
    )
    teacher_config["flow_multi_topk"] = kwargs.get("teacher_flow_multi_topk", 20)
    teacher_config["flow_rerank_recent"] = kwargs.get("teacher_flow_rerank_recent", 1)
    teacher_config["flow_rerank_scale"] = kwargs.get("teacher_flow_rerank_scale", 100.0)
    teacher_config["flow_rerank_keep"] = kwargs.get("teacher_flow_rerank_keep", 1)
    teacher_config["flow_rerank_confidence"] = kwargs.get(
        "teacher_flow_rerank_confidence", 0.30
    )
    teacher_config["flow_rerank_primary_bonus"] = kwargs.get(
        "teacher_flow_rerank_primary_bonus", 0.0
    )

    teacher = TeaRec(teacher_config, train_dataset).to(device)
    logger.info(
        set_color(
            f"Loading {teacher_variant} teacher: "
            f"GTF={teacher_uses_gtf}, MIC={teacher_uses_mic}, MIFG={teacher_uses_mifg}; "
            f"checkpoint={pretrained_file}",
            "yellow",
        )
    )

    state = safe_torch_load(pretrained_file, map_location=device)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    # Graph buffers are deterministic runtime data from the formal E4 GTF cache.
    # Preserve the current top500 graph instead of loading checkpoint-era buffers
    # whose topK shape may differ.
    runtime_graph_keys = {"graph_matrix", "graph_indices", "graph_values"}
    checkpoint_graph_keys = [key for key in state_dict if key in runtime_graph_keys]
    state_dict = {
        key: value for key, value in state_dict.items() if key not in runtime_graph_keys
    }
    filtered, skipped = filter_state_dict_for_model(teacher, state_dict)
    missing, unexpected = teacher.load_state_dict(filtered, strict=False)
    reported_missing = [key for key in missing if key not in runtime_graph_keys]
    logger.info(
        f"Teacher loaded. matched={len(filtered)}, skipped={len(skipped)}, "
        f"missing={len(reported_missing)}, unexpected={len(unexpected)}"
    )
    logger.info(
        f"Preserved formal E4 runtime graph buffers: {checkpoint_graph_keys or sorted(runtime_graph_keys)}"
    )
    if skipped:
        logger.warning(f"Skipped teacher keys example: {skipped[:10]}")

    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    enable_teacher_cache(teacher)
    return teacher


def finetune(
    dataset,
    pretrained_file="",
    student_pretrained_file="",
    resume_training_file="",
    **kwargs,
):
    kwargs = force_e4_full_config(kwargs, dataset_name=dataset)
    config, dataset_obj, train_data, valid_data, test_data, graph_state = (
        build_config_and_data(dataset, kwargs)
    )

    logger = getLogger()
    logger.info(config)
    if config["pure_student"]:
        if config["use_tfkd"] and config["use_ndcg_rank"]:
            mode_msg = (
                f"Running TF-KD + NDCG-aware hard-negative ranking "
                f"({config['teacher_variant']} teacher)."
            )
        elif config["use_tfkd"] and config["use_cmid"]:
            mode_msg = (
                f"Running pure S3-CMID ({config['teacher_variant']} teacher): "
                "plain PQKD Student + TF-KD + compact multi-interest distillation."
            )
        elif config["use_tfkd"] and config["use_tfca"]:
            mode_msg = (
                f"Running pure S3-TFCA ({config['teacher_variant']} teacher): "
                "plain PQKD Student + TF-KD + trajectory-flow context adapter."
            )
        elif config["use_tfkd"] and config["use_feature_kd"]:
            mode_msg = (
                f"Running pure S3-F ({config['teacher_variant']} teacher): "
                "plain PQKD Student + TF-KD + cosine Feature KD."
            )
        elif config["use_tfkd"] and config["use_tfpq"]:
            mode_msg = "Running pure S3: plain PQKD Student + TF-KD + TF-PQ."
        elif config["use_tfkd"]:
            mode_msg = "Running pure S1: plain PQKD Student + TF-KD only."
        elif config["use_tfpq"]:
            mode_msg = "Running pure S2: plain PQKD Student + TF-PQ only."
        else:
            mode_msg = "Running pure S0: plain PQKD Student."
    else:
        mode_msg = (
            "Running E4 Full: full GTF+MIC+MIFG teacher -> full PQ student "
            "with CE+KD+all auxiliary losses."
        )
    logger.info(set_color(mode_msg, "yellow"))

    device = config.device
    # Attach the shared GTF graph before constructing either E4 model.
    attach_graph(dataset_obj, graph_state)
    attach_graph(train_data, graph_state)
    attach_graph(valid_data, graph_state)
    attach_graph(test_data, graph_state)
    logger.info(
        set_color(
            f"[E4] Shared full graph attached to teacher and student. "
            f"feature_shape={tuple(graph_state['graph_embedding'].shape)}, "
            f"sparse_logit={'graph_indices' in graph_state}",
            "yellow",
        )
    )

    model = StuRec(config, train_data.dataset).to(device)
    if resume_training_file and student_pretrained_file:
        raise ValueError(
            "Use either --resume_training for exact continuation or "
            "--resume_student for weight-only fine-tuning, not both."
        )
    load_student_checkpoint_if_needed(model, student_pretrained_file, device, logger)

    if kwargs.get("test_only", False):
        if not student_pretrained_file:
            raise ValueError("--test_only requires --resume_student CHECKPOINT")
        trainer = StuRecTrainer(config, model, teacher=None)
        test_result = trainer.evaluate(
            test_data,
            load_best_model=False,
            show_progress=config["show_progress"],
        )
        logger.info(set_color("[TEST ONLY] student result", "yellow") + f": {test_result}")
        return None, test_result

    catf_grid = kwargs.get("catf_tune_grid", "")
    if catf_grid:
        if not config["use_catf"]:
            raise ValueError("--catf_tune_grid requires --use_catf")

        def parse_grid(raw_grid, cast, default):
            if not raw_grid:
                return list(default)
            raw_values = raw_grid if isinstance(raw_grid, (list, tuple)) else (
                str(raw_grid).strip().strip("()[]").split(",")
            )
            return sorted(
                {cast(str(value).strip()) for value in raw_values if str(value).strip()}
            )

        alpha_values = parse_grid(catf_grid, float, [])
        temperature_values = parse_grid(
            kwargs.get("catf_temperature_grid", ""),
            float,
            [config["catf_temperature"]],
        )
        topk_values = parse_grid(
            kwargs.get("catf_topk_grid", ""),
            int,
            [config["catf_uncertainty_topk"]],
        )
        if not alpha_values:
            raise ValueError("CATF alpha grid is empty")
        if any(value <= 0 for value in temperature_values):
            raise ValueError("CATF temperatures must be positive")
        if any(value < 2 for value in topk_values):
            raise ValueError("CATF Top-K values must be at least 2")
        trainer = StuRecTrainer(config, model, teacher=None)
        best_params = None
        best_key = None
        best_valid_result = None
        for topk in topk_values:
            model.catf_uncertainty_topk = topk
            for temperature in temperature_values:
                model.catf_temperature = temperature
                for alpha in alpha_values:
                    model.catf_alpha = alpha
                    valid_result = trainer.evaluate(
                        valid_data,
                        load_best_model=False,
                        show_progress=config["show_progress"],
                    )
                    selection_key = (
                        float(valid_result.get("hit@10", float("-inf"))),
                        float(valid_result.get("ndcg@10", float("-inf"))),
                        -alpha,
                        -temperature,
                        -topk,
                    )
                    logger.info(
                        set_color(
                            "[CATF valid] "
                            f"alpha={alpha:g}, temp={temperature:g}, topk={topk}",
                            "yellow",
                        )
                        + f" {valid_result}"
                    )
                    if best_key is None or selection_key > best_key:
                        best_key = selection_key
                        best_params = (alpha, temperature, topk)
                        best_valid_result = valid_result

        best_alpha, best_temperature, best_topk = best_params
        model.catf_alpha = best_alpha
        model.catf_temperature = best_temperature
        model.catf_uncertainty_topk = best_topk
        logger.info(
            set_color("[CATF selected]", "yellow")
            + " alpha="
            + f"{best_alpha:g}, temp={best_temperature:g}, topk={best_topk}, "
            + f"valid={best_valid_result}"
        )
        if kwargs.get("catf_valid_only", False):
            logger.info(
                set_color("[CATF valid-only] test set was not evaluated", "yellow")
            )
            return best_valid_result, None

        test_result = trainer.evaluate(
            test_data,
            load_best_model=False,
            show_progress=config["show_progress"],
        )
        logger.info(set_color("CATF test result", "yellow") + f": {test_result}")
        return best_valid_result, test_result

    teacher = load_teacher_if_needed(
        config=config,
        train_dataset=train_data.dataset,
        pretrained_file=pretrained_file,
        device=device,
        kwargs=kwargs,
        logger=logger,
    )

    trainer = StuRecTrainer(config, model, teacher=teacher)
    if resume_training_file:
        if not os.path.isfile(resume_training_file):
            raise FileNotFoundError(
                f"Training checkpoint not found: {resume_training_file}"
            )
        logger.info(
            f"Resuming exact training state from: {resume_training_file}"
        )
        checkpoint = safe_torch_load(
            resume_training_file,
            map_location=trainer.device,
        )
        trainer.saved_model_file = str(resume_training_file)
        trainer.start_epoch = checkpoint["epoch"] + 1
        trainer.cur_step = checkpoint["cur_step"]
        trainer.best_valid_score = checkpoint["best_valid_score"]
        trainer.model.load_state_dict(checkpoint["state_dict"])
        trainer.model.load_other_parameter(checkpoint.get("other_parameter"))
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(
            f"Checkpoint loaded. Resume training from epoch {trainer.start_epoch}"
        )
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config["show_progress"],
    )
    if kwargs.get("valid_only", False):
        logger.info(set_color("[VALID ONLY] best valid", "yellow") + f": {best_valid_result}")
        logger.info(
            set_color("[VALID ONLY] test set was not evaluated", "yellow")
        )
        logger.info(f"[VALID ONLY] best checkpoint: {trainer.saved_model_file}")
        return best_valid_result, None

    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=config["show_progress"],
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, default="NYC", help="dataset name")
    parser.add_argument("-p", type=str, default="", help="trained E4/E3 TeaRec checkpoint used by the frozen full teacher")
    parser.add_argument("--resume_student", type=str, default="", help="StuRec checkpoint path")
    parser.add_argument(
        "--resume_training",
        type=str,
        default="",
        help="resume model, optimizer, epoch and early-stopping state exactly",
    )
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--force_cuda", action="store_true", help="Force CUDA mode")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--flow_topk", type=int, default=None, help="override E4 GTF sparse topK")
    parser.add_argument("--graph_alpha", type=float, default=None, help="override E4 teacher graph score strength")
    parser.add_argument("--flow_2hop_weight", type=float, default=None, help="override E4 two-hop GTF weight")
    parser.add_argument("--graph_score_clip", type=float, default=None, help="override E4 graph score clip")
    parser.add_argument("--flow_inner_weight", type=float, default=None, help="override E4 inner-flow weight")
    parser.add_argument("--flow_target_weight", type=float, default=None, help="override E4 target-flow weight")
    parser.add_argument("--use_tfkd", action="store_true", help="enable S1 trajectory-flow candidate distillation")
    parser.add_argument("--use_tfca", action="store_true", help="enable S3 trajectory-flow context adapter")
    parser.add_argument("--use_catf", action="store_true", help="enable confidence-adaptive trajectory-flow score calibration")
    parser.add_argument("--use_feature_kd", action="store_true", help="enable cosine feature distillation for the pure student")
    parser.add_argument("--use_cmid", action="store_true", help="enable compact multi-interest distillation and score fusion")
    parser.add_argument("--use_tfpq", action="store_true", help="enable S2 trajectory-flow-preserving PQ relation loss")
    parser.add_argument("--use_ndcg_rank", action="store_true", help="enable NDCG-weighted trajectory hard-negative ranking")
    parser.add_argument("--pure_student", action="store_true", help="disable full-student GTF/MIC/rerank/aux losses for clean S0/S1 ablations")
    parser.add_argument("--valid_only", action="store_true", help="train/select on validation without evaluating test")
    parser.add_argument("--test_only", action="store_true", help="evaluate --resume_student on test without training")
    parser.add_argument("--epochs", type=int, default=None, help="override maximum training epochs")
    parser.add_argument("--stopping_step", type=int, default=None, help="early stop after this many non-improving evaluations")
    parser.add_argument("--train_batch_size", type=int, default=None, help="override training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="override evaluation batch size")
    parser.add_argument("--distill_loss_weight", type=float, default=None, help="override KD / TF-KD loss weight")
    parser.add_argument("--distill_temperature", type=float, default=None, help="override KD / TF-KD temperature")
    parser.add_argument("--feature_kd_loss_weight", type=float, default=None, help="weight of cosine feature distillation")
    parser.add_argument("--cmid_loss_weight", type=float, default=None, help="weight of compact multi-interest distillation")
    parser.add_argument("--cmid_diversity_weight", type=float, default=None, help="orthogonality weight for compact interests")
    parser.add_argument("--cmid_fusion_init", type=float, default=None, help="initial compact multi-interest score weight")
    parser.add_argument("--cmid_max_fusion_weight", type=float, default=None, help="maximum learnable compact multi-interest score weight")
    parser.add_argument("--tfkd_topk", type=int, default=None, help="teacher topK used by TF-KD")
    parser.add_argument("--tfkd_graph_topk", type=int, default=None, help="GTF reachable candidates used by TF-KD")
    parser.add_argument("--tfca_topk", type=int, default=None, help="GTF neighbors aggregated by the S3 context adapter")
    parser.add_argument("--tfca_gate_bias", type=float, default=None, help="initial bias of the S3 context gate")
    parser.add_argument("--tfca_residual_init", type=float, default=None, help="initial TFCA residual scale; zero exactly preserves the warm-start model")
    parser.add_argument("--learning_rate", type=float, default=None, help="override optimizer learning rate")
    parser.add_argument("--catf_tune_grid", type=str, default="", help="comma-separated CATF alphas")
    parser.add_argument("--catf_temperature_grid", type=str, default="", help="comma-separated CATF temperatures")
    parser.add_argument("--catf_topk_grid", type=str, default="", help="comma-separated CATF uncertainty Top-K values")
    parser.add_argument("--catf_valid_only", action="store_true", help="tune CATF on validation without evaluating test")
    parser.add_argument("--catf_uncertainty_topk", type=int, default=None, help="Top-K scores used to estimate CATF entropy")
    parser.add_argument("--catf_temperature", type=float, default=None, help="temperature used by CATF uncertainty estimation")
    parser.add_argument("--tfpq_loss_weight", type=float, default=None, help="weight of the TF-PQ relation-preserving loss")
    parser.add_argument("--tfpq_graph_topk", type=int, default=None, help="GTF neighbors per anchor used by TF-PQ")
    parser.add_argument("--ndcg_rank_loss_weight", type=float, default=None, help="weight of NDCG-aware hard-negative ranking")
    parser.add_argument("--ndcg_rank_candidate_topk", type=int, default=None, help="teacher candidates used by NDCG-aware ranking")
    parser.add_argument("--ndcg_rank_hard_neg_num", type=int, default=None, help="hard negatives per sample for NDCG-aware ranking")
    parser.add_argument("--ndcg_rank_cutoff", type=int, default=None, help="NDCG cutoff used to weight pairwise errors")
    parser.add_argument("--ndcg_rank_margin", type=float, default=None, help="score margin for active NDCG ranking pairs")
    parser.add_argument("--valid_metric", type=str, default=None, help="validation metric used for checkpoint selection")
    parser.add_argument("--rebuild_gtf", action="store_true", help="Force rebuild cached GTF graph")
    parser.add_argument("--teacher_variant", type=str, default="T3", choices=["T0", "T1", "T2", "T3", "E0", "E1", "E2", "E3", "E4", "BASE", "GTF", "GTF_MIC", "FULL", "GTF_MIC_MIFG"], help="teacher ablation variant used to build the frozen TeaRec shell")
    args, _ = parser.parse_known_args()

    extra_kwargs = {"seed": args.seed}
    if args.force_cuda:
        extra_kwargs["force_cuda"] = True
    if args.cpu:
        extra_kwargs["force_cpu"] = True
    if args.flow_topk is not None:
        extra_kwargs["flow_topk"] = args.flow_topk
    if args.graph_alpha is not None:
        extra_kwargs["graph_alpha"] = args.graph_alpha
    if args.flow_2hop_weight is not None:
        extra_kwargs["flow_2hop_weight"] = args.flow_2hop_weight
    if args.graph_score_clip is not None:
        extra_kwargs["graph_score_clip"] = args.graph_score_clip
    if args.flow_inner_weight is not None:
        extra_kwargs["flow_inner_weight"] = args.flow_inner_weight
    if args.flow_target_weight is not None:
        extra_kwargs["flow_target_weight"] = args.flow_target_weight
    if args.use_tfkd:
        extra_kwargs["use_tfkd"] = True
    if args.use_tfca:
        extra_kwargs["use_tfca"] = True
    if args.use_catf:
        extra_kwargs["use_catf"] = True
    if args.use_feature_kd:
        extra_kwargs["use_feature_kd"] = True
    if args.use_cmid:
        extra_kwargs["use_cmid"] = True
    if args.use_tfpq:
        extra_kwargs["use_tfpq"] = True
    if args.use_ndcg_rank:
        extra_kwargs["use_ndcg_rank"] = True
    if args.pure_student:
        extra_kwargs["pure_student"] = True
    if args.valid_only:
        extra_kwargs["valid_only"] = True
    if args.test_only:
        extra_kwargs["test_only"] = True
    if args.epochs is not None:
        extra_kwargs["epochs"] = args.epochs
    if args.stopping_step is not None:
        extra_kwargs["stopping_step"] = args.stopping_step
    if args.train_batch_size is not None:
        extra_kwargs["train_batch_size"] = args.train_batch_size
    if args.eval_batch_size is not None:
        extra_kwargs["eval_batch_size"] = args.eval_batch_size
    if args.distill_loss_weight is not None:
        extra_kwargs["distill_loss_weight"] = args.distill_loss_weight
    if args.distill_temperature is not None:
        extra_kwargs["distill_temperature"] = args.distill_temperature
    if args.feature_kd_loss_weight is not None:
        extra_kwargs["feature_kd_loss_weight"] = args.feature_kd_loss_weight
    if args.cmid_loss_weight is not None:
        extra_kwargs["cmid_loss_weight"] = args.cmid_loss_weight
    if args.cmid_diversity_weight is not None:
        extra_kwargs["cmid_diversity_weight"] = args.cmid_diversity_weight
    if args.cmid_fusion_init is not None:
        extra_kwargs["cmid_fusion_init"] = args.cmid_fusion_init
    if args.cmid_max_fusion_weight is not None:
        extra_kwargs["cmid_max_fusion_weight"] = args.cmid_max_fusion_weight
    if args.tfkd_topk is not None:
        extra_kwargs["tfkd_topk"] = args.tfkd_topk
    if args.tfkd_graph_topk is not None:
        extra_kwargs["tfkd_graph_topk"] = args.tfkd_graph_topk
    if args.tfca_topk is not None:
        extra_kwargs["tfca_topk"] = args.tfca_topk
    if args.tfca_gate_bias is not None:
        extra_kwargs["tfca_gate_bias"] = args.tfca_gate_bias
    if args.tfca_residual_init is not None:
        extra_kwargs["tfca_residual_init"] = args.tfca_residual_init
    if args.learning_rate is not None:
        extra_kwargs["learning_rate"] = args.learning_rate
    if args.catf_tune_grid:
        extra_kwargs["catf_tune_grid"] = args.catf_tune_grid
    if args.catf_temperature_grid:
        extra_kwargs["catf_temperature_grid"] = args.catf_temperature_grid
    if args.catf_topk_grid:
        extra_kwargs["catf_topk_grid"] = args.catf_topk_grid
    if args.catf_valid_only:
        extra_kwargs["catf_valid_only"] = True
    if args.catf_uncertainty_topk is not None:
        extra_kwargs["catf_uncertainty_topk"] = args.catf_uncertainty_topk
    if args.catf_temperature is not None:
        extra_kwargs["catf_temperature"] = args.catf_temperature
    if args.tfpq_loss_weight is not None:
        extra_kwargs["tfpq_loss_weight"] = args.tfpq_loss_weight
    if args.tfpq_graph_topk is not None:
        extra_kwargs["tfpq_graph_topk"] = args.tfpq_graph_topk
    if args.ndcg_rank_loss_weight is not None:
        extra_kwargs["ndcg_rank_loss_weight"] = args.ndcg_rank_loss_weight
    if args.ndcg_rank_candidate_topk is not None:
        extra_kwargs["ndcg_rank_candidate_topk"] = args.ndcg_rank_candidate_topk
    if args.ndcg_rank_hard_neg_num is not None:
        extra_kwargs["ndcg_rank_hard_neg_num"] = args.ndcg_rank_hard_neg_num
    if args.ndcg_rank_cutoff is not None:
        extra_kwargs["ndcg_rank_cutoff"] = args.ndcg_rank_cutoff
    if args.ndcg_rank_margin is not None:
        extra_kwargs["ndcg_rank_margin"] = args.ndcg_rank_margin
    if args.valid_metric is not None:
        extra_kwargs["valid_metric"] = args.valid_metric
    if args.rebuild_gtf:
        extra_kwargs["rebuild_gtf"] = True
    extra_kwargs["teacher_variant"] = args.teacher_variant

    finetune(
        args.d,
        pretrained_file=args.p,
        student_pretrained_file=args.resume_student,
        resume_training_file=args.resume_training,
        **extra_kwargs,
    )
