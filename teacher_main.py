# -*- coding: utf-8 -*-
"""
teacher_main.py

E4 Full Teacher GTF + MIC + MIFG Runner
==================================

用途：
- 在 E0 Teacher baseline 基础上打开 GTF 全局轨迹流图整包
- 使用图特征融合、二跳轨迹流图和 graph logit 先验
- 不使用训练期 flow 加分和 target-aware pooling
- 不使用 MIFG / flow_multi rerank
- 默认正常训练：epochs=300, lr=0.001
- 默认冻结整个 encoder，便于和 best teacher 控制变量对比
"""

import argparse
import os
from collections import defaultdict
from logging import getLogger
import torch

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color

from teacher import TeaRec
from data.dataset import TeaRecDataset


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def cuda_is_usable():
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.zeros(1, device="cuda")
        x = x + 1
        torch.cuda.synchronize()
        return True
    except Exception as e:
        print(f"[CUDA] CUDA detected but unusable, fallback to CPU. Reason: {e}")
        return False


def is_large_dataset_name(dataset_name):
    name = str(dataset_name).upper()
    return "STEPS" in name or "MASSIVE" in name


def force_e4_teacher_config(kwargs, dataset_name="NYC"):
    """Force E4 Full Teacher: GTF + MIC + local MIFG rerank."""
    kwargs = dict(kwargs)
    is_large = is_large_dataset_name(dataset_name)

    # ---------- 训练设置 ----------
    kwargs["epochs"] = kwargs.get("epochs", 300)
    kwargs["learning_rate"] = kwargs.get("learning_rate", 0.001)
    kwargs["stopping_step"] = kwargs.get("stopping_step", 4)
    kwargs["weight_decay"] = kwargs.get("weight_decay", 0.0)
    kwargs["train_batch_size"] = kwargs.get("train_batch_size", 128)
    kwargs["eval_batch_size"] = kwargs.get("eval_batch_size", 256)

    # ---------- Teacher 结构，保持和主实验一致 ----------
    kwargs["n_layers"] = 4
    kwargs["n_heads"] = 4
    kwargs["hidden_size"] = 300
    kwargs["inner_size"] = 1200
    kwargs["hidden_dropout_prob"] = 0.15
    kwargs["attn_dropout_prob"] = 0.15
    kwargs["hidden_act"] = "gelu"
    kwargs["layer_norm_eps"] = 1e-12
    kwargs["initializer_range"] = 0.02
    kwargs["loss_type"] = "CE"

    # ---------- PLM / MoE 设置，保持和主实验一致 ----------
    kwargs["item_drop_ratio"] = 0.2
    kwargs["item_drop_coefficient"] = 0.9
    kwargs["lambda"] = 1e-3
    kwargs["plm_suffix"] = "feat1CLS"
    kwargs["plm_suffix_aug"] = "feat2CLS"
    kwargs["train_stage"] = "transductive_ft"
    kwargs["plm_size"] = 1664

    kwargs["adaptor_dropout_prob"] = 0.10
    kwargs["adaptor_layers"] = [1664, 300]
    kwargs["temperature"] = 0.07
    kwargs["n_exps"] = 4
    kwargs["sinkhorn_iter"] = 3
    kwargs["reassign_steps"] = 5
    kwargs["stable_moe"] = False

    # ---------- E1 核心：GTF 全局轨迹流图整包 ----------
    kwargs["use_graph_feature"] = True

    kwargs["flow_inner_weight"] = kwargs.get("flow_inner_weight", 0.2)
    kwargs["flow_target_weight"] = kwargs.get("flow_target_weight", 1.0)
    kwargs["flow_smoothing"] = kwargs.get("flow_smoothing", 1e-8)
    # Full NYC keeps dense GTF. NYC_STEPS/MASSIVE uses stronger sparse GTF.
    kwargs["flow_topk"] = kwargs.get("flow_topk", 500 if is_large else 0)
    kwargs["use_2hop_flow"] = True
    kwargs["flow_2hop_weight"] = kwargs.get("flow_2hop_weight", 0.10 if is_large else 0.05)

    kwargs["use_graph_logit"] = True
    kwargs["graph_alpha"] = kwargs.get("graph_alpha", 0.02 if is_large else 0.004)
    kwargs["graph_score_clip"] = kwargs.get("graph_score_clip", 3.0 if is_large else 2.0)
    kwargs["use_flow_in_train"] = False
    kwargs["target_aware_graph_pooling"] = False

    kwargs["use_multi_interest"] = True
    kwargs["num_interests"] = kwargs.get("num_interests", 4)
    kwargs["mic_aggregation"] = "max"
    kwargs["orth_weight"] = kwargs.get("orth_weight", 0.01)
    kwargs["use_flow_multi_rerank"] = True
    kwargs["flow_multi_topk"] = kwargs.get("flow_multi_topk", 20)
    kwargs["flow_multi_beta"] = kwargs.get("flow_multi_beta", 0.015)
    kwargs["flow_rerank_recent"] = kwargs.get("flow_rerank_recent", 1)
    kwargs["flow_rerank_scale"] = kwargs.get("flow_rerank_scale", 100.0)
    kwargs["flow_rerank_keep"] = kwargs.get("flow_rerank_keep", 1)
    kwargs["flow_rerank_confidence"] = kwargs.get("flow_rerank_confidence", 0.30)
    kwargs["flow_rerank_primary_bonus"] = kwargs.get("flow_rerank_primary_bonus", 0.0)
    kwargs["use_local_rerank"] = False

    kwargs["dataset_class"] = "TeaRecDataset"
    kwargs["seed"] = kwargs.get("seed", 2020)
    kwargs["reproducibility"] = kwargs.get("reproducibility", True)
    kwargs["rebuild_gtf"] = kwargs.get("rebuild_gtf", False)

    return kwargs


def _interaction_field(interaction, *names):
    for name in names:
        try:
            if name in interaction:
                return interaction[name]
        except Exception:
            pass
        try:
            return interaction[name]
        except Exception:
            pass
    return None


def _flow_cache_path(project_root, dataset_name, n_items, config, kind):
    inner_weight = config["flow_inner_weight"] if "flow_inner_weight" in config else 0.2
    target_weight = config["flow_target_weight"] if "flow_target_weight" in config else 1.0
    smoothing = config["flow_smoothing"] if "flow_smoothing" in config else 1e-8
    flow_topk = config["flow_topk"] if "flow_topk" in config else 0
    use_2hop = config["use_2hop_flow"] if "use_2hop_flow" in config else True
    beta_2hop = config["flow_2hop_weight"] if "flow_2hop_weight" in config else 0.05
    hop_tag = f"hop2_b{beta_2hop}" if use_2hop else "hop1"
    return os.path.join(
        project_root,
        "saved",
        f"E1_GTF_{kind}_{dataset_name}_n{n_items}_in{inner_weight}_tgt{target_weight}"
        f"_sm{smoothing}_top{flow_topk}_{hop_tag}.pt",
    )


def build_dense_flow_map(train_data, n_items, config):
    inner_weight = float(config["flow_inner_weight"] if "flow_inner_weight" in config else 0.2)
    target_weight = float(config["flow_target_weight"] if "flow_target_weight" in config else 1.0)
    smoothing = float(config["flow_smoothing"] if "flow_smoothing" in config else 1e-8)
    flow_topk = int(config["flow_topk"] if "flow_topk" in config else 0)
    use_2hop = bool(config["use_2hop_flow"] if "use_2hop_flow" in config else True)
    beta_2hop = float(config["flow_2hop_weight"] if "flow_2hop_weight" in config else 0.05)
    flow = torch.zeros((n_items, n_items), dtype=torch.float32)

    for interaction in train_data:
        item_seq = interaction["item_id_list"].cpu()
        item_seq_len = interaction["item_length"].cpu()
        pos_items = _interaction_field(interaction, "item_id", "pos_item_id")
        pos_items = pos_items.cpu() if pos_items is not None else None

        for row, length, idx in zip(item_seq, item_seq_len, range(item_seq.size(0))):
            length = int(length.item())
            seq = row[:length].tolist()
            for src, dst in zip(seq[:-1], seq[1:]):
                if src != 0 and dst != 0:
                    flow[src, dst] += inner_weight
            if pos_items is not None and length > 0:
                src = int(seq[-1])
                dst = int(pos_items[idx].item())
                if src != 0 and dst != 0:
                    flow[src, dst] += target_weight

    flow[0, :] = 0.0
    flow[:, 0] = 0.0
    if smoothing > 0:
        flow[1:, 1:] += smoothing

    row_sum = flow.sum(dim=-1, keepdim=True)
    row_sum[row_sum == 0] = 1.0
    p1 = flow / row_sum

    if flow_topk > 0:
        k = min(flow_topk, n_items)
        values, indices = torch.topk(p1, k, dim=-1)
        sparse_p1 = torch.zeros_like(p1)
        sparse_p1.scatter_(1, indices, values)
        p1 = sparse_p1 / sparse_p1.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    if use_2hop and beta_2hop > 0:
        print(f"[E1-GTF] Mix dense 2-hop flow, beta={beta_2hop}")
        p2 = torch.mm(p1, p1)
        graph = (1.0 - beta_2hop) * p1 + beta_2hop * p2
        graph = graph / graph.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        graph[0].zero_()
        return graph

    return p1


def build_sparse_gtf_graph(train_data, n_items, plm_embedding, config):
    """为大数据集构建 sparse 一/二跳 GTF、图先验和流图特征表示。"""
    inner_weight = float(config["flow_inner_weight"] if "flow_inner_weight" in config else 0.2)
    target_weight = float(config["flow_target_weight"] if "flow_target_weight" in config else 1.0)
    configured_topk = int(config["flow_topk"] if "flow_topk" in config else 0)
    sparse_topk = configured_topk if configured_topk > 0 else 100
    use_2hop = bool(config["use_2hop_flow"] if "use_2hop_flow" in config else True)
    beta_2hop = float(config["flow_2hop_weight"] if "flow_2hop_weight" in config else 0.05)
    counts = defaultdict(lambda: defaultdict(float))

    for interaction in train_data:
        item_seq = interaction["item_id_list"].cpu()
        item_seq_len = interaction["item_length"].cpu()
        pos_items = _interaction_field(interaction, "item_id", "pos_item_id")
        pos_items = pos_items.cpu() if pos_items is not None else None

        for i in range(item_seq.size(0)):
            length = int(item_seq_len[i].item())
            if length <= 0:
                continue
            seq = item_seq[i, :length].tolist()
            if length > 1:
                for src, dst in zip(seq[:-1], seq[1:]):
                    if src != 0 and dst != 0:
                        counts[int(src)][int(dst)] += inner_weight
            if pos_items is not None:
                dst = int(pos_items[i].item())
                src = int(seq[-1])
                if src != 0 and dst != 0:
                    counts[src][dst] += target_weight

    first_hop = {}
    for src, dst_counts in counts.items():
        total = sum(dst_counts.values())
        if total <= 0:
            continue
        ranked = sorted(dst_counts.items(), key=lambda pair: pair[1], reverse=True)[:sparse_topk]
        first_hop[src] = [(dst, value / total) for dst, value in ranked]

    mixed_hop = first_hop
    if use_2hop and beta_2hop > 0:
        print(f"[E1-GTF] Mix sparse 2-hop flow, beta={beta_2hop}, topk={sparse_topk}")
        mixed_hop = {}
        for src, direct_edges in first_hop.items():
            mixed = defaultdict(float)
            for dst, prob in direct_edges:
                mixed[dst] += (1.0 - beta_2hop) * prob
                for dst2, prob2 in first_hop.get(dst, []):
                    mixed[dst2] += beta_2hop * prob * prob2
            ranked = sorted(mixed.items(), key=lambda pair: pair[1], reverse=True)[:sparse_topk]
            total = sum(value for _, value in ranked)
            if total > 0:
                mixed_hop[src] = [(dst, value / total) for dst, value in ranked]

    indices = torch.zeros((n_items, sparse_topk), dtype=torch.long)
    values = torch.zeros((n_items, sparse_topk), dtype=torch.float32)
    for src, edges in mixed_hop.items():
        if not (0 < src < n_items):
            continue
        valid_edges = [(dst, prob) for dst, prob in edges if 0 < dst < n_items]
        if not valid_edges:
            continue
        length = min(len(valid_edges), sparse_topk)
        indices[src, :length] = torch.tensor([dst for dst, _ in valid_edges[:length]])
        values[src, :length] = torch.tensor([prob for _, prob in valid_edges[:length]])

    plm = plm_embedding.detach().cpu().float()
    feature = torch.zeros((n_items, plm.size(1)), dtype=torch.float32)
    chunk_size = 256
    for start in range(0, n_items, chunk_size):
        end = min(start + chunk_size, n_items)
        dst_emb = plm[indices[start:end]]
        feature[start:end] = (dst_emb * values[start:end].unsqueeze(-1)).sum(dim=1)
    feature[0].zero_()

    return {
        "graph_embedding": feature,
        "graph_indices": indices,
        "graph_values": values,
    }


def get_or_build_e1_graph(project_root, dataset_name, dataset_obj, train_data, config):
    n_items = dataset_obj.item_num
    os.makedirs(os.path.join(project_root, "saved"), exist_ok=True)

    if n_items > 10000 or is_large_dataset_name(dataset_name):
        cache_path = _flow_cache_path(project_root, dataset_name, n_items, config, "sparse_gtf")
        if "rebuild_gtf" in config and config["rebuild_gtf"] and os.path.exists(cache_path):
            print(f"[E1-GTF] Rebuild requested, delete cache: {cache_path}")
            os.remove(cache_path)
        if os.path.exists(cache_path):
            print(f"[E1-GTF] Load sparse GTF graph: {cache_path}")
            return safe_torch_load(cache_path, map_location="cpu")

        print(f"[E1-GTF] Build sparse GTF graph for large dataset: {dataset_name}, item_num={n_items}")
        graph_state = build_sparse_gtf_graph(train_data, n_items, dataset_obj.plm_embedding, config)
        torch.save(graph_state, cache_path)
        print(f"[E1-GTF] Saved sparse GTF graph: {cache_path}")
        return graph_state

    cache_path = _flow_cache_path(project_root, dataset_name, n_items, config, "dense_flow")
    if "rebuild_gtf" in config and config["rebuild_gtf"] and os.path.exists(cache_path):
        print(f"[E1-GTF] Rebuild requested, delete cache: {cache_path}")
        os.remove(cache_path)
    if os.path.exists(cache_path):
        print(f"[E1-GTF] Load dense GTF graph: {cache_path}")
        return {"graph_embedding": safe_torch_load(cache_path, map_location="cpu").float()}

    print(f"[E1-GTF] Build dense GTF graph: {dataset_name}, item_num={n_items}")
    graph_tensor = build_dense_flow_map(train_data, n_items, config)
    torch.save(graph_tensor, cache_path)
    print(f"[E1-GTF] Saved dense GTF graph: {cache_path}")
    return {"graph_embedding": graph_tensor}


def attach_graph(dataset_like, graph_state):
    target = dataset_like.dataset if hasattr(dataset_like, "dataset") else dataset_like
    for name, value in graph_state.items():
        setattr(target, name, value)


def choose_device(kwargs):
    force_cpu = kwargs.pop("force_cpu", False)
    force_cuda = kwargs.pop("force_cuda", False)

    if force_cpu:
        kwargs["use_gpu"] = False
        kwargs["gpu_id"] = -1
        kwargs["device"] = "cpu"
        print("[Device] Forced CPU mode.")
    elif force_cuda:
        kwargs["use_gpu"] = True
        kwargs["gpu_id"] = 0
        kwargs["device"] = "cuda"
        print("[Device] Forced CUDA mode.")
    elif cuda_is_usable():
        kwargs["use_gpu"] = True
        kwargs["gpu_id"] = 0
        kwargs["device"] = "cuda"
        print("[Device] CUDA usable, using GPU.")
    else:
        kwargs["use_gpu"] = False
        kwargs["gpu_id"] = -1
        kwargs["device"] = "cpu"
        print("[Device] Using CPU mode.")
    return kwargs


def freeze_encoder_for_e4(model, logger):
    logger.info(set_color("Fix encoder parameters. [E4 Full GTF + MIC + MIFG]", "cyan"))

    if hasattr(model, "position_embedding"):
        for param in model.position_embedding.parameters():
            param.requires_grad = False

    if hasattr(model, "trm_encoder"):
        for param in model.trm_encoder.parameters():
            param.requires_grad = False


def finetune(dataset, pretrained_file="", fix_enc=True, eval_only=False, **kwargs):
    project_root = os.path.dirname(os.path.abspath(__file__))

    kwargs = force_e4_teacher_config(kwargs, dataset_name=dataset)
    kwargs = choose_device(kwargs)

    props = [
        os.path.join(project_root, "props", "TeaRec.yaml"),
        os.path.join(project_root, "props", "fintune.yaml"),
    ]

    config = Config(model=TeaRec, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()

    logger.info(config)
    logger.info(
        set_color(
            "Running E4 Full Teacher: GTF+MIC+MIFG enabled.",
            "yellow",
        )
    )

    dataset_obj = TeaRecDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    graph_state = get_or_build_e1_graph(project_root, dataset, dataset_obj, train_data, config)
    attach_graph(dataset_obj, graph_state)
    attach_graph(train_data, graph_state)
    attach_graph(valid_data, graph_state)
    attach_graph(test_data, graph_state)
    logger.info(
        set_color(
            f"[E4-FULL-TEACHER] Graph attached. feature_shape={tuple(graph_state['graph_embedding'].shape)}, "
            f"sparse_logit={'graph_indices' in graph_state}",
            "yellow",
        )
    )

    device = config["device"]
    model = TeaRec(config, train_data.dataset).to(device)

    if pretrained_file and os.path.exists(pretrained_file):
        logger.info(f"Loading pretrained model from: {pretrained_file}")
        checkpoint = safe_torch_load(pretrained_file, map_location=device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in state_dict.items()
            if k in model_dict and tuple(v.shape) == tuple(model_dict[k].shape)
        }

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(set_color(f"Loaded pretrained weights. matched={len(pretrained_dict)}", "yellow"))
    elif pretrained_file:
        logger.warning(set_color(f"Pretrained file {pretrained_file} NOT FOUND. Training from scratch!", "red"))

    if fix_enc:
        freeze_encoder_for_e4(model, logger)
    else:
        logger.info(set_color("Encoder is FULLY trainable. [E4 Full Teacher]", "red"))

    logger.info(model)
    logger.info(set_color("Trainable parameters", "yellow") + f": {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    if eval_only:
        if not pretrained_file or not os.path.exists(pretrained_file):
            raise ValueError("E4 --eval_only requires a trained TeaRec checkpoint.")
        valid_result = trainer.evaluate(
            valid_data, load_best_model=False, show_progress=config["show_progress"]
        )
        test_result = trainer.evaluate(
            test_data, load_best_model=False, show_progress=config["show_progress"]
        )
        logger.info(set_color("E4 valid result", "yellow") + f": {valid_result}")
        logger.info(set_color("E4 test result", "yellow") + f": {test_result}")
        return config["model"], config["dataset"], {
            "best_valid_result": valid_result,
            "test_result": test_result,
        }

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config["show_progress"]
    )

    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config["show_progress"])

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    return config["model"], config["dataset"], {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, default="NYC", help="dataset name")
    parser.add_argument("-p", type=str, default="", help="pre-trained model path")
    parser.add_argument("--full_train", action="store_true", help="Do not freeze encoder")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--force_cuda", action="store_true", help="Force CUDA mode")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--flow_topk", type=int, default=None, help="Override sparse GTF topK")
    parser.add_argument("--graph_alpha", type=float, default=None, help="Override graph logit alpha")
    parser.add_argument("--flow_2hop_weight", type=float, default=None, help="Override 2-hop flow weight")
    parser.add_argument("--graph_score_clip", type=float, default=None, help="Override graph score clipping")
    parser.add_argument("--flow_inner_weight", type=float, default=None, help="Override sequence inner flow weight")
    parser.add_argument("--flow_target_weight", type=float, default=None, help="Override target-transition flow weight")
    parser.add_argument("--num_interests", type=int, default=None, help="Override E2 MIC interest count")
    parser.add_argument("--orth_weight", type=float, default=None, help="Override E2 MIC orthogonality weight")
    parser.add_argument("--flow_multi_topk", type=int, default=None, help="Override E4 local rerank topK")
    parser.add_argument("--flow_multi_beta", type=float, default=None, help="Override E4 local rerank strength")
    parser.add_argument("--flow_rerank_recent", type=int, default=None, help="Override E4 recent trajectory anchors")
    parser.add_argument("--flow_rerank_scale", type=float, default=None, help="Override E4 flow probability scale")
    parser.add_argument("--flow_rerank_keep", type=int, default=None, help="Override number of strongest flow candidates to boost")
    parser.add_argument("--flow_rerank_confidence", type=float, default=None, help="Only rerank when the strongest flow path is sufficiently clear")
    parser.add_argument("--flow_rerank_primary_bonus", type=float, default=None, help="Extra boost for a confident strongest flow path")
    parser.add_argument("--eval_only", action="store_true", help="Evaluate E4 full teacher from a TeaRec checkpoint")
    parser.add_argument("--rebuild_gtf", action="store_true", help="Force rebuild cached GTF graph")
    args, _ = parser.parse_known_args()

    extra_kwargs = {}
    if args.cpu:
        extra_kwargs["force_cpu"] = True
    if args.force_cuda:
        extra_kwargs["force_cuda"] = True
    if args.epochs is not None:
        extra_kwargs["epochs"] = args.epochs
    if args.lr is not None:
        extra_kwargs["learning_rate"] = args.lr
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
    if args.num_interests is not None:
        extra_kwargs["num_interests"] = args.num_interests
    if args.orth_weight is not None:
        extra_kwargs["orth_weight"] = args.orth_weight
    if args.flow_multi_topk is not None:
        extra_kwargs["flow_multi_topk"] = args.flow_multi_topk
    if args.flow_multi_beta is not None:
        extra_kwargs["flow_multi_beta"] = args.flow_multi_beta
    if args.flow_rerank_recent is not None:
        extra_kwargs["flow_rerank_recent"] = args.flow_rerank_recent
    if args.flow_rerank_scale is not None:
        extra_kwargs["flow_rerank_scale"] = args.flow_rerank_scale
    if args.flow_rerank_keep is not None:
        extra_kwargs["flow_rerank_keep"] = args.flow_rerank_keep
    if args.flow_rerank_confidence is not None:
        extra_kwargs["flow_rerank_confidence"] = args.flow_rerank_confidence
    if args.flow_rerank_primary_bonus is not None:
        extra_kwargs["flow_rerank_primary_bonus"] = args.flow_rerank_primary_bonus
    if args.rebuild_gtf:
        extra_kwargs["rebuild_gtf"] = True

    finetune(
        dataset=args.d,
        pretrained_file=args.p,
        fix_enc=(not args.full_train),
        eval_only=args.eval_only,
        **extra_kwargs,
    )
