import argparse
import itertools
import os
from logging import getLogger

import lightgbm as lgb
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import Config
from data.dataset import TeaRecDataset
from recbole.data import data_preparation
from recbole.utils import init_logger, init_seed, set_color
from teacher import TeaRec


def safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def evaluate_flat_scores(score_flat, y_flat, topk, k=10):
    scores = score_flat.reshape(-1, topk)
    labels = y_flat.reshape(-1, topk)

    hits = []
    ndcgs = []

    for i in range(scores.shape[0]):
        order = np.argsort(scores[i])[::-1][:k]
        top_labels = labels[i][order]

        hit = 1 if np.sum(top_labels) > 0 else 0
        hits.append(hit)

        ndcg = 0.0
        for rank, label in enumerate(top_labels):
            if label == 1:
                ndcg = 1.0 / np.log2(rank + 2)
                break
        ndcgs.append(ndcg)

    return float(np.mean(hits)), float(np.mean(ndcgs))


def group_zscore_flat(score_flat, topk):
    scores = score_flat.reshape(-1, topk)
    mean = scores.mean(axis=1, keepdims=True)
    std = scores.std(axis=1, keepdims=True) + 1e-8
    return ((scores - mean) / std).reshape(-1)


def build_flow_map_path(project_root, dataset_name, n_items, config):
    inner_weight = config["flow_inner_weight"] if "flow_inner_weight" in config else 0.2
    target_weight = config["flow_target_weight"] if "flow_target_weight" in config else 1.0
    smoothing = config["flow_smoothing"] if "flow_smoothing" in config else 1e-8
    flow_topk = config["flow_topk"] if "flow_topk" in config else 0
    use_2hop = config["use_2hop_flow"] if "use_2hop_flow" in config else False
    beta_2hop = config["flow_2hop_weight"] if "flow_2hop_weight" in config else 0.05

    if use_2hop:
        hop_tag = "hop2_b" + str(beta_2hop)
    else:
        hop_tag = "hop1"

    flow_file_name = (
        f"{dataset_name}_n{n_items}_in{inner_weight}_tgt{target_weight}"
        f"_sm{smoothing}_top{flow_topk}_{hop_tag}.pt"
    )

    return os.path.join(project_root, "saved", flow_file_name)


def load_teacher_model(config, dataset_obj, ckpt_path, device, graph_alpha=None):
    model = TeaRec(config, dataset_obj).to(device)

    checkpoint = safe_torch_load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    if graph_alpha is not None:
        model.graph_alpha = float(graph_alpha)

    model.eval()
    return model


def compute_item_popularity(train_data, model, device):
    pop_tensor = torch.zeros(model.n_items, device=device)

    for batch in train_data:
        if isinstance(batch, tuple):
            batch = batch[0]

        seqs = batch[model.ITEM_SEQ].to(device)
        valid_items = seqs[seqs > 0]

        pop_tensor.scatter_add_(
            0,
            valid_items,
            torch.ones_like(valid_items, dtype=torch.float)
        )

    return torch.log1p(pop_tensor)


def extract_ltr_data(models, dataloader, pop_tensor, is_train=False, topk=200):
    """
    使用第 1 个模型作为候选召回器。
    所有模型都在同一批候选上打分，方便做多 checkpoint 融合。

    返回：
    X: LightGBM 特征
    y: 标签
    group: 每个样本的候选数量
    model_scores: shape = [num_models, num_rows]
    """
    primary_model = models[0]
    primary_model.eval()

    device = next(primary_model.parameters()).device
    num_models = len(models)

    X_list = []
    y_list = []
    group_list = []
    score_chunks = [[] for _ in range(num_models)]

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Extracting Features"):
            if isinstance(batch_data, tuple):
                interaction = batch_data[0]
            else:
                interaction = batch_data

            interaction = interaction.to(device)

            item_seq = interaction[primary_model.ITEM_SEQ]
            item_seq_len = interaction[primary_model.ITEM_SEQ_LEN]
            pos_items = interaction[primary_model.POS_ITEM_ID]
            batch_size = item_seq.size(0)

            # 1. 第一个模型负责生成候选 TopK
            primary_scores, _ = primary_model.get_teacher_outputs(interaction)
            topk_scores, topk_indices = torch.topk(primary_scores, topk, dim=-1)

            # 2. 训练集强制放入正样本，避免 LTR 没有正例可学
            if is_train:
                pos_mask = (topk_indices == pos_items.unsqueeze(1))
                hit_rows = pos_mask.any(dim=1)
                miss_rows = ~hit_rows

                if miss_rows.any():
                    topk_indices[miss_rows, -1] = pos_items[miss_rows]
                    topk_scores[miss_rows, -1] = primary_scores[
                        miss_rows,
                        pos_items[miss_rows]
                    ]

            # 3. 所有 checkpoint 对同一批候选打分
            model_topk_scores = []
            for m_idx, model in enumerate(models):
                if m_idx == 0:
                    full_scores = primary_scores
                else:
                    full_scores, _ = model.get_teacher_outputs(interaction)

                s = full_scores.gather(1, topk_indices)
                model_topk_scores.append(s)
                score_chunks[m_idx].append(s.cpu().numpy().reshape(-1))

            # ==========================
            # LightGBM 低维特征
            # ==========================

            # 1. primary teacher score
            f1 = model_topk_scores[0]

            # 2. primary teacher rank
            f2 = torch.arange(topk, device=device).unsqueeze(0).expand(batch_size, topk).float()

            # 当前序列最后一个有效 item
            valid_len = torch.clamp(item_seq_len - 1, min=0, max=item_seq.size(1) - 1)
            last_item = item_seq.gather(1, valid_len.unsqueeze(1)).squeeze(1)

            # 3. flow_score / 4. flow_rank
            if getattr(primary_model, "graph_matrix", None) is not None:
                flow_probs = primary_model.graph_matrix[last_item].to(device).float()
                flow_prob_k = flow_probs.gather(1, topk_indices)
                f3 = torch.log(flow_prob_k + 1e-8)

                flow_ranks = flow_probs.argsort(dim=-1, descending=True).argsort(dim=-1)
                f4 = flow_ranks.gather(1, topk_indices).float()
            else:
                f3 = torch.zeros_like(topk_scores)
                f4 = torch.zeros_like(topk_scores)

            # 5. PLM similarity between last item and candidate
            plm = primary_model.plm_embedding
            last_emb = F.normalize(plm[last_item], dim=-1)
            cand_emb = F.normalize(plm[topk_indices], dim=-1)
            f5 = (last_emb.unsqueeze(1) * cand_emb).sum(dim=-1)

            # 6. history_plm_sim_max / 7. history_plm_sim_mean
            seq_emb = F.normalize(plm[item_seq], dim=-1)
            sim_matrix = torch.bmm(seq_emb, cand_emb.transpose(1, 2))

            seq_mask = (item_seq > 0).unsqueeze(-1).expand(-1, -1, topk)

            sim_matrix_masked = sim_matrix.masked_fill(~seq_mask, -1.0)
            f6 = sim_matrix_masked.max(dim=1)[0]

            sim_sum = sim_matrix.masked_fill(~seq_mask, 0.0).sum(dim=1)
            f7 = sim_sum / item_seq_len.unsqueeze(1).clamp(min=1).float()

            # 8. candidate popularity
            f8 = pop_tensor[topk_indices]

            # 9. revisit flag
            f9 = (topk_indices.unsqueeze(2) == item_seq.unsqueeze(1)).any(dim=2).float()

            # 10. sequence length
            f10 = item_seq_len.unsqueeze(1).expand(batch_size, topk).float()

            # 11. last item popularity
            f11 = pop_tensor[last_item].unsqueeze(1).expand(batch_size, topk)

            # 12. graph score z
            mean = f3.mean(dim=-1, keepdim=True)
            std = f3.std(dim=-1, keepdim=True) + 1e-8
            f12 = (f3 - mean) / std

            base_features = [
                f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12
            ]

            # 额外加入其他 checkpoint 的分数作为 LGBM 特征
            if num_models > 1:
                for m_idx in range(1, num_models):
                    base_features.append(model_topk_scores[m_idx])

                stacked_scores = torch.stack(model_topk_scores, dim=-1)
                score_mean = stacked_scores.mean(dim=-1)
                score_std = stacked_scores.std(dim=-1)
                base_features.append(score_mean)
                base_features.append(score_std)

            batch_X = torch.stack(base_features, dim=-1)
            batch_y = (topk_indices == pos_items.unsqueeze(1)).int()

            X_list.append(batch_X.cpu().numpy().reshape(batch_size * topk, -1))
            y_list.append(batch_y.cpu().numpy().reshape(-1))
            group_list.extend([topk] * batch_size)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    group = np.array(group_list)

    model_scores = np.stack(
        [np.concatenate(score_chunks[i]) for i in range(num_models)],
        axis=0
    )

    return X, y, group, model_scores


def train_lightgbm_ranker(X_train, y_train, q_train, X_valid, y_valid, q_valid, logger):
    logger.info(set_color("Training LightGBM Reranker...", "green"))

    lgb_train = lgb.Dataset(X_train, y_train, group=q_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, group=q_valid, reference=lgb_train)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": 2020,
    }

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=20),
        ],
    )

    return gbm


def search_multi_fusion(
    valid_model_scores,
    test_model_scores,
    lgbm_valid_pred,
    lgbm_test_pred,
    y_valid,
    y_test,
    topk,
    logger,
):
    num_models = valid_model_scores.shape[0]

    valid_model_z = [
        group_zscore_flat(valid_model_scores[i], topk)
        for i in range(num_models)
    ]

    test_model_z = [
        group_zscore_flat(test_model_scores[i], topk)
        for i in range(num_models)
    ]

    lgbm_valid_z = group_zscore_flat(lgbm_valid_pred, topk)
    lgbm_test_z = group_zscore_flat(lgbm_test_pred, topk)

    # 第一个模型作为主模型，权重固定为 1.0
    extra_weight_grid = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00]
    lgbm_alpha_grid = [
        0.0,
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.08,
        0.10,
        0.15,
        0.20,
        0.30,
    ]

    # 为了避免组合爆炸，建议最多 4 个模型
    if num_models > 4:
        logger.warning(
            set_color(
                f"Too many checkpoints: {num_models}. Grid search may be slow. "
                f"建议最多 3 到 4 个 checkpoint。",
                "yellow"
            )
        )

    best = {
        "valid_hit": -1.0,
        "valid_ndcg": -1.0,
        "test_hit": -1.0,
        "test_ndcg": -1.0,
        "extra_weights": None,
        "lgbm_alpha": None,
    }

    logger.info(set_color("========== INDIVIDUAL MODEL BASELINES ==========", "cyan"))

    for i in range(num_models):
        v_hit, v_ndcg = evaluate_flat_scores(valid_model_scores[i], y_valid, topk, k=10)
        t_hit, t_ndcg = evaluate_flat_scores(test_model_scores[i], y_test, topk, k=10)

        logger.info(
            set_color(
                f"Model {i} | Valid HIT@10={v_hit:.4f} NDCG@10={v_ndcg:.4f} "
                f"| Test HIT@10={t_hit:.4f} NDCG@10={t_ndcg:.4f}",
                "cyan"
            )
        )

    logger.info(set_color("========== PURE LIGHTGBM RERANK ==========", "cyan"))
    lgbm_valid_hit, lgbm_valid_ndcg = evaluate_flat_scores(lgbm_valid_pred, y_valid, topk, k=10)
    lgbm_test_hit, lgbm_test_ndcg = evaluate_flat_scores(lgbm_test_pred, y_test, topk, k=10)

    logger.info(set_color(f"LightGBM Valid HIT@10  : {lgbm_valid_hit:.4f}", "cyan"))
    logger.info(set_color(f"LightGBM Valid NDCG@10 : {lgbm_valid_ndcg:.4f}", "cyan"))
    logger.info(set_color(f"LightGBM Test  HIT@10  : {lgbm_test_hit:.4f}", "cyan"))
    logger.info(set_color(f"LightGBM Test  NDCG@10 : {lgbm_test_ndcg:.4f}", "cyan"))

    logger.info(set_color("========== MULTI-CHECKPOINT FUSION GRID SEARCH ==========", "green"))

    if num_models == 1:
        weight_product = [()]
    else:
        weight_product = itertools.product(extra_weight_grid, repeat=num_models - 1)

    for extra_weights in weight_product:
        base_valid = valid_model_z[0].copy()
        base_test = test_model_z[0].copy()

        for idx, w in enumerate(extra_weights, start=1):
            base_valid += w * valid_model_z[idx]
            base_test += w * test_model_z[idx]

        for lgbm_alpha in lgbm_alpha_grid:
            valid_final = base_valid + lgbm_alpha * lgbm_valid_z

            v_hit, v_ndcg = evaluate_flat_scores(valid_final, y_valid, topk, k=10)

            if v_hit > best["valid_hit"] or (
                v_hit == best["valid_hit"] and v_ndcg > best["valid_ndcg"]
            ):
                test_final = base_test + lgbm_alpha * lgbm_test_z
                t_hit, t_ndcg = evaluate_flat_scores(test_final, y_test, topk, k=10)

                best.update(
                    {
                        "valid_hit": v_hit,
                        "valid_ndcg": v_ndcg,
                        "test_hit": t_hit,
                        "test_ndcg": t_ndcg,
                        "extra_weights": extra_weights,
                        "lgbm_alpha": lgbm_alpha,
                    }
                )

                logger.info(
                    set_color(
                        f"[NEW BEST] extra_weights={extra_weights}, "
                        f"lgbm_alpha={lgbm_alpha:.3f} | "
                        f"Valid HIT@10={v_hit:.4f}, NDCG@10={v_ndcg:.4f} | "
                        f"Test HIT@10={t_hit:.4f}, NDCG@10={t_ndcg:.4f}",
                        "yellow"
                    )
                )

    logger.info(set_color("========== FINAL MULTI-FUSION RESULTS ==========", "green"))
    logger.info(set_color(f"Best extra model weights : {best['extra_weights']}", "green"))
    logger.info(set_color(f"Best LightGBM alpha      : {best['lgbm_alpha']}", "green"))
    logger.info(set_color(f"Fusion Valid HIT@10      : {best['valid_hit']:.4f}", "green"))
    logger.info(set_color(f"Fusion Valid NDCG@10     : {best['valid_ndcg']:.4f}", "green"))
    logger.info(set_color(f"Fusion Test HIT@10       : {best['test_hit']:.4f}", "green"))
    logger.info(set_color(f"Fusion Test NDCG@10      : {best['test_ndcg']:.4f}", "green"))

    return best


def run_pipeline(dataset_name, ckpt_paths, model_alphas=None, topk=200):
    project_root = os.path.dirname(os.path.abspath(__file__))

    kwargs = {
        "use_gpu": True,
        "gpu_id": "0",
        "device": "cuda",
        "eval_batch_size": 256,
        "train_batch_size": 128,
    }

    props = [
        os.path.join(project_root, "props/TeaRec.yaml"),
        os.path.join(project_root, "props/fintune.yaml"),
    ]

    config = Config(
        model=TeaRec,
        dataset=dataset_name,
        config_file_list=props,
        config_dict=kwargs,
    )

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    logger = getLogger()
    logger.info(config)

    dataset_obj = TeaRecDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    flow_map_path = build_flow_map_path(
        project_root=project_root,
        dataset_name=dataset_name,
        n_items=dataset_obj.item_num,
        config=config,
    )

    if os.path.exists(flow_map_path):
        graph_tensor = safe_torch_load(flow_map_path, map_location="cpu")

        dataset_obj.graph_embedding = graph_tensor
        train_data.dataset.graph_embedding = graph_tensor
        valid_data.dataset.graph_embedding = graph_tensor
        test_data.dataset.graph_embedding = graph_tensor

        logger.info(set_color(f"Loaded flow map: {flow_map_path}", "green"))
    else:
        logger.warning(set_color(f"Flow map NOT FOUND: {flow_map_path}", "red"))

    device = config["device"]

    if model_alphas is not None and len(model_alphas) != len(ckpt_paths):
        raise ValueError("如果传 --model_alphas，它的数量必须和 checkpoint 数量一致。")

    logger.info(set_color("Loading Teacher checkpoints...", "green"))

    models = []

    for idx, ckpt_path in enumerate(ckpt_paths):
        alpha = None
        if model_alphas is not None:
            alpha = model_alphas[idx]

        model = load_teacher_model(
            config=config,
            dataset_obj=train_data.dataset,
            ckpt_path=ckpt_path,
            device=device,
            graph_alpha=alpha,
        )

        logger.info(
            set_color(
                f"Loaded model {idx}: {ckpt_path} | graph_alpha={getattr(model, 'graph_alpha', None)}",
                "cyan"
            )
        )

        models.append(model)

    pop_tensor = compute_item_popularity(train_data, models[0], device)

    logger.info(set_color("Extracting Train LTR Features...", "yellow"))
    X_train, y_train, q_train, train_model_scores = extract_ltr_data(
        models=models,
        dataloader=train_data,
        pop_tensor=pop_tensor,
        is_train=True,
        topk=topk,
    )

    logger.info(set_color("Extracting Valid LTR Features...", "yellow"))
    X_valid, y_valid, q_valid, valid_model_scores = extract_ltr_data(
        models=models,
        dataloader=valid_data,
        pop_tensor=pop_tensor,
        is_train=False,
        topk=topk,
    )

    logger.info(set_color("Extracting Test LTR Features...", "yellow"))
    X_test, y_test, q_test, test_model_scores = extract_ltr_data(
        models=models,
        dataloader=test_data,
        pop_tensor=pop_tensor,
        is_train=False,
        topk=topk,
    )

    # 训练 LGBM
    gbm = train_lightgbm_ranker(
        X_train=X_train,
        y_train=y_train,
        q_train=q_train,
        X_valid=X_valid,
        y_valid=y_valid,
        q_valid=q_valid,
        logger=logger,
    )

    lgbm_valid_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    lgbm_test_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    best = search_multi_fusion(
        valid_model_scores=valid_model_scores,
        test_model_scores=test_model_scores,
        lgbm_valid_pred=lgbm_valid_pred,
        lgbm_test_pred=lgbm_test_pred,
        y_valid=y_valid,
        y_test=y_test,
        topk=topk,
        logger=logger,
    )

    saved_dir = os.path.join(project_root, "saved")
    model_save_path = os.path.join(saved_dir, "lightgbm_multi_ltr_reranker.txt")
    gbm.save_model(model_save_path)

    logger.info(set_color(f"Saved LightGBM model to: {model_save_path}", "green"))

    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", type=str, default="NYC", help="dataset name")

    parser.add_argument(
        "-p",
        type=str,
        default=None,
        help="Single checkpoint path. 兼容旧命令。"
    )

    parser.add_argument(
        "--ckpts",
        nargs="+",
        default=None,
        help="多个 Teacher checkpoint 路径，按主模型 A、辅助模型 B、C 的顺序传。"
    )

    parser.add_argument(
        "--model_alphas",
        nargs="+",
        type=float,
        default=None,
        help="每个 checkpoint 使用的 graph_alpha。例如：--model_alphas 0.004 0.008 0.004"
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=200,
        help="Teacher 召回候选数，建议先 200，再试 300。"
    )

    args = parser.parse_args()

    if args.ckpts is not None:
        ckpt_paths = args.ckpts
    elif args.p is not None:
        ckpt_paths = [args.p]
    else:
        raise ValueError("必须传 -p 或 --ckpts。")

    run_pipeline(
        dataset_name=args.d,
        ckpt_paths=ckpt_paths,
        model_alphas=args.model_alphas,
        topk=args.topk,
    )
