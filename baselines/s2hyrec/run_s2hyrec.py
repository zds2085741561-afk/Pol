# -*- coding: utf-8 -*-
"""Train S2HyRec under the formal EffiPOI/RecBole comparison protocol."""

import argparse
import json
import os
import sys
from logging import getLogger

import torch
from recbole.data import data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed, set_color

BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASELINE_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config  # noqa: E402
from data.dataset import TeaRecDataset  # noqa: E402
from s2hyrec_recbole import S2HyRecRecBole  # noqa: E402


def load_trusted_state_dict(model, checkpoint, device):
    """Load a checkpoint produced locally by this training script.

    PyTorch 2.6 changed ``torch.load`` to ``weights_only=True`` by default,
    while RecBole checkpoints also contain optimizer/config metadata. The
    checkpoint is generated in the same local process, so loading the complete
    payload is intentional here.
    """
    try:
        payload = torch.load(
            checkpoint,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)

    state_dict = payload.get("state_dict", payload)
    return model.load_state_dict(state_dict, strict=True)


def parse_args():
    parser = argparse.ArgumentParser(description="S2HyRec formal baseline")
    parser.add_argument("--dataset", default="NYC", choices=["NYC", "NYC_STEPS"])
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train_batch_size", "--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--stopping_step", type=int, default=10)
    parser.add_argument("--learning_rate", "--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--inner_size", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hyper_num", type=int, default=128)
    parser.add_argument("--time_slices", type=int, default=4)
    parser.add_argument("--hyper_temperature", type=float, default=0.2)
    parser.add_argument("--global_intent_weight", "--alpha", type=float, default=0.5)
    parser.add_argument("--intent_residual_weight", type=float, default=1.0)
    parser.add_argument("--ssl_weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--show_progress", action="store_true", default=True)
    parser.add_argument("--hide_progress", action="store_true")
    return parser.parse_args()


def build_config(args):
    checkpoint_dir = os.path.join(BASELINE_DIR, "saved")
    os.makedirs(checkpoint_dir, exist_ok=True)

    config_dict = {
        "data_path": os.path.join(PROJECT_ROOT, "data"),
        "benchmark_filename": ["train", "valid", "test"],
        "dataset_class": "TeaRecDataset",
        "use_gpu": bool(args.use_gpu),
        "gpu_id": args.gpu_id,
        "seed": args.seed,
        "reproducibility": True,
        "show_progress": args.show_progress and not args.hide_progress,
        "epochs": args.epochs,
        "eval_step": 1,
        "stopping_step": args.stopping_step,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss_type": "CE",
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "hidden_size": args.hidden_size,
        "inner_size": args.inner_size,
        "hidden_dropout_prob": args.dropout,
        "attn_dropout_prob": args.dropout,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
        "hyper_num": args.hyper_num,
        "time_slices": args.time_slices,
        "hyper_temperature": args.hyper_temperature,
        "global_intent_weight": args.global_intent_weight,
        "intent_residual_weight": args.intent_residual_weight,
        "ssl_weight": args.ssl_weight,
        "metrics": ["HIT", "NDCG"],
        "topk": [1, 5, 10, 20],
        "valid_metric": "HIT@10",
        "checkpoint_dir": checkpoint_dir,
        "train_neg_sample_args": None,
    }

    props = [os.path.join(PROJECT_ROOT, "props", "fintune.yaml")]
    return Config(
        model=S2HyRecRecBole,
        dataset=args.dataset,
        config_file_list=props,
        config_dict=config_dict,
    )


def main():
    args = parse_args()
    config = build_config(args)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()

    logger.info("=" * 70)
    logger.info("[Baseline] S2HyRec formal RecBole/EffiPOI comparison")
    logger.info(
        f"[Dataset] {config['dataset']} | hyperedges={config['hyper_num']} | "
        f"time_slices={config['time_slices']} | alpha={config['global_intent_weight']} | "
        f"ssl={config['ssl_weight']} | device={config['device']}"
    )
    logger.info("[Protocol] Fixed benchmark split + RecBole full-sort HIT/NDCG")
    logger.info(
        "[Temporal context] Relative chronological segments are used because "
        "the benchmark files contain no absolute timestamps."
    )
    logger.info("=" * 70)

    dataset = TeaRecDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = S2HyRecRecBole(config, train_data.dataset).to(config["device"])
    logger.info(model)
    logger.info(
        set_color("Trainable parameters", "yellow")
        + f": {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config["show_progress"],
    )
    incompatible = load_trusted_state_dict(
        model,
        trainer.saved_model_file,
        config["device"],
    )
    logger.info(
        "[Best checkpoint loaded] "
        f"{trainer.saved_model_file} | "
        f"missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )
    test_result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=config["show_progress"],
    )

    logger.info(set_color("best valid", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result_dir = os.path.join(BASELINE_DIR, "results")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(
        result_dir,
        f"S2HyRec_{config['dataset']}_result.json",
    )
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "model": "S2HyRec",
                "dataset": config["dataset"],
                "protocol": "RecBole full-sort",
                "temporal_context": "relative chronological segments",
                "best_valid_score": float(best_valid_score),
                "best_valid_result": {
                    key: float(value) for key, value in best_valid_result.items()
                },
                "test_result": {
                    key: float(value) for key, value in test_result.items()
                },
                "config": {
                    "hidden_size": config["hidden_size"],
                    "n_layers": config["n_layers"],
                    "n_heads": config["n_heads"],
                    "hyper_num": config["hyper_num"],
                    "time_slices": config["time_slices"],
                    "global_intent_weight": config["global_intent_weight"],
                    "ssl_weight": config["ssl_weight"],
                },
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"[Saved] JSON: {result_path}")


if __name__ == "__main__":
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    main()
