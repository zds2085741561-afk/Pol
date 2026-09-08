# -*- coding: utf-8 -*-
"""Train or evaluate BSARec under the formal EffiPOI comparison protocol."""

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

from bsarec_recbole import BSARecRecBole  # noqa: E402
from config import Config  # noqa: E402
from data.dataset import TeaRecDataset  # noqa: E402


def load_trusted_checkpoint(model, checkpoint, device):
    """Load a checkpoint created locally by this script under PyTorch 2.6+."""
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
    parser = argparse.ArgumentParser(description="BSARec formal AAAI baseline")
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
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--cutoff", "--c", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--hide_progress", action="store_true")
    parser.add_argument("--valid_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
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
        "show_progress": not args.hide_progress,
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
        "bsarec_alpha": args.alpha,
        "bsarec_cutoff": args.cutoff,
        "metrics": ["HIT", "NDCG"],
        "topk": [1, 5, 10, 20],
        "valid_metric": "HIT@10",
        "checkpoint_dir": checkpoint_dir,
        "train_neg_sample_args": None,
    }
    props = [os.path.join(PROJECT_ROOT, "props", "fintune.yaml")]
    return Config(
        model=BSARecRecBole,
        dataset=args.dataset,
        config_file_list=props,
        config_dict=config_dict,
    )


def float_metrics(metrics):
    return {key: float(value) for key, value in metrics.items()}


def save_result(args, config, payload):
    result_dir = os.path.join(BASELINE_DIR, "results")
    os.makedirs(result_dir, exist_ok=True)
    result_path = args.output or os.path.join(
        result_dir,
        f"BSARec_{config['dataset']}_result.json",
    )
    if not os.path.isabs(result_path):
        result_path = os.path.join(PROJECT_ROOT, result_path)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return result_path


def main():
    args = parse_args()
    if args.eval_only and not args.checkpoint:
        raise ValueError("--eval_only requires --checkpoint.")

    config = build_config(args)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()

    logger.info("=" * 70)
    logger.info("[Baseline] BSARec (AAAI 2024) formal comparison")
    logger.info(
        f"[Dataset] {config['dataset']} | alpha={config['bsarec_alpha']} | "
        f"c={config['bsarec_cutoff']} | hidden={config['hidden_size']} | "
        f"layers={config['n_layers']} | heads={config['n_heads']}"
    )
    logger.info("[Protocol] Fixed benchmark split + RecBole full-sort HIT/NDCG")
    logger.info("=" * 70)

    dataset = TeaRecDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = BSARecRecBole(config, train_data.dataset).to(config["device"])
    logger.info(model)
    logger.info(
        set_color("Trainable parameters", "yellow")
        + f": {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    trainer = Trainer(config, model)

    if args.eval_only:
        incompatible = load_trusted_checkpoint(model, args.checkpoint, config["device"])
        logger.info(
            f"[Checkpoint loaded] {args.checkpoint} | "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
        test_result = trainer.evaluate(
            test_data,
            load_best_model=False,
            show_progress=config["show_progress"],
        )
        payload = {
            "model": "BSARec",
            "paper": "Shin et al., AAAI 2024",
            "dataset": config["dataset"],
            "protocol": "fixed benchmark split; RecBole full-sort",
            "checkpoint": os.path.abspath(args.checkpoint),
            "test_result": float_metrics(test_result),
        }
    else:
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=True,
            show_progress=config["show_progress"],
        )
        incompatible = load_trusted_checkpoint(
            model,
            trainer.saved_model_file,
            config["device"],
        )
        logger.info(
            f"[Best checkpoint loaded] {trainer.saved_model_file} | "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
        test_result = None
        if not args.valid_only:
            test_result = trainer.evaluate(
                test_data,
                load_best_model=False,
                show_progress=config["show_progress"],
            )
        logger.info(set_color("best valid", "yellow") + f": {best_valid_result}")
        if test_result is None:
            logger.info("[VALID ONLY] test set was not evaluated")
        else:
            logger.info(set_color("test result", "yellow") + f": {test_result}")
        payload = {
            "model": "BSARec",
            "paper": "Shin et al., AAAI 2024",
            "dataset": config["dataset"],
            "protocol": "fixed benchmark split; RecBole full-sort",
            "best_checkpoint": trainer.saved_model_file,
            "best_valid_score": float(best_valid_score),
            "best_valid_result": float_metrics(best_valid_result),
            "test_result": (
                None if test_result is None else float_metrics(test_result)
            ),
            "config": {
                "hidden_size": config["hidden_size"],
                "inner_size": config["inner_size"],
                "n_layers": config["n_layers"],
                "n_heads": config["n_heads"],
                "dropout": config["hidden_dropout_prob"],
                "alpha": config["bsarec_alpha"],
                "cutoff": config["bsarec_cutoff"],
                "learning_rate": config["learning_rate"],
                "seed": config["seed"],
            },
        }

    result_path = save_result(args, config, payload)
    logger.info(f"[Saved] JSON: {result_path}")


if __name__ == "__main__":
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    main()
