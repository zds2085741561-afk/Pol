# -*- coding: utf-8 -*-
"""Run CL-SASRec inside the existing EffiPOI/RecBole pipeline."""

import argparse
import json
import os
import sys
from logging import getLogger
from time import time

import torch
from recbole.data import data_preparation
from recbole.trainer import Trainer
from recbole.utils import get_trainer, init_logger, init_seed, set_color

BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASELINE_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config  # noqa: E402
from data.dataset import TeaRecDataset  # noqa: E402
from cl_sasrec import CLSASRec  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="CL-SASRec formal baseline")
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
    parser.add_argument("--cl_weight", type=float, default=0.1)
    parser.add_argument("--cl_temperature", type=float, default=0.2)
    parser.add_argument("--aug_types", default="crop,mask,reorder")
    parser.add_argument("--crop_ratio", type=float, default=0.7)
    parser.add_argument("--mask_ratio", type=float, default=0.2)
    parser.add_argument("--reorder_ratio", type=float, default=0.2)
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
        "cl_weight": args.cl_weight,
        "cl_temperature": args.cl_temperature,
        "aug_types": args.aug_types,
        "crop_ratio": args.crop_ratio,
        "mask_ratio": args.mask_ratio,
        "reorder_ratio": args.reorder_ratio,
        "metrics": ["HIT", "NDCG"],
        "topk": [1, 5, 10, 20],
        "valid_metric": "HIT@10",
        "checkpoint_dir": checkpoint_dir,
        "train_neg_sample_args": None,
    }

    props = [os.path.join(PROJECT_ROOT, "props", "fintune.yaml")]
    return Config(
        model=CLSASRec,
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
    logger.info("[Baseline] CL-SASRec formal RecBole/EffiPOI comparison")
    logger.info(
        f"[Dataset] {config['dataset']} | aug={config['aug_types']} | "
        f"cl_weight={config['cl_weight']} | device={config['device']}"
    )
    logger.info("[Protocol] Original Dataset + RecBole Trainer + RecBole full-sort metrics")
    logger.info("=" * 70)

    dataset = TeaRecDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = CLSASRec(config, train_data.dataset).to(config["device"])
    logger.info(model)
    logger.info(set_color("Trainable parameters", "yellow") + f": {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config["show_progress"],
    )
    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=config["show_progress"],
    )

    logger.info(set_color("best valid", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result_dir = os.path.join(BASELINE_DIR, "results")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"CLSASRec_{config['dataset']}_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": config["dataset"],
                "best_valid_score": float(best_valid_score),
                "best_valid_result": dict(best_valid_result),
                "test_result": dict(test_result),
                "config": {
                    "hidden_size": config["hidden_size"],
                    "n_layers": config["n_layers"],
                    "n_heads": config["n_heads"],
                    "cl_weight": config["cl_weight"],
                    "cl_temperature": config["cl_temperature"],
                    "aug_types": config["aug_types"],
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"[Saved] JSON: {result_path}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high") if hasattr(torch, "set_float32_matmul_precision") else None
    main()
