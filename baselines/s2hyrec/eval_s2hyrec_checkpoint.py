# -*- coding: utf-8 -*-
"""Evaluate a trained S2HyRec checkpoint on the fixed test split only."""

import argparse
import json
import os
import sys

import torch
from recbole.data import data_preparation
from recbole.trainer import Trainer
from recbole.utils import init_logger, init_seed

from run_s2hyrec import S2HyRecRecBole, TeaRecDataset, build_config, parse_args


def load_trusted_state_dict(model, checkpoint, device):
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


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", required=True)
    eval_args, remaining_args = parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining_args]
        base_args = parse_args()
    finally:
        sys.argv = original_argv

    checkpoint = os.path.abspath(eval_args.checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config = build_config(base_args)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = TeaRecDataset(config)
    train_data, _, test_data = data_preparation(config, dataset)
    model = S2HyRecRecBole(config, train_data.dataset).to(config["device"])
    incompatible = load_trusted_state_dict(model, checkpoint, config["device"])
    print(
        "[Checkpoint loaded] "
        f"missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )

    trainer = Trainer(config, model)
    test_result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=config["show_progress"],
    )

    payload = {
        "model": "S2HyRec",
        "dataset": config["dataset"],
        "split": "test",
        "protocol": "RecBole full-sort",
        "checkpoint": checkpoint,
        "test_result": {key: float(value) for key, value in test_result.items()},
    }
    result_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(
        result_dir,
        f"S2HyRec_{config['dataset']}_test_only.json",
    )
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("[TEST ONLY] S2HyRec")
    print(f"[Dataset] {config['dataset']}")
    print(f"[Checkpoint] {checkpoint}")
    print(f"[Test result] {dict(test_result)}")
    print(f"[Saved] {result_path}")
    print("=" * 70)


if __name__ == "__main__":
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    main()
