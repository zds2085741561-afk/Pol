# -*- coding: utf-8 -*-
"""Run MIMARRecBole inside the existing EffiPOI/RecBole pipeline."""

import argparse
import json
import os
import sys
from logging import getLogger

import torch
from torch.cuda import amp
from torch.nn.utils.clip_grad import clip_grad_norm_
from recbole.data import data_preparation
from recbole.trainer import Trainer
from recbole.utils import get_trainer, init_logger, init_seed, set_color
from recbole.utils.utils import get_gpu_usage
from tqdm import tqdm

BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASELINE_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config  # noqa: E402
from data.dataset import TeaRecDataset  # noqa: E402
from mimar_recbole import MIMARRecBole  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="MIMARRecBole formal baseline")
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
    parser.add_argument("--intent_windows", "--windows", default="5,20,50")
    parser.add_argument("--num_intents", type=int, default=4)
    parser.add_argument("--intent_residual_weight", type=float, default=1.0)
    parser.add_argument("--contrastive_temp", type=float, default=0.2)
    parser.add_argument("--proto_weight", type=float, default=0.01)
    parser.add_argument("--seqcl_weight", type=float, default=0.01)
    parser.add_argument("--adv_eps", type=float, default=0.03)
    parser.add_argument("--adv_weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--show_progress", action="store_true", default=True)
    parser.add_argument("--hide_progress", action="store_true")
    parser.add_argument(
        "--log_every",
        type=int,
        default=0,
        help="Print batch-level loss every N batches. 0 keeps one progress bar per epoch.",
    )
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
        "log_every": args.log_every,
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
        "intent_windows": args.intent_windows,
        "num_intents": args.num_intents,
        "intent_residual_weight": args.intent_residual_weight,
        "contrastive_temp": args.contrastive_temp,
        "proto_weight": args.proto_weight,
        "seqcl_weight": args.seqcl_weight,
        "adv_eps": args.adv_eps,
        "adv_weight": args.adv_weight,
        "metrics": ["HIT", "NDCG"],
        "topk": [1, 5, 10, 20],
        "valid_metric": "HIT@10",
        "checkpoint_dir": checkpoint_dir,
        "train_neg_sample_args": None,
    }

    # Keep only the project data/evaluation protocol here. Do not load TeaRec.yaml
    # because it contains E4 teacher-specific switches that should not affect an
    # external comparison baseline.
    props = [os.path.join(PROJECT_ROOT, "props", "fintune.yaml")]
    return Config(
        model=MIMARRecBole,
        dataset=args.dataset,
        config_file_list=props,
        config_dict=config_dict,
    )


class VerboseTrainer(Trainer):
    """Trainer that prints batch-level loss for visible baseline runs."""

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        log_every = int(self.config["log_every"]) if "log_every" in self.config else 20

        iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=120,
                desc=set_color(f"Train {epoch_idx:>5}", "pink"),
            )
            if show_progress
            else train_data
        )

        if not self.config["single_spec"] and train_data.shuffle:
            train_data.sampler.set_epoch(epoch_idx)

        scaler = amp.GradScaler(enabled=self.enable_scaler)
        running_loss = 0.0
        running_count = 0

        for batch_idx, interaction in enumerate(iter_data):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            sync_loss = 0
            if not self.config["single_spec"]:
                self.set_reduce_hook()
                sync_loss = self.sync_grad_loss()

            with torch.autocast(device_type=self.device.type, enabled=self.enable_amp):
                losses = loss_func(interaction)

            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = (
                    loss_tuple
                    if total_loss is None
                    else tuple(map(sum, zip(total_loss, loss_tuple)))
                )
                loss_value = float(loss.detach().cpu())
            else:
                loss = losses
                loss_value = float(loss.detach().cpu())
                total_loss = loss_value if total_loss is None else total_loss + loss_value

            self._check_nan(loss)
            scaler.scale(loss + sync_loss).backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            scaler.step(self.optimizer)
            scaler.update()

            running_loss += loss_value
            running_count += 1
            if log_every > 0 and ((batch_idx + 1) % log_every == 0 or (batch_idx + 1) == len(train_data)):
                avg_loss = running_loss / max(running_count, 1)
                msg = (
                    f"[Train] epoch={epoch_idx:03d} "
                    f"batch={batch_idx + 1:04d}/{len(train_data):04d} "
                    f"loss={loss_value:.6f} avg_loss={avg_loss:.6f}"
                )
                if self.gpu_available:
                    msg += f" gpu={get_gpu_usage(self.device)}"
                print(msg, flush=True)
                running_loss = 0.0
                running_count = 0

            if self.gpu_available and show_progress:
                iter_data.set_postfix_str(
                    set_color(
                        f"loss={loss_value:.4f}, GPU RAM: {get_gpu_usage(self.device)}",
                        "yellow",
                    )
                )

        return total_loss


def main():
    args = parse_args()
    config = build_config(args)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()

    logger.info("=" * 70)
    logger.info("[Baseline] MIMARRecBole formal RecBole/EffiPOI comparison")
    logger.info(
        f"[Dataset] {config['dataset']} | windows={config['intent_windows']} | "
        f"intents={config['num_intents']} | device={config['device']}"
    )
    logger.info("[Protocol] Original Dataset + RecBole Trainer + RecBole full-sort metrics")
    logger.info("=" * 70)

    dataset = TeaRecDataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = MIMARRecBole(config, train_data.dataset).to(config["device"])
    logger.info(model)
    logger.info(set_color("Trainable parameters", "yellow") + f": {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    trainer = VerboseTrainer(config, model)
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
    result_path = os.path.join(result_dir, f"MIMARRecBole_{config['dataset']}_result.json")
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
                    "intent_windows": config["intent_windows"],
                    "num_intents": config["num_intents"],
                    "proto_weight": config["proto_weight"],
                    "seqcl_weight": config["seqcl_weight"],
                    "adv_weight": config["adv_weight"],
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"[Saved] JSON: {result_path}")


if __name__ == "__main__":
    # Keep CUDA selection consistent with the existing project config.
    torch.set_float32_matmul_precision("high") if hasattr(torch, "set_float32_matmul_precision") else None
    main()
