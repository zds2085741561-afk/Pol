# -*- coding: utf-8 -*-
"""Validation-only incremental parameter selection for TF-SID-CPR."""

import argparse
import json
import os
import random
import time
from copy import copy

import numpy as np
import torch

from evaluate import (
    build_flow_community_tokens,
    dataset_immutability_guard,
    evaluate,
    innovation1_immutability_guard,
    load_base_model,
    load_continuous_samples,
    resolve_checkpoint,
    resolve_interaction_file,
    resolve_project_root,
)


def float_grid(text):
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one float")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select formal C4 weights on the validation split only."
    )
    parser.add_argument("-d", "--dataset", default="NYC")
    parser.add_argument(
        "--base_model",
        choices=["teacher", "student"],
        default="student",
        help="formal path base is the frozen S3-KF student",
    )
    parser.add_argument("-p", "--checkpoint", default="")
    parser.add_argument("--project_root", default="")
    parser.add_argument("--horizon", type=int, choices=[2, 3, 5], default=2)
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--expand_topk", type=int, default=30)
    parser.add_argument("--flow_candidate_topk", type=int, default=10)
    parser.add_argument("--flow_candidate_margin", type=float, default=2.5)
    parser.add_argument(
        "--base_preserve_k",
        type=int,
        default=0,
        help="diagnostic only: preserve this many base-model ranking slots",
    )
    parser.add_argument(
        "--rank_replace_k",
        type=int,
        default=0,
        help="diagnostic only: allow path-aware decoding to replace only this many tail slots",
    )
    parser.add_argument("--gtf_grid", type=float_grid, default="0.005,0.01,0.02")
    parser.add_argument("--sid_grid", type=float_grid, default="0.01,0.02,0.05")
    parser.add_argument("--repeat_grid", type=float_grid, default="0.005,0.01,0.02")
    parser.add_argument("--sid_recent", type=int, default=3)
    parser.add_argument("--community_topk", type=int, default=5)
    parser.add_argument("--community_resolution", type=float, default=1.0)
    parser.add_argument("--transition_topk", type=int, default=20)
    parser.add_argument("--transition_eps", type=float, default=1e-6)
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--output_dir", default="results/continuous_distilled")
    parser.add_argument("--force_cuda", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_base_local_rerank", action="store_true")
    args = parser.parse_args()
    for field in ("gtf_grid", "sid_grid", "repeat_grid"):
        value = getattr(args, field)
        if isinstance(value, str):
            setattr(args, field, float_grid(value))
    return args


STAGE_KEYS = (
    "C1-Beam",
    "C2-Beam+GTF",
    "C3-Beam+GTF+TF-SID",
    "C4-TF-SID-CPR-Full",
)
BASELINE_KEY = "C0-Greedy"
C4_KEY = "C4-TF-SID-CPR-Full"
HIT_KEYS = ("avg_hit@1", "avg_hit@5", "avg_hit@10")


def monotonic_summary(stage_metrics):
    # Formal ablation monotonicity must start from C0, not only C1->C4.
    check_keys = [BASELINE_KEY] + list(STAGE_KEYS)
    comparisons = []
    for metric in HIT_KEYS:
        values = [stage_metrics[stage][metric] for stage in check_keys]
        comparisons.extend(
            values[index + 1] > values[index] + 1e-12
            for index in range(len(values) - 1)
        )
    return {
        "strict_hit_gain_count": sum(comparisons),
        "strict_hit_gain_total": len(comparisons),
        "all_hit_metrics_strictly_monotonic": all(comparisons),
    }


def selection_key(stage_metrics):
    """Prefer C4 that stays competitive with C0 and improves path quality."""
    monotonic = monotonic_summary(stage_metrics)
    c0 = stage_metrics[BASELINE_KEY]
    full = stage_metrics[C4_KEY]
    hit1_gain = full["avg_hit@1"] - c0["avg_hit@1"]
    hit5_gain = full["avg_hit@5"] - c0["avg_hit@5"]
    hit10_gain = full["avg_hit@10"] - c0["avg_hit@10"]
    ndcg_gain = full["avg_ndcg@10"] - c0["avg_ndcg@10"]
    route_gain = full["route_recall"] - c0["route_recall"]
    transition_gain = full["transition_validity"] - c0["transition_validity"]
    repeat_drop = c0["repeat_rate"] - full["repeat_rate"]
    hit_floor = int(
        hit1_gain >= -0.008 and hit5_gain >= -0.008 and hit10_gain >= -0.008
    )
    ndcg_not_bad = int(ndcg_gain >= -0.005)
    return (
        hit_floor,
        ndcg_not_bad,
        route_gain,
        repeat_drop,
        transition_gain,
        hit10_gain,
        hit5_gain,
        hit1_gain,
        ndcg_gain,
        int(monotonic["all_hit_metrics_strictly_monotonic"]),
        monotonic["strict_hit_gain_count"],
        full["avg_hit@10"],
        full["route_recall"],
        -full["repeat_rate"],
    )


def evaluate_one(model, dataset_obj, samples, args, method, gtf_beta, sid_beta, repeat):
    trial_args = copy(args)
    trial_args.method = method
    trial_args.gtf_beta = gtf_beta
    trial_args.sid_beta = sid_beta
    trial_args.repeat_penalty = repeat
    trial_args.save_examples = 0
    result, _ = evaluate(model, dataset_obj, samples, trial_args)
    return next(iter(result.values()))


def main():
    args = parse_args()
    if args.force_cuda and args.cpu:
        raise ValueError("Choose only one of --force_cuda and --cpu.")
    if args.beam_size < 10:
        raise ValueError("beam_size must be at least 10 for formal HIT@10 evaluation.")
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
    valid_file = resolve_interaction_file(
        project_root, args.dataset, "valid", explicit_path=""
    )
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    dataset_dir = os.path.join(project_root, "data", args.dataset)
    with innovation1_immutability_guard(project_root), dataset_immutability_guard(
        dataset_dir
    ):
        model, dataset_obj, _ = load_base_model(
            project_root, args.dataset, checkpoint_path, args
        )
        model.tf_sid_communities = build_flow_community_tokens(
            model, args.dataset, args, output_dir
        )
        samples = load_continuous_samples(
            valid_file, dataset_obj, args.horizon, args.max_samples
        )
        if not samples:
            raise ValueError("No eligible validation samples were found.")

        print("\n[Tune] Evaluate C0 Greedy baseline once.")
        c0_metrics = evaluate_one(
            model, dataset_obj, samples, args, "greedy", 0.0, 0.0, 0.0
        )

        print("\n[Tune] Evaluate corrected C1 Beam baseline once.")
        c1_metrics = evaluate_one(
            model, dataset_obj, samples, args, "beam", 0.0, 0.0, 0.0
        )

        c2_cache = {}
        for gtf_beta in args.gtf_grid:
            print(f"\n[Tune] C2 GTF: gtf_beta={gtf_beta}")
            c2_cache[gtf_beta] = evaluate_one(
                model, dataset_obj, samples, args, "beam_gtf", gtf_beta, 0.0, 0.0
            )

        c3_cache = {}
        for gtf_beta in args.gtf_grid:
            for sid_beta in args.sid_grid:
                print(f"\n[Tune] C3 GTF+TF-SID: gtf_beta={gtf_beta}, sid_beta={sid_beta}")
                c3_cache[(gtf_beta, sid_beta)] = evaluate_one(
                    model,
                    dataset_obj,
                    samples,
                    args,
                    "beam_sid",
                    gtf_beta,
                    sid_beta,
                    0.0,
                )

        trials = []
        trial_count = len(args.gtf_grid) * len(args.sid_grid) * len(args.repeat_grid)
        trial_index = 0
        for gtf_beta in args.gtf_grid:
            for sid_beta in args.sid_grid:
                for repeat_penalty in args.repeat_grid:
                    trial_index += 1
                    print(
                        f"\n[Tune] C4 {trial_index}/{trial_count}: "
                        f"gtf_beta={gtf_beta}, sid_beta={sid_beta}, "
                        f"repeat_penalty={repeat_penalty}"
                    )
                    c4_metrics = evaluate_one(
                        model,
                        dataset_obj,
                        samples,
                        args,
                        "full",
                        gtf_beta,
                        sid_beta,
                        repeat_penalty,
                    )
                    stage_metrics = {
                        BASELINE_KEY: c0_metrics,
                        "C1-Beam": c1_metrics,
                        "C2-Beam+GTF": c2_cache[gtf_beta],
                        "C3-Beam+GTF+TF-SID": c3_cache[(gtf_beta, sid_beta)],
                        C4_KEY: c4_metrics,
                    }
                    monotonic = monotonic_summary(stage_metrics)
                    key = selection_key(stage_metrics)
                    hit1_gain = c4_metrics["avg_hit@1"] - c0_metrics["avg_hit@1"]
                    hit5_gain = c4_metrics["avg_hit@5"] - c0_metrics["avg_hit@5"]
                    hit10_gain = (
                        c4_metrics["avg_hit@10"] - c0_metrics["avg_hit@10"]
                    )
                    ndcg_gain = (
                        c4_metrics["avg_ndcg@10"] - c0_metrics["avg_ndcg@10"]
                    )
                    repeat_drop = (
                        c0_metrics["repeat_rate"] - c4_metrics["repeat_rate"]
                    )
                    trial = {
                        "gtf_beta": gtf_beta,
                        "sid_beta": sid_beta,
                        "repeat_penalty": repeat_penalty,
                        "monotonic": monotonic,
                        "c4_vs_c0": {
                            "avg_hit@1_gain": hit1_gain,
                            "avg_hit@5_gain": hit5_gain,
                            "avg_hit@10_gain": hit10_gain,
                            "avg_ndcg@10_gain": ndcg_gain,
                            "route_recall_gain": (
                                c4_metrics["route_recall"]
                                - c0_metrics["route_recall"]
                            ),
                            "transition_validity_gain": (
                                c4_metrics["transition_validity"]
                                - c0_metrics["transition_validity"]
                            ),
                            "repeat_rate_drop": repeat_drop,
                        },
                        "selection_key": list(key),
                        "metrics": c4_metrics,
                        "stage_metrics": stage_metrics,
                    }
                    trials.append(trial)
                    print(
                        f"[Tune] strict_hit_gains="
                        f"{monotonic['strict_hit_gain_count']}/"
                        f"{monotonic['strict_hit_gain_total']}, "
                        f"C4_hit@10={c4_metrics['avg_hit@10']:.6f}, "
                        f"C4-C0_hit@1/5/10="
                        f"{hit1_gain:+.6f}/{hit5_gain:+.6f}/{hit10_gain:+.6f}, "
                        f"C4_ndcg@10={c4_metrics['avg_ndcg@10']:.6f}, "
                        f"C4-C0_ndcg@10={ndcg_gain:+.6f}, "
                        f"repeat_drop={repeat_drop:+.6f}"
                    )

    trials.sort(key=lambda trial: tuple(trial["selection_key"]), reverse=True)
    best = trials[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = os.path.join(
        output_dir, f"{args.dataset}_valid_h{args.horizon}_c4_tuning_{timestamp}.json"
    )
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            {
            "rule": "validation-only parameter selection on the frozen S3-KF Student",
                "base_model": args.base_model,
                "dataset": args.dataset,
                "horizon": args.horizon,
                "checkpoint": checkpoint_path,
                "validation_file": valid_file,
                "max_samples": args.max_samples,
                "selection_rule": (
                    "lexicographic validation selection: keep C4 within the relaxed "
                    "C0 floor on HIT@1/5/10 and NDCG@10, then prioritize route "
                    "recall gain, repeat-rate drop, transition-validity gain, "
                    "followed by single-step gains and C0->C4 monotonicity"
                ),
                "best": best,
                "trials": trials,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== Best Validation Parameters ===")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()
