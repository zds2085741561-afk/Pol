"""Validation-only joint tuning for trajectory-consistency calibration (TCC).

The student checkpoint stays frozen. The script searches the calibration scope,
graph-score scale, and fusion weight on validation data. Test evaluation is
disabled by default so repeated tuning cannot leak test information.
"""

import argparse
import json
import os

from recbole.utils import init_logger, init_seed

from student import StuRec
from student_main import (
    attach_graph,
    build_config_and_data,
    force_e4_full_config,
    safe_torch_load,
)
from trainer import StuRecTrainer


def parse_float_list(value):
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def evaluate_once(config, model, data):
    trainer = StuRecTrainer(config, model, teacher=None)
    return trainer.evaluate(
        data,
        load_best_model=False,
        show_progress=config["show_progress"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="NYC")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rerank_topks", default="5,10,20,30")
    parser.add_argument("--graph_scales", default="40,80,120")
    parser.add_argument(
        "--betas",
        default="0.008,0.010,0.012,0.015,0.018,0.020,0.025",
    )
    parser.add_argument("--hit10_tolerance", type=float, default=0.0)
    parser.add_argument("--hit5_tolerance", type=float, default=0.001)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--evaluate_test",
        action="store_true",
        help="Evaluate test once after validation selection. Omit while tuning.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    topks = parse_int_list(args.rerank_topks)
    scales = parse_float_list(args.graph_scales)
    betas = parse_float_list(args.betas)
    if not topks or not scales or not betas:
        raise ValueError("rerank_topks, graph_scales, and betas must be non-empty")

    kwargs = force_e4_full_config(
        {
            "seed": args.seed,
            "pure_student": True,
            "teacher_variant": "T2",
            "use_tfkd": True,
            "use_feature_kd": False,
            "use_tfpq": False,
            "eval_batch_size": args.eval_batch_size,
        },
        dataset_name=args.dataset,
    )
    config, dataset, train_data, valid_data, test_data, graph_state = (
        build_config_and_data(args.dataset, kwargs)
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    for obj in (dataset, train_data, valid_data, test_data):
        attach_graph(obj, graph_state)

    model = StuRec(config, train_data.dataset).to(config.device)
    checkpoint = safe_torch_load(args.checkpoint, map_location=config.device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.load_other_parameter(checkpoint.get("other_parameter"))
    model.eval()

    model.use_local_rerank = True
    model.local_repeat_beta = 0.0
    model.local_interest_beta = 0.0

    model.local_rerank_topk = topks[0]
    model.graph_logit_scale = scales[0]
    model.local_graph_beta = 0.0
    baseline = dict(evaluate_once(config, model, valid_data))
    print(
        "[Baseline] "
        f"hit@5={baseline['hit@5']:.4f} "
        f"hit@10={baseline['hit@10']:.4f} "
        f"ndcg@10={baseline['ndcg@10']:.4f}"
    )

    rows = []
    for topk in topks:
        model.local_rerank_topk = topk
        for scale in scales:
            model.graph_logit_scale = scale
            for beta in betas:
                model.local_graph_beta = beta
                result = evaluate_once(config, model, valid_data)
                row = {
                    "rerank_topk": topk,
                    "graph_scale": scale,
                    "beta": beta,
                    **dict(result),
                }
                row["eligible"] = bool(
                    row["hit@10"]
                    >= baseline["hit@10"] - args.hit10_tolerance
                    and row["hit@5"]
                    >= baseline["hit@5"] - args.hit5_tolerance
                )
                rows.append(row)
                print(
                    f"[Valid] topk={topk} scale={scale:g} beta={beta:g} "
                    f"hit@5={result['hit@5']:.4f} "
                    f"hit@10={result['hit@10']:.4f} "
                    f"ndcg@10={result['ndcg@10']:.4f} "
                    f"eligible={row['eligible']}"
                )

    eligible_rows = [row for row in rows if row["eligible"]]
    selection_pool = eligible_rows or rows
    best = max(
        selection_pool,
        key=lambda row: (
            row["ndcg@10"],
            row["ndcg@5"],
            row["hit@1"],
            row["hit@10"],
            -row["beta"],
        ),
    )

    payload = {
        "variant": "TCC validation-only joint tuning",
        "dataset": args.dataset,
        "checkpoint": os.path.abspath(args.checkpoint),
        "selection_rule": {
            "constraints": {
                "hit@10_tolerance": args.hit10_tolerance,
                "hit@5_tolerance": args.hit5_tolerance,
            },
            "objective": ["ndcg@10", "ndcg@5", "hit@1", "hit@10"],
        },
        "baseline_validation": baseline,
        "selected": best,
        "validation": rows,
    }

    if args.evaluate_test:
        model.local_rerank_topk = best["rerank_topk"]
        model.graph_logit_scale = best["graph_scale"]
        model.local_graph_beta = best["beta"]
        payload["test"] = dict(evaluate_once(config, model, test_data))

    print(
        "[Selected] "
        f"topk={best['rerank_topk']} "
        f"scale={best['graph_scale']:g} "
        f"beta={best['beta']:g} "
        f"valid_hit@10={best['hit@10']:.4f} "
        f"valid_ndcg@10={best['ndcg@10']:.4f}"
    )
    if "test" in payload:
        print("[Test] " + json.dumps(payload["test"], ensure_ascii=False))
    else:
        print("[Validation only] Test set was not evaluated.")

    output = args.output or os.path.join(
        "results",
        f"TCC_joint_validation_{args.dataset}.json",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"[Saved] {os.path.abspath(output)}")


if __name__ == "__main__":
    main()
