"""Validation-selected trajectory-flow reranking for a frozen S1 student.

The S1 checkpoint and all model parameters remain frozen. Only the inference
time Top-K trajectory-flow reranking strength is selected on validation data;
the test set is evaluated once with the selected value.
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
    parser.add_argument("--betas", default="0,0.001,0.002,0.005,0.01,0.02")
    parser.add_argument("--rerank_topk", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    kwargs = force_e4_full_config(
        {
            "seed": args.seed,
            "pure_student": True,
            "teacher_variant": "T1",
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
    model.local_rerank_topk = args.rerank_topk
    model.local_repeat_beta = 0.0
    model.local_interest_beta = 0.0

    validation_rows = []
    for beta in parse_float_list(args.betas):
        model.local_graph_beta = beta
        result = evaluate_once(config, model, valid_data)
        row = {"beta": beta, **dict(result)}
        validation_rows.append(row)
        print(
            f"[Valid] beta={beta:g} "
            f"hit@10={result['hit@10']:.4f} "
            f"ndcg@10={result['ndcg@10']:.4f}"
        )

    best = max(
        validation_rows,
        key=lambda row: (row["hit@10"], row["ndcg@10"], -row["beta"]),
    )
    model.local_graph_beta = best["beta"]
    test_result = evaluate_once(config, model, test_data)
    payload = {
        "dataset": args.dataset,
        "checkpoint": os.path.abspath(args.checkpoint),
        "rerank_topk": args.rerank_topk,
        "selection_metric": ["hit@10", "ndcg@10"],
        "validation": validation_rows,
        "selected_beta": best["beta"],
        "test": dict(test_result),
    }

    print(
        f"[Selected] beta={best['beta']:g} "
        f"valid_hit@10={best['hit@10']:.4f} "
        f"valid_ndcg@10={best['ndcg@10']:.4f}"
    )
    print("[Test] " + json.dumps(payload["test"], ensure_ascii=False))

    output = args.output or os.path.join(
        "results",
        f"S1_TrajectoryFlowRerank_{args.dataset}.json",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    print(f"[Saved] {os.path.abspath(output)}")


if __name__ == "__main__":
    main()
