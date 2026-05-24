import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import numpy as np
import torch
from scipy.stats import spearmanr

from model import ReWiNDTransformer
from utils.utils import generate_rewind_data


def checkpoint_label(path):
    stem = Path(path).stem
    if stem.startswith("rewind_metaworld_"):
        stem = stem.replace("rewind_metaworld_", "")
    parent = Path(path).parent.name
    return f"{parent}_{stem}"


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    model = ReWiNDTransformer(
        args=args,
        video_dim=768,
        text_dim=384,
        hidden_dim=512,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return args, model


def compute_ranking_rows(all_fail_matrix, close_success_matrix, success_matrix, tasks):
    fail_diag = np.diag(all_fail_matrix)
    close_diag = np.diag(close_success_matrix)
    success_diag = np.diag(success_matrix)
    gt_ranks = np.array([3, 2, 1])

    rows = []
    spearmans = []
    correct = 0
    for task, fail, close, success in zip(tasks, fail_diag, close_diag, success_diag):
        values = np.asarray([fail, close, success], dtype=np.float32)
        sorted_indices = np.argsort(values)
        ranks = np.zeros(3, dtype=np.int32)
        ranks[sorted_indices[0]] = 3
        ranks[sorted_indices[1]] = 2
        ranks[sorted_indices[2]] = 1

        is_correct = bool(fail < close < success)
        correct += int(is_correct)
        rho = spearmanr(gt_ranks, ranks).statistic
        if not np.isfinite(rho):
            rho = 0.0
        spearmans.append(float(rho))

        rows.append(
            {
                "task": task,
                "all_fail_reward": float(fail),
                "close_success_reward": float(close),
                "success_reward": float(success),
                "all_fail_rank": int(ranks[0]),
                "close_success_rank": int(ranks[1]),
                "success_rank": int(ranks[2]),
                "correct_order_all_fail_lt_close_lt_success": is_correct,
                "spearman_with_gt_ranking": float(rho),
                "success_minus_close": float(success - close),
                "close_minus_fail": float(close - fail),
                "success_minus_fail": float(success - fail),
            }
        )

    summary = {
        "num_tasks": len(rows),
        "ground_truth_ranking_success_rate": correct / len(rows) if rows else 0.0,
        "average_reward_ranking_spearman": float(np.mean(spearmans)) if spearmans else 0.0,
        "mean_all_fail_reward": float(np.mean(fail_diag)) if len(fail_diag) else 0.0,
        "mean_close_success_reward": float(np.mean(close_diag)) if len(close_diag) else 0.0,
        "mean_success_reward": float(np.mean(success_diag)) if len(success_diag) else 0.0,
        "mean_success_minus_close": float(np.mean(success_diag - close_diag)) if len(success_diag) else 0.0,
        "mean_close_minus_fail": float(np.mean(close_diag - fail_diag)) if len(close_diag) else 0.0,
        "mean_success_minus_fail": float(np.mean(success_diag - fail_diag)) if len(success_diag) else 0.0,
    }
    return rows, summary


def evaluate_checkpoint(checkpoint_path, paths, output_dir, set_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args, model = load_model(checkpoint_path, device)
    label = checkpoint_label(checkpoint_path)

    with (
        h5py.File(paths["success"], "r") as success_h5,
        h5py.File(paths["close_success"], "r") as close_h5,
        h5py.File(paths["all_fail"], "r") as fail_h5,
        open(paths["task_list"], "r") as task_file,
    ):
        task_list = json.load(task_file)
        success_matrix, _, tasks, _ = generate_rewind_data(
            h5_file=success_h5,
            task_subset=task_list,
            set_type=set_type,
            rewind_model=model,
            args=model_args,
            device=str(device),
        )
        close_matrix, _, _, _ = generate_rewind_data(
            h5_file=close_h5,
            task_subset=task_list,
            set_type=set_type,
            rewind_model=model,
            args=model_args,
            device=str(device),
        )
        fail_matrix, _, _, _ = generate_rewind_data(
            h5_file=fail_h5,
            task_subset=task_list,
            set_type=set_type,
            rewind_model=model,
            args=model_args,
            device=str(device),
        )

    rows, summary = compute_ranking_rows(fail_matrix, close_matrix, success_matrix, tasks)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{label}_ranking_rows.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["task"])
        writer.writeheader()
        writer.writerows(rows)

    summary["checkpoint"] = label
    summary_path = output_dir / f"{label}_ranking_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"{label}: wrote {csv_path}")
    print(f"{label}: wrote {summary_path}")
    print(
        f"{label}: ranking_success_rate={summary['ground_truth_ranking_success_rate']:.4f}, "
        f"avg_spearman={summary['average_reward_ranking_spearman']:.4f}, "
        f"success-fail={summary['mean_success_minus_fail']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone ReWiND policy-rollout reward-ranking eval. "
            "Ranks all_fail, close_success, and success demos by diagonal reward."
        )
    )
    parser.add_argument("--checkpoint-paths", nargs="+", required=True)
    parser.add_argument("--success-h5", default="datasets_rewind_scratch/metaworld_embeddings_eval.h5")
    parser.add_argument(
        "--close-success-h5",
        default="datasets_rewind_scratch/metaworld_dino_embeddings_eval_close_succ.h5",
    )
    parser.add_argument(
        "--all-fail-h5",
        default="datasets_rewind_scratch/metaworld_dino_embeddings_eval_all_fail.h5",
    )
    parser.add_argument("--task-list", default="utils/new_task_v2.json")
    parser.add_argument("--set-type", choices=["train", "eval", "test"], default="eval")
    parser.add_argument("--output-dir", default="reward_ranking_eval")
    args = parser.parse_args()

    paths = {
        "success": args.success_h5,
        "close_success": args.close_success_h5,
        "all_fail": args.all_fail_h5,
        "task_list": args.task_list,
    }
    output_dir = Path(args.output_dir)
    for checkpoint_path in args.checkpoint_paths:
        evaluate_checkpoint(checkpoint_path, paths, output_dir, args.set_type)


if __name__ == "__main__":
    main()
