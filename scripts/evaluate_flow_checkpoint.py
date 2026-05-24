import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import h5py
import matplotlib
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import ReWiNDTransformer


def trajectory_keys(group):
    return [
        key
        for key in group.keys()
        if "lang" not in key
        and not key.startswith("flow_progress_")
        and not key.startswith("flow_signal_")
    ]


def normalize_sequence(values, max_length, pad_value=None):
    values = np.asarray(values)
    if values.shape[0] == 0:
        raise ValueError("Cannot normalize an empty sequence.")

    if values.shape[0] > max_length:
        indices = np.linspace(0, values.shape[0] - 1, max_length).astype(int)
        return values[indices], indices

    indices = np.arange(values.shape[0])
    if values.shape[0] < max_length:
        if pad_value is None:
            pad_value = values[-1]
        pad_shape = (max_length - values.shape[0],) + values.shape[1:]
        pad = np.broadcast_to(pad_value, pad_shape)
        values = np.concatenate([values, pad], axis=0)
    return values, indices


def safe_corr(pred, target):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.shape[0] < 2 or np.allclose(pred, pred[0]) or np.allclose(target, target[0]):
        return 0.0, 0.0
    pearson = pearsonr(pred, target).statistic
    spearman = spearmanr(pred, target).statistic
    if not np.isfinite(pearson):
        pearson = 0.0
    if not np.isfinite(spearman):
        spearman = 0.0
    return float(pearson), float(spearman)


def checkpoint_label(path):
    stem = Path(path).stem
    if stem.startswith("rewind_metaworld_"):
        stem = stem.replace("rewind_metaworld_", "")
    return stem


def build_target(group, traj_key, target_type):
    video_len = group[traj_key].shape[0]
    if target_type == "linear":
        return np.linspace(1.0 / video_len, 1.0, video_len, dtype=np.float32), 0.0

    progress_key = f"flow_progress_{traj_key}"
    signal_key = f"flow_signal_{traj_key}"
    if progress_key not in group or signal_key not in group:
        return None, None
    target = np.asarray(group[progress_key], dtype=np.float32)
    flow_signal = np.asarray(group[signal_key], dtype=np.float32)
    return target, float(flow_signal.max())


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint["args"]
    model = ReWiNDTransformer(
        args=model_args,
        video_dim=768,
        text_dim=384,
        hidden_dim=512,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model_args, model


def evaluate_checkpoint(
    checkpoint_path,
    h5_path,
    output_dir,
    max_groups,
    max_trajs_per_group,
    target_type,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args, model = load_model(checkpoint_path, device)
    max_length = int(model_args.max_length)
    label = f"{checkpoint_label(checkpoint_path)}_{target_type}"
    plot_dir = output_dir / label
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    plotted = 0
    with h5py.File(h5_path, "r") as h5_file, torch.no_grad():
        group_names = list(h5_file.keys())
        if max_groups > 0:
            group_names = group_names[:max_groups]

        for group_name in group_names:
            group = h5_file[group_name]
            text = np.asarray(group["minilm_lang_embedding"], dtype=np.float32)[0]
            text_tensor = torch.from_numpy(text).unsqueeze(0).to(device).float()

            keys = trajectory_keys(group)
            if max_trajs_per_group > 0:
                keys = keys[:max_trajs_per_group]

            for traj_key in keys:
                video = np.asarray(group[traj_key], dtype=np.float32)
                target, signal_max = build_target(group, traj_key, target_type)
                if target is None:
                    print(f"Skipping {group_name}/{traj_key}: missing flow target.")
                    continue
                if video.shape[0] != target.shape[0]:
                    print(
                        f"Skipping {group_name}/{traj_key}: video len {video.shape[0]} "
                        f"!= target len {target.shape[0]}"
                    )
                    continue

                video_norm, kept_indices = normalize_sequence(video, max_length)
                target_norm, _ = normalize_sequence(target, max_length, pad_value=target[-1])
                valid_len = min(len(kept_indices), max_length)

                video_tensor = torch.from_numpy(video_norm).unsqueeze(0).to(device).float()
                pred = model(video_tensor, text_tensor).squeeze(0).squeeze(-1).detach().cpu().numpy()

                pred_valid = pred[:valid_len]
                target_valid = target_norm[:valid_len]
                pearson, spearman = safe_corr(pred_valid, target_valid)
                mse = float(np.mean((pred_valid - target_valid) ** 2))
                target_final = float(target[-1])
                target_max = float(target.max())

                rows.append(
                    {
                        "checkpoint": label,
                        "target_type": target_type,
                        "group": group_name,
                        "traj": traj_key,
                        "raw_len": int(video.shape[0]),
                        "eval_len": int(valid_len),
                        "target_start": float(target[0]),
                        "target_end": target_final,
                        "target_max": target_max,
                        "flow_signal_max": signal_max,
                        "pred_start": float(pred_valid[0]),
                        "pred_end": float(pred_valid[-1]),
                        "pred_min": float(pred_valid.min()),
                        "pred_max": float(pred_valid.max()),
                        "pearson": pearson,
                        "spearman": spearman,
                        "mse": mse,
                    }
                )

                fig, ax = plt.subplots(figsize=(7, 4))
                x = np.arange(valid_len)
                ax.plot(x, target_valid, label=f"target {target_type}", linewidth=2)
                ax.plot(x, pred_valid, label="predicted progress", linewidth=2)
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(
                    f"{group_name} / traj {traj_key}\n"
                    f"pearson={pearson:.3f}, spearman={spearman:.3f}, mse={mse:.4f}"
                )
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best")
                fig.tight_layout()
                safe_group = "".join(c if c.isalnum() or c in "-_." else "_" for c in group_name)[:80]
                fig.savefig(plot_dir / f"{safe_group}_traj_{traj_key}.png", dpi=150)
                plt.close(fig)
                plotted += 1

    csv_path = output_dir / f"{label}_summary.csv"
    fieldnames = [
        "checkpoint",
        "target_type",
        "group",
        "traj",
        "raw_len",
        "eval_len",
        "target_start",
        "target_end",
        "target_max",
        "flow_signal_max",
        "pred_start",
        "pred_end",
        "pred_min",
        "pred_max",
        "pearson",
        "spearman",
        "mse",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        pearsons = np.asarray([row["pearson"] for row in rows], dtype=np.float32)
        spearmans = np.asarray([row["spearman"] for row in rows], dtype=np.float32)
        mses = np.asarray([row["mse"] for row in rows], dtype=np.float32)
        bad_targets = sum(
            not np.isclose(row["target_end"], 1.0, atol=1e-4)
            for row in rows
            if target_type == "linear" or row["flow_signal_max"] > 0
        )
        print(f"{label}: wrote {plotted} plots to {plot_dir}")
        print(f"{label}: wrote summary to {csv_path}")
        print(
            f"{label}: pearson mean={pearsons.mean():.4f}, "
            f"spearman mean={spearmans.mean():.4f}, mse mean={mses.mean():.4f}"
        )
        print(f"{label}: full targets not ending at 1: {bad_targets}/{len(rows)}")
    else:
        print(f"{label}: no trajectories evaluated.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone checkpoint sanity check for optical-flow reward models. "
            "Plots full-trajectory targets against model predictions on H5 trajectories."
        )
    )
    parser.add_argument("--checkpoint-paths", nargs="+", required=True)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--output-dir", default="flow_checkpoint_eval")
    parser.add_argument(
        "--target-type",
        choices=["linear", "optical_flow"],
        default="optical_flow",
        help="Target to plot against predictions. Use linear for baseline ReWiND checkpoints.",
    )
    parser.add_argument("--max-groups", type=int, default=5)
    parser.add_argument("--max-trajs-per-group", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint_path in args.checkpoint_paths:
        evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            h5_path=args.h5_path,
            output_dir=output_dir,
            max_groups=args.max_groups,
            max_trajs_per_group=args.max_trajs_per_group,
            target_type=args.target_type,
        )


if __name__ == "__main__":
    main()
