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
from tqdm import tqdm

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


def normalize_embedding_sequence(video_embedding, max_length):
    video_embedding = np.asarray(video_embedding, dtype=np.float32)
    if video_embedding.ndim == 1:
        video_embedding = video_embedding.reshape(1, -1)
    if video_embedding.shape[0] == 0:
        return None
    if video_embedding.shape[0] < max_length:
        pad = np.repeat(video_embedding[:1], max_length - video_embedding.shape[0], axis=0)
        video_embedding = np.concatenate([pad, video_embedding], axis=0)
    elif video_embedding.shape[0] > max_length:
        frame_idx = np.linspace(0, video_embedding.shape[0] - 1, max_length).astype(int)
        video_embedding = video_embedding[frame_idx]
    return video_embedding.astype(np.float32)


def checkpoint_label(path):
    stem = Path(path).stem
    if stem.startswith("rewind_metaworld_"):
        stem = stem.replace("rewind_metaworld_", "")
    return stem


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


def compute_confusion_matrix(h5_path, model, max_length, max_envs):
    device = next(model.parameters()).device
    with h5py.File(h5_path, "r") as h5_file:
        envs = list(h5_file.keys())
        if max_envs > 0:
            envs = envs[:max_envs]

        text_embeddings = []
        for env in envs:
            embedding = np.asarray(h5_file[env]["minilm_lang_embedding"], dtype=np.float32)[0]
            text_embeddings.append(embedding.reshape(1, -1))
        text_embeddings = torch.from_numpy(np.concatenate(text_embeddings, axis=0)).to(device).float()

        matrix_rows = []
        plotted_envs = []
        with torch.no_grad():
            for env in tqdm(envs, desc=f"Computing {Path(h5_path).name}"):
                trajs = []
                for key in trajectory_keys(h5_file[env]):
                    video_embedding = normalize_embedding_sequence(h5_file[env][key], max_length)
                    if video_embedding is not None:
                        trajs.append(video_embedding)
                if not trajs:
                    continue

                feature_dims = {traj.shape[1] for traj in trajs}
                if len(feature_dims) != 1:
                    print(f"Skipping {env}: inconsistent embedding dims {sorted(feature_dims)}")
                    continue

                traj_tensor = torch.from_numpy(np.stack(trajs, axis=0)).to(device).float()
                env_scores = []
                for traj in traj_tensor:
                    batch = traj.unsqueeze(0).repeat(text_embeddings.shape[0], 1, 1)
                    pred = model(batch, text_embeddings)[:, -1].squeeze(-1)
                    env_scores.append(pred.detach().cpu().numpy())
                matrix_rows.append(np.mean(np.stack(env_scores, axis=0), axis=0))
                plotted_envs.append(env)

    return np.asarray(matrix_rows, dtype=np.float32), plotted_envs, envs


def save_matrix(matrix, row_names, col_names, output_prefix, title):
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_prefix.with_suffix(".csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + col_names)
        for row_name, row in zip(row_names, matrix):
            writer.writerow([row_name] + [float(v) for v in row])

    if matrix.size == 0:
        print(f"No matrix rows for {title}; wrote only {csv_path}")
        return

    m_min = float(matrix.min())
    m_max = float(matrix.max())
    if m_max == m_min:
        matrix_plot = np.zeros_like(matrix)
    else:
        matrix_plot = (matrix - m_min) / (m_max - m_min)

    fig, ax = plt.subplots(figsize=(max(6, len(col_names) * 0.45), max(5, len(row_names) * 0.45)))
    im = ax.imshow(matrix_plot, cmap="Blues", interpolation="nearest", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Text prompt")
    ax.set_ylabel("Trajectory group")
    ax.set_xticks(range(len(col_names)))
    ax.set_yticks(range(len(row_names)))
    ax.set_xticklabels(col_names, rotation=90, fontsize=6)
    ax.set_yticklabels(row_names, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    png_path = output_prefix.with_suffix(".png")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot local confusion matrices for a saved ReWiND reward-model checkpoint."
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--split-name", default="eval")
    parser.add_argument("--output-dir", default="checkpoint_confusion_matrices")
    parser.add_argument("--max-envs", type=int, default=-1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args, model = load_model(args.checkpoint_path, device)
    label = checkpoint_label(args.checkpoint_path)

    matrix, row_names, col_names = compute_confusion_matrix(
        h5_path=args.h5_path,
        model=model,
        max_length=int(model_args.max_length),
        max_envs=args.max_envs,
    )
    output_prefix = Path(args.output_dir) / f"{label}_{args.split_name}_confusion_matrix"
    save_matrix(
        matrix=matrix,
        row_names=row_names,
        col_names=col_names,
        output_prefix=output_prefix,
        title=f"{label} {args.split_name} confusion matrix",
    )


if __name__ == "__main__":
    main()
