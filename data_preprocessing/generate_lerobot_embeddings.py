import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from data_preprocessing.generate_openx_bridge_embeddings import (
    DINO_BATCH_SIZE,
    center_crop,
    encode_text,
    get_dino_embeddings,
    sanitize_h5_key,
    sample_frames,
)
from utils.progress_utils import compute_frame_diff_progress


def _normalize_camera_key(camera_key):
    if camera_key.startswith("observation.images."):
        return camera_key
    return f"observation.images.{camera_key}"


def _has_lerobot_data(dataset_dir):
    return bool(list((dataset_dir / "data").glob("**/*.parquet")))


def _resolve_dataset_dir(dataset_dir):
    dataset_dir = Path(dataset_dir)
    if _has_lerobot_data(dataset_dir):
        return dataset_dir

    candidates = []
    for child in dataset_dir.iterdir() if dataset_dir.exists() else []:
        if child.is_dir() and _has_lerobot_data(child):
            candidates.append(child)

    if len(candidates) == 1:
        return candidates[0]

    found = sorted(str(path) for path in dataset_dir.glob("**/data/**/*.parquet"))[:20]
    hint = "\n".join(found) if found else "No parquet files found below this path."
    raise FileNotFoundError(
        f"No LeRobot parquet files found under {dataset_dir}/data. "
        f"Set DATASET_DIR to the directory containing data/, meta/, and videos/.\n{hint}"
    )


def _read_parquet_files(paths):
    frames = []
    for path in paths:
        frame = pd.read_parquet(path)
        frame["_source_parquet"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No parquet files found under data/")
    return pd.concat(frames, ignore_index=True)


def _iter_episode_rows(data_files):
    for path in data_files:
        data = pd.read_parquet(path)
        data["_source_parquet"] = str(path)

        if "episode_index" in data.columns:
            grouped = data.groupby("episode_index", sort=True)
        else:
            data["_episode_index"] = path.as_posix()
            grouped = data.groupby("_episode_index", sort=True)

        for _, rows in grouped:
            yield rows


def _load_tasks(dataset_dir):
    tasks_path = dataset_dir / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        return {}
    tasks = pd.read_parquet(tasks_path)
    if "task_index" not in tasks.columns:
        return {}
    text_col = None
    for candidate in ("task", "language_instruction", "instruction"):
        if candidate in tasks.columns:
            text_col = candidate
            break
    if text_col is None:
        return {}
    return {
        int(row["task_index"]): str(row[text_col])
        for _, row in tasks.iterrows()
        if not pd.isna(row[text_col])
    }


def _episode_instruction(rows, task_map, dataset_name):
    if "language_instruction" in rows.columns:
        value = rows["language_instruction"].dropna()
        if len(value):
            return str(value.iloc[0])
    if "task" in rows.columns:
        value = rows["task"].dropna()
        if len(value):
            return str(value.iloc[0])
    if "task_index" in rows.columns:
        value = rows["task_index"].dropna()
        if len(value):
            task_index = int(value.iloc[0])
            if task_index in task_map:
                return task_map[task_index]
    return f"{dataset_name}_episode"


def _row_to_video_ref(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "as_py"):
        return _row_to_video_ref(value.as_py())
    return None


def _video_roots(dataset_dir, camera_col):
    videos_dir = dataset_dir / "videos"
    roots = [videos_dir / camera_col]
    prefix = "observation.images."
    camera_names = [camera_col]
    if camera_col.startswith(prefix):
        roots.append(videos_dir / camera_col[len(prefix) :])
        camera_names.append(camera_col[len(prefix) :])

    if videos_dir.exists():
        for camera_name in camera_names:
            roots.extend(path for path in videos_dir.glob(f"**/{camera_name}") if path.is_dir())

    seen = set()
    unique_roots = []
    for root in roots:
        if root not in seen:
            unique_roots.append(root)
            seen.add(root)
    return unique_roots


def _candidate_video_names(file_index):
    return [
        f"file_{file_index:06d}.mp4",
        f"episode_{file_index:06d}.mp4",
        f"file-{file_index:03d}.mp4",
    ]


def _candidate_video_paths(dataset_dir, camera_col, source_parquet):
    if not source_parquet:
        return []
    match = re.search(r"chunk-(\d+)/(?:file[-_]|episode_)(\d+)\.parquet$", source_parquet)
    if not match:
        return []

    chunk_index = int(match.group(1))
    file_index = int(match.group(2))
    chunk_name = f"chunk-{chunk_index:03d}"
    names = _candidate_video_names(file_index)
    candidates = []
    for root in _video_roots(dataset_dir, camera_col):
        for name in names:
            candidates.append(root / chunk_name / name)
            candidates.append(root / name)
            candidates.append(dataset_dir / "videos" / chunk_name / root.name / name)
    return candidates


def _resolve_video_path(dataset_dir, camera_col, ref, source_parquet=None):
    if ref and ref.get("path"):
        path = Path(ref["path"])
        return path if path.is_absolute() else dataset_dir / path

    video_roots = [root for root in _video_roots(dataset_dir, camera_col) if root.exists()]
    if not video_roots:
        expected = ", ".join(str(root) for root in _video_roots(dataset_dir, camera_col))
        raise FileNotFoundError(f"Missing LeRobot video directory. Expected one of: {expected}")

    for candidate in _candidate_video_paths(dataset_dir, camera_col, source_parquet):
        if candidate.exists():
            return candidate

    videos = sorted(video for root in video_roots for video in root.glob("**/*.mp4"))
    if len(videos) == 1:
        return videos[0]
    raise FileNotFoundError(
        f"Could not infer video file for {camera_col}; found {len(videos)} videos"
    )


def _read_video_frame(video_path, frame_index=None, timestamp=None):
    reader = imageio.get_reader(str(video_path), "ffmpeg")
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps") or 0.0)
        if frame_index is None:
            if timestamp is None or fps <= 0:
                frame_index = 0
            else:
                frame_index = int(round(float(timestamp) * fps))
        frame = reader.get_data(max(0, int(frame_index)))
    finally:
        reader.close()
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    return np.asarray(frame[:, :, :3], dtype=np.uint8)


def _load_frame(dataset_dir, row, camera_col):
    value = row[camera_col] if camera_col in row.index else None
    ref = _row_to_video_ref(value)
    if ref:
        frame_index = ref.get("frame_index")
        timestamp = ref.get("timestamp")
    else:
        frame_index = row.get("frame_index", None)
        timestamp = row.get("timestamp", None)

    video_path = _resolve_video_path(
        dataset_dir,
        camera_col,
        ref,
        source_parquet=row.get("_source_parquet", None),
    )
    return _read_video_frame(video_path, frame_index=frame_index, timestamp=timestamp)


def build_lerobot_h5(
    dataset_name,
    dataset_dir,
    output_path,
    max_length,
    max_episodes,
    camera_key,
):
    dataset_dir = _resolve_dataset_dir(dataset_dir)
    camera_col = _normalize_camera_key(camera_key)

    data_files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir / 'data'}")

    first_data = pd.read_parquet(data_files[0])
    video_roots = _video_roots(dataset_dir, camera_col)
    if camera_col not in first_data.columns and not any(root.exists() for root in video_roots):
        video_dirs = sorted(str(path.relative_to(dataset_dir)) for path in (dataset_dir / "videos").glob("**/*") if path.is_dir())[:30]
        raise KeyError(
            f"Could not find camera '{camera_col}'. Available columns include: "
            f"{[c for c in first_data.columns if 'image' in c or 'video' in c][:20]}. "
            f"Available video dirs include: {video_dirs}"
        )
    del first_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_model = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14", force_reload=False
    ).to(device)
    minilm_tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L12-v2"
    )
    minilm_model = AutoModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L12-v2"
    ).to(device)

    task_map = _load_tasks(dataset_dir)
    per_group_counts = defaultdict(int)

    with h5py.File(output_path, "w") as h5_file:
        episodes = _iter_episode_rows(data_files)
        for episode_i, rows in enumerate(tqdm(episodes, desc=f"Processing {dataset_name}")):
            if max_episodes > 0 and episode_i >= max_episodes:
                break
            rows = rows.sort_values("frame_index") if "frame_index" in rows.columns else rows
            sampled_rows = rows.iloc[
                np.linspace(0, len(rows) - 1, min(max_length, len(rows)), dtype=int)
            ]
            frames = [_load_frame(dataset_dir, row, camera_col) for _, row in sampled_rows.iterrows()]
            if not frames:
                continue

            if len(frames) < max_length:
                frames = sample_frames(frames, max_length)
            sampled_frames = [center_crop(frame, 224) for frame in frames]
            instruction = _episode_instruction(rows, task_map, dataset_name)
            group_name = sanitize_h5_key(instruction)
            if group_name not in h5_file:
                h5_file.create_group(group_name)

            flow_progress, flow_signal = compute_frame_diff_progress(sampled_frames)
            dino_embeddings = get_dino_embeddings(
                sampled_frames, dino_model=dino_model, device=device
            )

            traj_id = str(per_group_counts[group_name])
            per_group_counts[group_name] += 1

            h5_file[group_name].create_dataset(traj_id, data=dino_embeddings)
            h5_file[group_name].create_dataset(f"flow_progress_{traj_id}", data=flow_progress)
            h5_file[group_name].create_dataset(f"flow_signal_{traj_id}", data=flow_signal)
            if "minilm_lang_embedding" not in h5_file[group_name]:
                lang_embedding = encode_text(
                    instruction,
                    tokenizer=minilm_tokenizer,
                    model=minilm_model,
                    device=device,
                )
                h5_file[group_name].create_dataset("minilm_lang_embedding", data=lang_embedding)


def main():
    parser = argparse.ArgumentParser(
        description="Generate DINO embeddings and frame-diff progress targets from LeRobot datasets."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--camera-key", default="image")
    args = parser.parse_args()

    build_lerobot_h5(
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        max_length=args.max_length,
        max_episodes=args.max_episodes,
        camera_key=args.camera_key,
    )


if __name__ == "__main__":
    main()
