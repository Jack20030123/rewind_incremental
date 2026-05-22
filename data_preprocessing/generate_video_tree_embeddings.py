import argparse
import pickle
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from data_preprocessing.generate_openx_bridge_embeddings import (
    center_crop,
    encode_text,
    get_dino_embeddings,
    sample_frames,
    sanitize_h5_key,
)
from utils.progress_utils import compute_frame_diff_progress


IMAGE_KEY_HINTS = ("image", "rgb", "frame")


def _task_instruction(dataset_name, video_path):
    match = re.search(r"task_(\d+)", video_path.name)
    if match:
        return f"{dataset_name} task {match.group(1)}"
    for part in reversed(video_path.parts):
        stem = Path(part).stem
        if stem and stem not in {".", ".."} and not stem.startswith(("cam_", "video")):
            text = re.sub(r"\.(tar|tgz)$", "", stem)
            text = text.replace("_", " ")
            return f"{dataset_name} {text}"
    return f"{dataset_name} video"


def _value_to_image(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.ndim == 3:
            return np.asarray(value[:, :, :3], dtype=np.uint8)
        if value.ndim == 4 and value.shape[0] > 0:
            return _value_to_image(value[0])
        if value.dtype == object and value.size == 1:
            return _value_to_image(value.item())
    if isinstance(value, (bytes, bytearray)):
        try:
            from PIL import Image
            import io

            return np.asarray(Image.open(io.BytesIO(value)).convert("RGB"), dtype=np.uint8)
        except Exception:
            return None
    return None


def _find_image_sequence(obj):
    if isinstance(obj, dict):
        if "image_list" in obj:
            found = _find_image_sequence(obj["image_list"])
            if found:
                return found
        for key, value in obj.items():
            if any(hint in str(key).lower() for hint in IMAGE_KEY_HINTS):
                image = _value_to_image(value)
                if image is not None:
                    return [image]
                if isinstance(value, np.ndarray) and value.ndim == 4:
                    return [
                        np.asarray(frame[:, :, :3], dtype=np.uint8)
                        for frame in value
                    ]
        for value in obj.values():
            found = _find_image_sequence(value)
            if found:
                return found
    elif isinstance(obj, (list, tuple)):
        images = []
        for value in obj:
            image = _value_to_image(value)
            if image is not None:
                images.append(image)
        if images:
            return images
        for value in obj:
            found = _find_image_sequence(value)
            if found:
                return found
    else:
        image = _value_to_image(obj)
        if image is not None:
            return [image]
    return []


def _load_pickle_frames(path):
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return _find_image_sequence(data)


def _read_sampled_video_frames(video_path, max_length):
    reader = imageio.get_reader(str(video_path), "ffmpeg")
    try:
        try:
            frame_count = int(reader.count_frames())
        except Exception:
            frame_count = 0

        if frame_count > 0:
            indices = np.linspace(0, frame_count - 1, max_length, dtype=int)
            frames = [reader.get_data(int(index)) for index in indices]
        else:
            frames = []
            for frame in reader:
                frames.append(frame)
            frames = sample_frames(frames, max_length)
    finally:
        reader.close()

    return [np.asarray(frame[:, :, :3], dtype=np.uint8) for frame in frames]


def _iter_videos(dataset_dir, camera_key):
    videos = sorted(Path(dataset_dir).glob("**/*.mp4"))
    if camera_key:
        videos = [path for path in videos if path.parent.name == camera_key]
    return videos


def _iter_pickle_files(dataset_dir):
    return sorted(Path(dataset_dir).glob("**/*.pickle")) + sorted(Path(dataset_dir).glob("**/*.pkl"))


def _extract_archives(dataset_dir, extract_dir):
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    archive_paths = sorted(Path(dataset_dir).glob("**/*.tar*"))
    for archive_path in tqdm(archive_paths, desc="Extracting archives"):
        target_dir = extract_dir / archive_path.relative_to(dataset_dir).with_suffix("")
        if target_dir.exists() and any(target_dir.iterdir()):
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path) as archive:
            archive.extractall(target_dir)
    return extract_dir


def build_video_tree_h5(
    dataset_name,
    dataset_dir,
    output_path,
    max_length,
    max_episodes,
    camera_key,
    extract_dir,
):
    source_dir = Path(dataset_dir)
    if extract_dir:
        source_dir = _extract_archives(source_dir, extract_dir)

    media_paths = _iter_videos(source_dir, camera_key)
    pickle_paths = [] if media_paths else _iter_pickle_files(source_dir)
    media_paths = media_paths or pickle_paths
    if not media_paths:
        hint = f" under camera directory {camera_key}" if camera_key else ""
        raise FileNotFoundError(f"No .mp4 or .pickle files found in {source_dir}{hint}")
    if max_episodes > 0:
        media_paths = media_paths[:max_episodes]

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

    per_group_counts = defaultdict(int)

    with h5py.File(output_path, "w") as h5_file:
        for media_path in tqdm(media_paths, desc=f"Processing {dataset_name} media"):
            if media_path.suffix == ".mp4":
                frames = _read_sampled_video_frames(media_path, max_length=max_length)
            else:
                frames = sample_frames(_load_pickle_frames(media_path), max_length)
            if not frames:
                continue

            sampled_frames = [center_crop(frame, 224) for frame in frames]
            instruction = _task_instruction(dataset_name, media_path)
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
            h5_file[group_name].create_dataset(
                f"flow_progress_{traj_id}", data=flow_progress
            )
            h5_file[group_name].create_dataset(
                f"flow_signal_{traj_id}", data=flow_signal
            )
            if "minilm_lang_embedding" not in h5_file[group_name]:
                lang_embedding = encode_text(
                    instruction,
                    tokenizer=minilm_tokenizer,
                    model=minilm_model,
                    device=device,
                )
                h5_file[group_name].create_dataset(
                    "minilm_lang_embedding", data=lang_embedding
                )


def main():
    parser = argparse.ArgumentParser(
        description="Generate DINO embeddings and frame-diff progress targets from a recursive video tree."
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument(
        "--camera-key",
        default="",
        help="Optional camera directory basename to keep, e.g. cam_035622060973.",
    )
    parser.add_argument(
        "--extract-dir",
        default="",
        help="Optional directory for extracting .tar/.tar.gz archives before processing.",
    )
    args = parser.parse_args()

    build_video_tree_h5(
        dataset_name=args.dataset_name,
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        max_length=args.max_length,
        max_episodes=args.max_episodes,
        camera_key=args.camera_key,
        extract_dir=args.extract_dir,
    )


if __name__ == "__main__":
    main()
