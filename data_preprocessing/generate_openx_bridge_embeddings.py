import argparse
import re
from collections import defaultdict

import h5py
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from utils.processing_utils import dino_load_image, mean_pooling
from utils.progress_utils import compute_frame_diff_progress


DINO_BATCH_SIZE = 32


def center_crop(image, size=224):
    h, w = image.shape[:2]
    x = max((w - size) // 2, 0)
    y = max((h - size) // 2, 0)
    return image[y : y + size, x : x + size]


def sample_frames(frames, max_length):
    if len(frames) == 0:
        return []
    indices = np.linspace(0, len(frames) - 1, max_length, dtype=int)
    return [frames[i] for i in indices]


def sanitize_h5_key(text):
    text = text.replace("\x00", "").replace("/", " or ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or "empty_instruction"


def decode_instruction(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if hasattr(value, "numpy"):
        return decode_instruction(value.numpy())
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_instruction(value.item())
        if value.size == 0:
            return ""
        return decode_instruction(value.reshape(-1)[0])
    return str(value)


def encode_text(text, tokenizer, model, device):
    encoded_input = tokenizer(
        [text], padding=False, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        model_output = model(**encoded_input)
        return (
            mean_pooling(model_output, encoded_input["attention_mask"])
            .cpu()
            .detach()
            .numpy()
            .astype(np.float32)
        )


def get_dino_embeddings(frames, dino_model, device):
    with torch.inference_mode():
        episode_images_dino = [dino_load_image(img) for img in frames]
        episode_images_dino = [
            torch.concatenate(episode_images_dino[i : i + DINO_BATCH_SIZE])
            for i in range(0, len(episode_images_dino), DINO_BATCH_SIZE)
        ]

        embedding_list = []
        for batch in episode_images_dino:
            episode_image_embeddings = (
                dino_model(batch.to(device)).detach().cpu().numpy()
            )
            if episode_image_embeddings.ndim == 1:
                episode_image_embeddings = np.expand_dims(episode_image_embeddings, 0)
            embedding_list.append(episode_image_embeddings)
        return np.concatenate(embedding_list, axis=0).astype(np.float32)


def extract_episode_instruction(steps):
    for step in steps:
        if "language_instruction" in step:
            instruction = decode_instruction(step["language_instruction"])
            if instruction:
                return instruction
    return ""


def build_bridge_h5(
    dataset_name,
    split,
    data_dir,
    builder_dir,
    output_path,
    max_length,
    max_episodes,
    camera_key,
):
    import tensorflow as tf
    import tensorflow_datasets as tfds

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

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

    read_config = tfds.ReadConfig(
        interleave_cycle_length=1,
        shuffle_seed=0,
    )
    if builder_dir:
        builder = tfds.builder_from_directory(builder_dir)
        ds = builder.as_dataset(split=split, read_config=read_config)
    else:
        ds = tfds.load(
            dataset_name,
            split=split,
            data_dir=data_dir,
            read_config=read_config,
        )
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    options.experimental_deterministic = True
    ds = ds.with_options(options)
    if max_episodes > 0:
        ds = ds.take(max_episodes)

    per_group_counts = defaultdict(int)

    with h5py.File(output_path, "w") as h5_file:
        for episode in tqdm(ds, desc=f"Processing {dataset_name}:{split}"):
            steps = list(episode["steps"])
            if not steps:
                continue

            frames = [
                np.asarray(step["observation"][camera_key].numpy(), dtype=np.uint8)
                for step in steps
            ]
            if not frames:
                continue

            instruction = extract_episode_instruction(steps)
            if not instruction:
                instruction = f"{dataset_name}_{split}_episode"
            group_name = sanitize_h5_key(instruction)
            if group_name not in h5_file:
                h5_file.create_group(group_name)

            sampled_frames = sample_frames(frames, max_length=max_length)
            sampled_frames = [center_crop(frame, 224) for frame in sampled_frames]

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
        description="Generate Bridge OpenX DINO embeddings and frame-diff progress targets."
    )
    parser.add_argument("--dataset-name", type=str, default="bridge")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--builder-dir", type=str, default=None)
    parser.add_argument(
        "--output-path", type=str, default="datasets/bridge_embeddings_train.h5"
    )
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--camera-key", type=str, default="image_0")
    args = parser.parse_args()

    build_bridge_h5(
        dataset_name=args.dataset_name,
        split=args.split,
        data_dir=args.data_dir,
        builder_dir=args.builder_dir,
        output_path=args.output_path,
        max_length=args.max_length,
        max_episodes=args.max_episodes,
        camera_key=args.camera_key,
    )


if __name__ == "__main__":
    main()
