#!/usr/bin/env python3
"""Stack matched Linear and Pixel-Difference policy videos for presentation."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


DEFAULT_ENVS = "window-close-v2,reach-wall-v2,faucet-close-v2"
LINEAR_COLOR = (230, 57, 70)
PIXEL_COLOR = (39, 125, 161)
HEADER_BACKGROUND = (22, 25, 29)
SEPARATOR_COLOR = (242, 244, 246)
HEADER_HEIGHT = 48
SEPARATOR_HEIGHT = 16


@dataclass(frozen=True)
class PolicyVideo:
    arm: str
    run_dir: Path
    video_path: Path
    metadata_path: Path
    metadata: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one vertically stacked Linear-vs-Pixel policy video per task "
            "from existing rollout_with_reward_curve.mp4 artifacts."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--envs", default=DEFAULT_ENVS)
    parser.add_argument("--policy-seed", type=int, default=42)
    parser.add_argument("--rollout-seed", type=int, default=450)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--input-name", default="rollout_with_reward_curve.mp4")
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace paired videos that already exist.",
    )
    return parser.parse_args()


def parse_envs(raw: str) -> list[str]:
    envs = [item.strip() for item in raw.split(",") if item.strip()]
    if not envs:
        raise ValueError("--envs must contain at least one environment")
    if len(envs) != len(set(envs)):
        raise ValueError(f"--envs contains duplicates: {raw}")
    return envs


def load_policy_video(
    input_root: Path,
    env_id: str,
    arm: str,
    policy_seed: int,
    rollout_seed: int,
    episode: int,
    input_name: str,
) -> PolicyVideo:
    run_name = (
        f"policy_{arm}_policyseed{policy_seed}_"
        f"envseed{rollout_seed}_episode{episode}"
    )
    run_dir = input_root / env_id / run_name
    video_path = run_dir / input_name
    metadata_path = run_dir / "metadata.json"

    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing input video: {video_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    expected = {
        "environment": env_id,
        "policy_arm": arm,
        "policy_seed": policy_seed,
        "rollout_seed": rollout_seed,
        "episode": episode,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"{metadata_path}: expected {key}={value!r}, "
                f"found {metadata.get(key)!r}"
            )

    return PolicyVideo(
        arm=arm,
        run_dir=run_dir,
        video_path=video_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        video_metadata = reader.get_meta_data()
        fps = float(video_metadata.get("fps", 20.0))
        frames = [normalize_frame(frame) for frame in reader]
    finally:
        reader.close()

    if not frames:
        raise RuntimeError(f"Input video contains no frames: {path}")
    return frames, fps


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Unsupported video frame shape: {array.shape}")
    array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def validate_frame_count(policy_video: PolicyVideo, frames: list[np.ndarray]) -> None:
    expected_steps = int(policy_video.metadata["num_steps"])
    if len(frames) != expected_steps:
        raise RuntimeError(
            f"{policy_video.video_path}: metadata reports {expected_steps} steps, "
            f"but the video has {len(frames)} frames"
        )


def resize_with_letterbox(
    frame: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    if width == target_width and height == target_height:
        return frame

    scale = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resampling = getattr(Image, "Resampling", Image)
    resized = Image.fromarray(frame).resize(
        (resized_width, resized_height),
        resampling.LANCZOS,
    )
    canvas = np.full((target_height, target_width, 3), 18, dtype=np.uint8)
    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = np.asarray(resized)
    return canvas


def font_candidates() -> list[Path]:
    prefix = Path(sys.prefix)
    return [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        prefix / "share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        prefix / "fonts/DejaVuSans.ttf",
    ]


def fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int):
    for size in range(22, 11, -1):
        for candidate in font_candidates():
            if not candidate.is_file():
                continue
            font = ImageFont.truetype(str(candidate), size=size)
            bounds = draw.textbbox((0, 0), text, font=font)
            if bounds[2] - bounds[0] <= max_width:
                return font
    return ImageFont.load_default()


def status_text(policy_video: PolicyVideo) -> str:
    metadata = policy_video.metadata
    success_step = metadata.get("success_step")
    if success_step is None:
        return f"FAILURE | {metadata['num_steps']} steps"
    return f"SUCCESS @ step {success_step} | {metadata['num_steps']} steps"


def make_header(
    width: int,
    policy_video: PolicyVideo,
    accent_color: tuple[int, int, int],
    held: bool,
) -> np.ndarray:
    arm_label = (
        "LINEAR FINAL POLICY"
        if policy_video.arm == "linear"
        else "PIXEL-DIFFERENCE FINAL POLICY"
    )
    metadata = policy_video.metadata
    text = (
        f"{arm_label} | {status_text(policy_video)} | "
        f"policy seed {metadata['policy_seed']} | env seed {metadata['rollout_seed']}"
    )
    if held:
        text += " | FINAL FRAME HELD"

    header = Image.new("RGB", (width, HEADER_HEIGHT), HEADER_BACKGROUND)
    draw = ImageDraw.Draw(header)
    draw.rectangle((0, 0, 8, HEADER_HEIGHT), fill=accent_color)
    font = fit_font(draw, text, max_width=width - 32)
    bounds = draw.textbbox((0, 0), text, font=font)
    text_height = bounds[3] - bounds[1]
    y_position = max(0, (HEADER_HEIGHT - text_height) // 2 - bounds[1])
    draw.text((20, y_position), text, fill=(248, 249, 250), font=font)
    return np.asarray(header)


def write_paired_video(
    output_path: Path,
    linear_video: PolicyVideo,
    pixel_video: PolicyVideo,
    quality: int,
    overwrite: bool,
) -> dict:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass --overwrite to replace it"
        )

    linear_frames, linear_fps = read_video(linear_video.video_path)
    pixel_frames, pixel_fps = read_video(pixel_video.video_path)
    validate_frame_count(linear_video, linear_frames)
    validate_frame_count(pixel_video, pixel_frames)
    if abs(linear_fps - pixel_fps) > 0.01:
        raise RuntimeError(
            f"Input FPS mismatch for {linear_video.metadata['environment']}: "
            f"linear={linear_fps}, pixel={pixel_fps}"
        )

    target_width = max(linear_frames[0].shape[1], pixel_frames[0].shape[1])
    target_height = max(linear_frames[0].shape[0], pixel_frames[0].shape[0])
    total_frames = max(len(linear_frames), len(pixel_frames))

    headers = {
        ("linear", False): make_header(
            target_width, linear_video, LINEAR_COLOR, held=False
        ),
        ("linear", True): make_header(
            target_width, linear_video, LINEAR_COLOR, held=True
        ),
        ("pixel", False): make_header(
            target_width, pixel_video, PIXEL_COLOR, held=False
        ),
        ("pixel", True): make_header(
            target_width, pixel_video, PIXEL_COLOR, held=True
        ),
    }
    separator = np.full(
        (SEPARATOR_HEIGHT, target_width, 3),
        SEPARATOR_COLOR,
        dtype=np.uint8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.stem}.tmp.mp4")
    temporary_path.unlink(missing_ok=True)
    writer = imageio.get_writer(
        str(temporary_path),
        fps=linear_fps,
        codec="libx264",
        quality=quality,
        macro_block_size=16,
    )
    try:
        for index in range(total_frames):
            linear_held = index >= len(linear_frames)
            pixel_held = index >= len(pixel_frames)
            linear_frame = linear_frames[min(index, len(linear_frames) - 1)]
            pixel_frame = pixel_frames[min(index, len(pixel_frames) - 1)]
            linear_frame = resize_with_letterbox(
                linear_frame, target_width, target_height
            )
            pixel_frame = resize_with_letterbox(
                pixel_frame, target_width, target_height
            )
            combined = np.concatenate(
                [
                    headers[("linear", linear_held)],
                    linear_frame,
                    separator,
                    headers[("pixel", pixel_held)],
                    pixel_frame,
                ],
                axis=0,
            )
            writer.append_data(combined)
    except Exception:
        writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()
        temporary_path.replace(output_path)

    result = {
        "environment": linear_video.metadata["environment"],
        "output_path": str(output_path),
        "fps": linear_fps,
        "output_frames": total_frames,
        "timeline_rule": (
            "Shared environment-step clock; the shorter rollout holds its final frame."
        ),
        "linear": {
            "source_video": str(linear_video.video_path),
            "source_metadata": str(linear_video.metadata_path),
            "frames": len(linear_frames),
            "success": bool(linear_video.metadata["success"]),
            "success_step": linear_video.metadata.get("success_step"),
        },
        "pixel": {
            "source_video": str(pixel_video.video_path),
            "source_metadata": str(pixel_video.metadata_path),
            "frames": len(pixel_frames),
            "success": bool(pixel_video.metadata["success"]),
            "success_step": pixel_video.metadata.get("success_step"),
        },
    }
    del linear_frames, pixel_frames
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    envs = parse_envs(args.envs)
    pairs = []
    for env_id in envs:
        linear_video = load_policy_video(
            args.input_root,
            env_id,
            "linear",
            args.policy_seed,
            args.rollout_seed,
            args.episode,
            args.input_name,
        )
        pixel_video = load_policy_video(
            args.input_root,
            env_id,
            "pixel",
            args.policy_seed,
            args.rollout_seed,
            args.episode,
            args.input_name,
        )
        pairs.append((env_id, linear_video, pixel_video))
        print(
            f"PAIR_PREFLIGHT_OK env={env_id} "
            f"linear={linear_video.video_path} pixel={pixel_video.video_path}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for env_id, linear_video, pixel_video in pairs:
        output_path = args.output_dir / f"{env_id}_linear_vs_pixel.mp4"
        result = write_paired_video(
            output_path,
            linear_video,
            pixel_video,
            args.quality,
            args.overwrite,
        )
        metadata_path = output_path.with_suffix(".json")
        metadata_path.write_text(json.dumps(result, indent=2) + "\n")
        manifest.append(result)
        print(
            f"PAIR_OK env={env_id} frames={result['output_frames']} "
            f"fps={result['fps']:.3f} output={output_path}",
            flush=True,
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"PAIR_COMPARISON_OK videos={len(manifest)} "
        f"manifest={manifest_path} output_dir={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
