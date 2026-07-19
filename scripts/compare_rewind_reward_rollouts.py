#!/usr/bin/env python3
"""Roll out fixed policies and compare two ReWiND checkpoints per step.

Each trajectory is generated once, then every trajectory prefix is scored by
both the matched-linear ReWiND checkpoint and the pixel-difference checkpoint.
The plotted scores are raw progress predictions only: simulator rewards,
success bonuses, base rewards, and progress differences are not added.
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_LINEAR_REWARD = (
    "/scratch1/haobaizh/rewind_incremental/checkpoints_linear_matched/"
    "openx_linear_matched_20ep/rewind_metaworld_epoch_19.pth"
)
DEFAULT_PIXEL_REWARD = (
    "/scratch1/haobaizh/rewind_incremental/checkpoints_flow/"
    "openx_flow_20ep/rewind_metaworld_epoch_19.pth"
)
DEFAULT_LINEAR_POLICY_ROOT = (
    "/scratch1/haobaizh/rewind_incremental/no_action_online/"
    "openx_linear_matched_epoch19"
)
DEFAULT_PIXEL_POLICY_ROOT = (
    "/scratch1/haobaizh/rewind_incremental/no_action_online/openx_flow_epoch19"
)

POLICY_EXPERIMENTS = {
    "linear": "rewind_openx_linear_matched_epoch19_bonus200_no_base_10critics",
    "pixel": "rewind_openx_flow_epoch19_bonus200_no_base_10critics",
}


def parse_list(value: str) -> list[str]:
    normalized = value.replace(":", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def normalize_policy_arm(value: str) -> str:
    value = value.strip().lower().replace("-", "_")
    aliases = {
        "linear": "linear",
        "rewind": "linear",
        "pixel": "pixel",
        "pixel_difference": "pixel",
        "flow": "pixel",
        "optical_flow": "pixel",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported policy arm: {value}")
    return aliases[value]


def policy_checkpoint_path(
    arm: str,
    env_id: str,
    policy_seed: int,
    linear_root: Path,
    pixel_root: Path,
) -> Path:
    root = linear_root if arm == "linear" else pixel_root
    experiment = POLICY_EXPERIMENTS[arm]
    filename = f"{env_id}_online_{experiment}_seed{policy_seed}.zip"
    return root / env_id / filename


def expected_policy_paths(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    paths = []
    for env_id in args.envs:
        for arm in args.policy_arms:
            paths.append(
                (
                    env_id,
                    arm,
                    policy_checkpoint_path(
                        arm,
                        env_id,
                        args.policy_seed,
                        args.linear_policy_root,
                        args.pixel_policy_root,
                    ),
                )
            )
    return paths


def require_files(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    required = [
        ("linear reward checkpoint", args.linear_reward_checkpoint),
        ("pixel-difference reward checkpoint", args.pixel_reward_checkpoint),
    ]
    policy_paths = expected_policy_paths(args)
    required.extend(
        (f"{env_id} {arm} policy checkpoint", path)
        for env_id, arm, path in policy_paths
    )
    missing = [(label, path) for label, path in required if not path.is_file()]
    if missing:
        details = "\n".join(f"  {label}: {path}" for label, path in missing)
        raise FileNotFoundError(f"Missing required files:\n{details}")
    return policy_paths


def import_no_action_modules(no_action_repo: Path) -> SimpleNamespace:
    if not no_action_repo.is_dir():
        raise FileNotFoundError(f"No-action repository not found: {no_action_repo}")
    sys.path.insert(0, str(no_action_repo))

    import wandb
    from gym.wrappers.time_limit import TimeLimit
    from stable_baselines3.common.save_util import load_from_zip_file
    from stable_baselines3.common.vec_env import DummyVecEnv

    from envs.metaworld_envs.metaworld import (
        create_wrapped_env,
        environment_to_instruction,
    )
    from models.encoders.dino_miniLM_encoder import Dino_miniLM_Encoder
    from offline_rl_algorithms.rlpd import RLPD

    return SimpleNamespace(
        wandb=wandb,
        TimeLimit=TimeLimit,
        load_from_zip_file=load_from_zip_file,
        DummyVecEnv=DummyVecEnv,
        create_wrapped_env=create_wrapped_env,
        environment_to_instruction=environment_to_instruction,
        Dino_miniLM_Encoder=Dino_miniLM_Encoder,
        RLPD=RLPD,
    )


class DenseEvalRewardStub:
    """Minimal reward object for the no-action visual observation wrapper."""

    name = "dense"
    reward_at_every_step = False
    reward_divisor = 1.0
    success_bonus = 0.0

    def __init__(self, device: torch.device):
        self.device = device


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_to_metaworld_base(env):
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


def set_fixed_env_seed(vec_env, seed: int, time_limit_cls) -> None:
    for wrapped_env in vec_env.envs:
        base_env = unwrap_to_metaworld_base(wrapped_env)
        if not hasattr(base_env, "all_env_types"):
            raise RuntimeError("Could not locate MetaworldBase in wrapper stack")
        try:
            base_env.base_env.close()
        except Exception:
            pass
        base_env.rank = seed
        base_env.random_reset = "fixed"
        base_env.base_env = base_env.all_env_types[base_env.env_id](seed=seed)
        base_env.base_env = time_limit_cls(
            base_env.base_env,
            max_episode_steps=base_env.max_episode_steps,
        )


def normalize_saved_policy_kwargs(data: dict) -> None:
    policy_kwargs = data.get("policy_kwargs")
    if not isinstance(policy_kwargs, dict):
        return
    policy_kwargs.pop("device", None)
    saved_net_arch = policy_kwargs.get("net_arch")
    if (
        isinstance(saved_net_arch, list)
        and saved_net_arch
        and isinstance(saved_net_arch[0], dict)
    ):
        policy_kwargs["net_arch"] = saved_net_arch[0]


def load_rlpd_for_eval(modules: SimpleNamespace, path: Path, env, device: str):
    bootstrap_model = modules.RLPD(
        policy="MlpPolicy",
        env=env,
        offline_algo=None,
        device=device,
        _init_setup_model=False,
    )
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }
    data, params, pytorch_variables = modules.load_from_zip_file(
        str(path),
        device=device,
        custom_objects=custom_objects,
    )
    normalize_saved_policy_kwargs(data)

    init_signature = inspect.signature(bootstrap_model.__class__.__init__)
    excluded = {"self", "policy", "env", "offline_algo", "device", "_init_setup_model"}
    init_kwargs = {
        name: data[name]
        for name in init_signature.parameters
        if name in data and name not in excluded
    }
    model = bootstrap_model.__class__(
        policy=data["policy_class"],
        env=env,
        offline_algo=None,
        device=device,
        **init_kwargs,
    )
    model.__dict__.update(data)
    filtered_params = {
        name: state for name, state in params.items() if "optimizer" not in name
    }
    model.set_parameters(filtered_params, exact_match=False, device=device)

    if pytorch_variables is not None:
        for name, value in pytorch_variables.items():
            if value is None:
                continue
            parts = name.split(".")
            attr = model
            for part in parts[:-1]:
                attr = getattr(attr, part)
            getattr(attr, parts[-1]).data = value.data

    model.policy.set_training_mode(False)
    model.env = env
    model.n_envs = env.num_envs
    return model


def load_reward_checkpoint(
    path: Path,
    expected_target: str,
    device: torch.device,
):
    from model import ReWiNDTransformer

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_args = checkpoint.get("args")
    if model_args is None:
        raise KeyError(f"Checkpoint has no args: {path}")
    actual_target = getattr(model_args, "progress_target_type", "linear")
    if actual_target != expected_target:
        raise ValueError(
            f"Checkpoint target mismatch for {path}: "
            f"expected={expected_target} actual={actual_target}"
        )
    if not bool(getattr(model_args, "subsample_video", True)):
        raise ValueError(f"Checkpoint does not use the online prefix sampler: {path}")

    model = ReWiNDTransformer(
        args=model_args,
        video_dim=768,
        text_dim=384,
        hidden_dim=512,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, model_args


def prefix_batch(
    embeddings: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """Build online-style prefixes, including reset frame and current frame."""

    prefixes = []
    # Step 1 observes [reset frame, first post-action frame].
    for current_idx in range(1, embeddings.shape[0]):
        prefix = embeddings[: current_idx + 1]
        if prefix.shape[0] > max_length:
            indices = np.linspace(0, prefix.shape[0] - 1, max_length).astype(int)
            prefix = prefix[torch.as_tensor(indices, device=prefix.device)]
        elif prefix.shape[0] < max_length:
            padding = prefix[-1:].repeat(max_length - prefix.shape[0], 1)
            prefix = torch.cat([prefix, padding], dim=0)
        prefixes.append(prefix)
    if not prefixes:
        raise ValueError("A rollout must contain at least one environment step")
    return torch.stack(prefixes, dim=0)


def score_prefixes(
    model,
    model_args,
    embeddings: np.ndarray,
    text_embedding: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    video = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    if bool(getattr(model_args, "normalize_embedding", False)):
        video = F.normalize(video, p=2, dim=-1)
    prefixes = prefix_batch(video, int(model_args.max_length))
    text = torch.as_tensor(text_embedding, dtype=torch.float32, device=device).reshape(1, -1)

    scores = []
    with torch.inference_mode():
        for start in range(0, prefixes.shape[0], batch_size):
            video_batch = prefixes[start : start + batch_size]
            text_batch = text.repeat(video_batch.shape[0], 1)
            predictions = model(video_batch, text_batch)
            scores.append(predictions[:, -1, 0].detach().cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


def uint8_frame(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)[..., :3]
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating) and frame.max(initial=0) <= 1.0:
            frame = frame * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def create_rollout_env(
    modules: SimpleNamespace,
    env_id: str,
    encoder,
    instruction_embedding: np.ndarray,
    device: torch.device,
):
    reward_stub = DenseEvalRewardStub(device)
    factory = modules.create_wrapped_env(
        env_id,
        reward_model=reward_stub,
        image_encoder=encoder,
        language_features_policy=instruction_embedding,
        language_features_reward=instruction_embedding,
        monitor=False,
        goal_observable=True,
        success_bonus=0.0,
        is_state_based=False,
        mode="eval",
        use_proprio=True,
        dense_rewards_at_end=False,
        normalize_reward=False,
        terminate_on_success=True,
        use_progress_diff=False,
        use_base_reward=False,
    )
    return modules.DummyVecEnv([factory])


def rollout_policy(
    modules: SimpleNamespace,
    policy_path: Path,
    env_id: str,
    encoder,
    instruction_embedding: np.ndarray,
    device: torch.device,
    rollout_seed: int,
    max_steps: int,
):
    vec_env = create_rollout_env(
        modules,
        env_id,
        encoder,
        instruction_embedding,
        device,
    )
    set_fixed_env_seed(vec_env, rollout_seed, modules.TimeLimit)
    model = load_rlpd_for_eval(modules, policy_path, vec_env, str(device))
    env = vec_env.envs[0]

    obs = env.reset()
    frames = [uint8_frame(env.render(mode="rgb_array"))]
    simulator_rewards = []
    success_flags = []
    success_step = None

    try:
        for step in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action).reshape(env.action_space.shape)
            obs, simulator_reward, done, info = env.step(action)
            frames.append(uint8_frame(env.render(mode="rgb_array")))
            success = bool(info.get("success", info.get("is_success", False)))
            simulator_rewards.append(float(simulator_reward))
            success_flags.append(success)
            if success and success_step is None:
                success_step = step
            if bool(done):
                break
    finally:
        vec_env.close()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "frames": np.stack(frames, axis=0),
        "simulator_rewards": np.asarray(simulator_rewards, dtype=np.float32),
        "success_flags": np.asarray(success_flags, dtype=bool),
        "success_step": success_step,
    }


def encode_rollout_frames(
    encoder,
    frames: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    # The legacy encoder squeezes singleton DINO batches. A 129-frame rollout
    # therefore mixes [64, 768] and [768] arrays. Batch explicitly so the final
    # singleton remains [1, 768] without changing preprocessing or weights.
    transformed = [encoder.dino_load_image(frame) for frame in frames]
    embedding_batches = []
    with torch.inference_mode():
        for start in range(0, len(transformed), batch_size):
            image_batch = torch.cat(transformed[start : start + batch_size], dim=0)
            output = encoder.dinov2_vits14(image_batch.to(encoder.device))
            if output.ndim == 1:
                output = output.unsqueeze(0)
            embedding_batches.append(output.detach().cpu().numpy())
    embeddings = np.concatenate(embedding_batches, axis=0)
    if embeddings.shape != (len(frames), 768):
        raise ValueError(
            f"Unexpected DINO embedding shape {embeddings.shape}; "
            f"expected {(len(frames), 768)}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("DINO embeddings contain NaN or Inf")
    return embeddings.astype(np.float32)


def save_raw_video(path: Path, frames: np.ndarray, fps: int) -> None:
    import imageio.v2 as imageio

    with imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def save_scores_csv(
    path: Path,
    linear_scores: np.ndarray,
    pixel_scores: np.ndarray,
    simulator_rewards: np.ndarray,
    success_flags: np.ndarray,
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "step",
                "linear_progress_score",
                "pixel_difference_progress_score",
                "pixel_minus_linear",
                "linear_progress_delta",
                "pixel_difference_progress_delta",
                "simulator_reward",
                "success",
            ]
        )
        for index in range(len(linear_scores)):
            linear_delta = (
                linear_scores[index] - linear_scores[index - 1]
                if index > 0
                else ""
            )
            pixel_delta = (
                pixel_scores[index] - pixel_scores[index - 1]
                if index > 0
                else ""
            )
            writer.writerow(
                [
                    index + 1,
                    float(linear_scores[index]),
                    float(pixel_scores[index]),
                    float(pixel_scores[index] - linear_scores[index]),
                    linear_delta,
                    pixel_delta,
                    float(simulator_rewards[index]),
                    bool(success_flags[index]),
                ]
            )


def configure_curve_axis(
    ax,
    steps: np.ndarray,
    linear_scores: np.ndarray,
    pixel_scores: np.ndarray,
    success_step: int | None,
    upto: int | None = None,
) -> None:
    linear_color = "#e63946"
    pixel_color = "#277da1"
    ax.plot(
        steps,
        linear_scores,
        color=linear_color,
        linewidth=1.3,
        alpha=0.2 if upto is not None else 1.0,
        label="Linear-progress ReWiND",
    )
    ax.plot(
        steps,
        pixel_scores,
        color=pixel_color,
        linewidth=1.3,
        alpha=0.2 if upto is not None else 1.0,
        label="Pixel-difference objective",
    )
    if upto is not None:
        ax.plot(steps[:upto], linear_scores[:upto], color=linear_color, linewidth=2.2)
        ax.plot(steps[:upto], pixel_scores[:upto], color=pixel_color, linewidth=2.2)
        ax.scatter(steps[upto - 1], linear_scores[upto - 1], color=linear_color, s=34)
        ax.scatter(steps[upto - 1], pixel_scores[upto - 1], color=pixel_color, s=34)
        ax.axvline(steps[upto - 1], color="black", linewidth=0.8, alpha=0.3)
    if success_step is not None:
        ax.axvline(
            success_step,
            color="#2a9d8f",
            linestyle="--",
            linewidth=1.4,
            label="Success step",
        )
    ax.set_xlim(1, max(int(steps[-1]), 2))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Environment step")
    ax.set_ylabel("Raw progress score P(prefix)")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=9)


def save_curve_plot(
    path: Path,
    linear_scores: np.ndarray,
    pixel_scores: np.ndarray,
    success_step: int | None,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(1, len(linear_scores) + 1)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    configure_curve_axis(ax, steps, linear_scores, pixel_scores, success_step)
    ax.set_title(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_combined_video(
    path: Path,
    post_action_frames: np.ndarray,
    linear_scores: np.ndarray,
    pixel_scores: np.ndarray,
    success_step: int | None,
    title: str,
    fps: int,
) -> None:
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(1, len(linear_scores) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), dpi=100)
    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    )
    try:
        for index, frame in enumerate(post_action_frames):
            for ax in axes:
                ax.clear()
            step = index + 1
            axes[0].imshow(frame)
            axes[0].set_title(
                f"{title}\nstep {step}/{len(post_action_frames)} | "
                f"linear={linear_scores[index]:.3f} | "
                f"pixel={pixel_scores[index]:.3f}"
            )
            axes[0].axis("off")
            configure_curve_axis(
                axes[1],
                steps,
                linear_scores,
                pixel_scores,
                success_step,
                upto=step,
            )
            axes[1].set_title("Same rollout, two reward-model objectives")
            fig.tight_layout()
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(rgba[..., :3])
    finally:
        writer.close()
        plt.close(fig)


def save_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare linear and pixel-difference ReWiND scores on fixed rollouts."
    )
    parser.add_argument(
        "--no-action-repo",
        type=Path,
        default=Path("/project2/biyik_1165/haobaizh/rewind_no-action-chunk"),
    )
    parser.add_argument(
        "--linear-reward-checkpoint", type=Path, default=Path(DEFAULT_LINEAR_REWARD)
    )
    parser.add_argument(
        "--pixel-reward-checkpoint", type=Path, default=Path(DEFAULT_PIXEL_REWARD)
    )
    parser.add_argument(
        "--linear-policy-root", type=Path, default=Path(DEFAULT_LINEAR_POLICY_ROOT)
    )
    parser.add_argument(
        "--pixel-policy-root", type=Path, default=Path(DEFAULT_PIXEL_POLICY_ROOT)
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/scratch1/haobaizh/rewind_incremental/reward_rollout_comparison"
        ),
    )
    parser.add_argument(
        "--envs",
        type=parse_list,
        default=parse_list("window-close-v2,reach-wall-v2,faucet-close-v2"),
    )
    parser.add_argument(
        "--policy-arms",
        type=lambda value: [normalize_policy_arm(item) for item in parse_list(value)],
        default=["linear", "pixel"],
    )
    parser.add_argument("--policy-seed", type=int, default=42)
    parser.add_argument("--rollout-seed", type=int, default=450)
    parser.add_argument("--episodes-per-policy", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.policy_arms = list(dict.fromkeys(args.policy_arms))
    if not args.envs:
        raise ValueError("At least one environment is required")
    if args.episodes_per_policy <= 0:
        raise ValueError("episodes-per-policy must be positive")
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")

    policy_paths = require_files(args)
    print("PREFLIGHT_OK")
    print(f"linear_reward={args.linear_reward_checkpoint}")
    print(f"pixel_reward={args.pixel_reward_checkpoint}")
    for env_id, arm, path in policy_paths:
        print(f"policy env={env_id} arm={arm} path={path}")
    if args.preflight_only:
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    set_global_seed(args.rollout_seed)
    modules = import_no_action_modules(args.no_action_repo)
    modules.wandb.init(project="rewind-reward-rollout-analysis", mode="disabled")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encoder = modules.Dino_miniLM_Encoder(
        use_pca=False,
        device=str(device),
        dino_batch_size=64,
        max_num_frames_per_episode=args.max_steps + 1,
        batch_size=64,
    )
    linear_model, linear_args = load_reward_checkpoint(
        args.linear_reward_checkpoint, "linear", device
    )
    pixel_model, pixel_args = load_reward_checkpoint(
        args.pixel_reward_checkpoint, "optical_flow", device
    )
    print(
        "REWARD_MODELS_OK "
        f"linear_max_length={linear_args.max_length} "
        f"pixel_max_length={pixel_args.max_length}"
    )

    summary_rows = []
    try:
        for env_id in args.envs:
            if env_id not in modules.environment_to_instruction:
                raise KeyError(f"No instruction mapping for environment: {env_id}")
            instruction = modules.environment_to_instruction[env_id]
            text_embedding = np.asarray(encoder.encode_text(instruction), dtype=np.float32).reshape(-1)
            if text_embedding.shape != (384,):
                raise ValueError(
                    f"Unexpected text embedding shape for {env_id}: {text_embedding.shape}"
                )

            for arm in args.policy_arms:
                policy_path = policy_checkpoint_path(
                    arm,
                    env_id,
                    args.policy_seed,
                    args.linear_policy_root,
                    args.pixel_policy_root,
                )
                for episode in range(args.episodes_per_policy):
                    env_seed = args.rollout_seed + episode
                    run_name = (
                        f"policy_{arm}_policyseed{args.policy_seed}_"
                        f"envseed{env_seed}_episode{episode}"
                    )
                    run_dir = args.output_dir / env_id / run_name
                    run_dir.mkdir(parents=True, exist_ok=True)
                    print(
                        f"ROLLOUT_START env={env_id} policy_arm={arm} "
                        f"episode={episode} env_seed={env_seed}"
                    )
                    rollout = rollout_policy(
                        modules,
                        policy_path,
                        env_id,
                        encoder,
                        text_embedding,
                        device,
                        env_seed,
                        args.max_steps,
                    )
                    frames = rollout["frames"]
                    embeddings = encode_rollout_frames(encoder, frames)
                    linear_scores = score_prefixes(
                        linear_model,
                        linear_args,
                        embeddings,
                        text_embedding,
                        device,
                        args.score_batch_size,
                    )
                    pixel_scores = score_prefixes(
                        pixel_model,
                        pixel_args,
                        embeddings,
                        text_embedding,
                        device,
                        args.score_batch_size,
                    )
                    num_steps = len(frames) - 1
                    if not (
                        len(linear_scores)
                        == len(pixel_scores)
                        == len(rollout["simulator_rewards"])
                        == num_steps
                    ):
                        raise RuntimeError("Rollout and reward-curve lengths do not match")

                    title = (
                        f"{env_id} | policy={arm} | env seed={env_seed} | "
                        f"success={rollout['success_step'] is not None}"
                    )
                    raw_video_path = run_dir / "rollout.mp4"
                    curve_path = run_dir / "reward_curve.png"
                    combined_path = run_dir / "rollout_with_reward_curve.mp4"
                    csv_path = run_dir / "scores.csv"
                    metadata_path = run_dir / "metadata.json"

                    save_raw_video(raw_video_path, frames, args.fps)
                    save_curve_plot(
                        curve_path,
                        linear_scores,
                        pixel_scores,
                        rollout["success_step"],
                        title,
                    )
                    save_combined_video(
                        combined_path,
                        frames[1:],
                        linear_scores,
                        pixel_scores,
                        rollout["success_step"],
                        title,
                        args.fps,
                    )
                    save_scores_csv(
                        csv_path,
                        linear_scores,
                        pixel_scores,
                        rollout["simulator_rewards"],
                        rollout["success_flags"],
                    )

                    metadata = {
                        "environment": env_id,
                        "instruction": instruction,
                        "policy_arm": arm,
                        "policy_checkpoint": str(policy_path),
                        "policy_seed": args.policy_seed,
                        "rollout_seed": env_seed,
                        "episode": episode,
                        "num_steps": num_steps,
                        "success": rollout["success_step"] is not None,
                        "success_step": rollout["success_step"],
                        "linear_reward_checkpoint": str(args.linear_reward_checkpoint),
                        "pixel_reward_checkpoint": str(args.pixel_reward_checkpoint),
                        "score_definition": "raw P(prefix), no success bonus/base reward/difference",
                        "linear_final_score": float(linear_scores[-1]),
                        "pixel_final_score": float(pixel_scores[-1]),
                        "linear_max_score": float(linear_scores.max()),
                        "pixel_max_score": float(pixel_scores.max()),
                    }
                    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
                    summary_rows.append(metadata)
                    print(
                        f"ROLLOUT_OK env={env_id} policy_arm={arm} "
                        f"steps={num_steps} success_step={rollout['success_step']} "
                        f"output={run_dir}"
                    )

        summary_path = args.output_dir / "summary.csv"
        save_summary_csv(summary_path, summary_rows)
        print(
            f"COMPARISON_OK rollouts={len(summary_rows)} "
            f"summary={summary_path} output_dir={args.output_dir}"
        )
    finally:
        modules.wandb.finish()


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
