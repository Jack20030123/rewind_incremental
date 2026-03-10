import inspect
import random

import numpy as np

import hydra
import torch as th
import wandb
from gym.wrappers.time_limit import TimeLimit
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.save_util import load_from_zip_file

from offline_rl_algorithms.rlpd import RLPD
from train_policy import create_envs, parse_reward_model


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def _unwrap_to_metaworld_base(env):
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


def set_eval_env_seed(env, seed: int) -> None:
    for sub_env in env.envs:
        base_env = _unwrap_to_metaworld_base(sub_env)
        if not hasattr(base_env, "all_env_types"):
            continue

        # Freeze reset randomness so reset() reuses the exact requested seed.
        base_env.rank = seed
        base_env.random_reset = "fixed"
        base_env.base_env = base_env.all_env_types[base_env.env_id](seed=seed)
        base_env.base_env = TimeLimit(
            base_env.base_env,
            max_episode_steps=base_env.max_episode_steps,
        )


def _normalize_saved_policy_kwargs(data: dict) -> None:
    if "policy_kwargs" not in data:
        return

    if "device" in data["policy_kwargs"]:
        del data["policy_kwargs"]["device"]

    if "net_arch" not in data["policy_kwargs"]:
        return

    saved_net_arch = data["policy_kwargs"]["net_arch"]
    if (
        isinstance(saved_net_arch, list)
        and len(saved_net_arch) > 0
        and isinstance(saved_net_arch[0], dict)
    ):
        data["policy_kwargs"]["net_arch"] = saved_net_arch[0]


def load_rlpd_for_eval(
    bootstrap_model: RLPD,
    path: str,
    env,
    device: str = "auto",
) -> RLPD:
    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
    }
    data, params, pytorch_variables = load_from_zip_file(
        path,
        device=device,
        custom_objects=custom_objects,
    )

    _normalize_saved_policy_kwargs(data)

    init_signature = inspect.signature(bootstrap_model.__class__.__init__)
    init_kwargs = {
        name: data[name]
        for name in init_signature.parameters
        if name in data
        and name
        not in {"self", "policy", "env", "offline_algo", "device", "_init_setup_model"}
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
        name: state
        for name, state in params.items()
        if "optimizer" not in name
    }
    model.set_parameters(filtered_params, exact_match=False, device=device)

    if pytorch_variables is not None:
        for name, value in pytorch_variables.items():
            if value is None:
                continue
            recursive_name = name.split(".")
            attr = model
            for part in recursive_name[:-1]:
                attr = getattr(attr, part)
            getattr(attr, recursive_name[-1]).data = value.data

    model.policy.set_training_mode(False)
    model.env = env
    model.n_envs = env.num_envs
    return model


@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def main(cfg: DictConfig) -> None:
    print("====== EVAL POLICY DEBUG ======")
    print(OmegaConf.to_yaml(cfg))

    cfg.environment.n_envs = 1
    eval_seed = int(cfg.general_training.seed)
    set_global_seed(eval_seed)

    ckpt_path = to_absolute_path(cfg.general_training.ckpt_path)
    wandb.init(
        project=cfg.logging.wandb_project_name,
        entity=cfg.logging.wandb_entity_name,
        mode="disabled",
    )
    reward_model = parse_reward_model(cfg.reward_model)
    _, eval_env = create_envs(cfg, reward_model)
    set_eval_env_seed(eval_env, eval_seed)

    bootstrap_model = RLPD(
        policy=cfg.model.policy_type,
        env=eval_env,
        offline_algo=None,
        device="auto",
        _init_setup_model=False,
    )
    model = load_rlpd_for_eval(bootstrap_model, ckpt_path, eval_env)

    obs = eval_env.reset()
    done = False
    timestep = 0
    episode_start = np.array([True], dtype=bool)

    print(f"checkpoint: {ckpt_path}")
    print(f"eval_seed: {eval_seed}")
    print("timestep, reward, done, success")
    while not done:
        action, _ = model.predict(
            obs,
            deterministic=True,
            episode_start=episode_start,
        )
        obs, reward, done_array, info = eval_env.step(action)

        reward_value = float(reward[0])
        done = bool(done_array[0])
        success = bool(info[0].get("success", info[0].get("is_success", False)))

        print(f"{timestep}, {reward_value:.4f}, {done}, {success}")
        timestep += 1
        episode_start = done_array.astype(bool)


if __name__ == "__main__":
    th.set_grad_enabled(False)
    try:
        main()
    finally:
        if wandb.run is not None:
            wandb.finish()
