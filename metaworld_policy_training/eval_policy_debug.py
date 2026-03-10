import os
import hydra
import torch as th
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

from train_policy import parse_reward_model, create_envs
from stable_baselines3 import SAC
from offline_rl_algorithms.iql import IQL
from offline_rl_algorithms.rlpd import RLPD


@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def main(cfg: DictConfig):

    print("====== EVAL ONLY DEBUG MODE ======")
    print(OmegaConf.to_yaml(cfg))

    # ---- Force single env ----
    cfg.environment.n_envs = 1

    # ---- Load reward model ----
    reward_model = parse_reward_model(cfg.reward_model)

    # ---- Create environment (same as training) ----
    envs, _ = create_envs(cfg, reward_model)

    # ---- Determine algorithm ----
    algo = cfg.general_training.algo.lower()

    ckpt_path = str(cfg.general_training.ckpt_path)
    print("Loading checkpoint:", ckpt_path)

    if algo == "sac":
        model = SAC.load(ckpt_path, env=envs)
    elif algo == "iql":
        model = IQL.load(
            ckpt_path,
            env=envs,
            custom_objects={
                "observation_space": envs.observation_space,
                "action_space": envs.action_space,
            },
        )
    elif algo == "rlpd":
        # Use same loading pattern as train_policy
        model = RLPD.load(
            ckpt_path,
            env=envs,
            custom_objects={
                "observation_space": envs.observation_space,
                "action_space": envs.action_space,
            },
            print_system_info=True,
            load_torch_params_only=True,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    print("Model loaded successfully.")

    # ---- Run one episode ----
    obs = envs.reset()
    done = False
    t = 0

    print("\nStarting episode...\n")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = envs.step(action)

        success_flag = info[0].get("success", False)

        print(
            f"t={t:03d} | reward={reward[0]:.4f} | done={done[0]} | success={success_flag}"
        )

        t += 1

    print("\nEpisode finished.")


if __name__ == "__main__":
    main()