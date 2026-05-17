# ReWiND: Language-Guided Rewards Teach Robot Policies without New Demonstrations. (Oral Presentation @ CoRL 2025) 

![ReWiND Teaser](rewind_teaser.png)

<p align="center">
  <a href="https://arxiv.org/abs/2505.10911">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv2505.10911-b31b1b.svg">
  </a>
   <a href="https://rewind-reward.github.io/">
   <img alt="Website" src="https://img.shields.io/badge/Website-rewind--reward.github.io-blue">
   </a>
</p>

We provide code to train ReWiND reward models and policies on MetaWorld.
The overall pipeline is as follows:
- Train the ReWiND Reward Model on MetaWorld + OXE data.
- Label the offline training dataset with the ReWiND Reward Model.
- Train the ReWiND Policy with offline to online RL for new tasks.

## Installation Instructions:
```bash
git clone git@github.com:Jiahui-3205/ReWiND_Release.git
cd ReWiND_Release/
```


### Create Environment
```bash
# Run the setup script to create environment and install all dependencies
bash -i setup_ReWiND_env.sh
conda activate rewind
```



### WandB Configuration

This project uses Weights & Biases (WandB) for experiment tracking. Before running experiments:

1. **For Policy Training**: Edit `metaworld_policy_training/configs/base_config.yaml` lines 15-16:
   ```yaml
   wandb_entity_name: your-wandb-entity
   wandb_project_name: rewind-policy-training
   ```

2. **To Disable WandB**: Set `logging.wandb=false` when running policy training commands.

## Data Preparation (We recommend to run it with the Default path)

**Data Processing (Recommend run with Default path)**
```bash
# Download preprocessed OpenX DinoV2 Embeddings
python download_data.py --download_path DOWNLOADPATH(Default:datasets)
```

**Generate MetaWorld Trajectories for ReWiND Reward Training (Recommend run with Default path)**
```bash
# Generate Metaworld trajectories
python data_generation/metaworld_generation.py --save_path SAVE_DATA_PATH(Default:datasets)
# Centercrop the videos and convert to DinoV2 features
python data_preprocessing/metaworld_center_crop.py --video_path SAVE_DATA_PATH(Default:datasets) --target_path TARGET_DATASET_PATH(Default:datasets)  
python data_preprocessing/generate_dino_embeddings.py --video_path_folder TARGET_DATASET_PATH(Default:datasets) --target_path EMBEDDING_TARGET_PATH(Default:datasets)
```

## ReWiND Reward Model Training 
```bash
# require wandb entity
python train_reward.py --wandb_entity YOUR_WANDB_ENTITY(Required) \
--wandb_project WANDB_Project_NAME(Default:rewind-reward-training) \
--rewind \
--subsample_video \
--clip_grad \
--cosine_scheduler \
--batch_size 1024 \
--worker 1
```

### Reward Model Variants: Freeze and Optical Flow

The original ReWiND setup trains on forward and rewound DINOv2 embedding sequences and uses DINO goal-distance progress targets by default (`--progress_target_type dino_goal_distance`).
This repository also includes two MetaWorld-focused variants:

- **Freeze augmentation** (`--use_freeze`): inserts repeated frames into rewound training sequences. This teaches the reward model that visual stalling should not create artificial task progress.
- **Optical-flow progress targets** (`--progress_target_type optical_flow`): replaces DINO goal-distance progress with a scalar motion/progress target computed from frame differences in raw MetaWorld videos.

OpenX optical-flow preprocessing is experimental/WIP. For now, use optical-flow training primarily with MetaWorld data that was preprocessed from raw frames.

### MetaWorld Optical-Flow Preprocessing

Optical-flow targets are currently implemented for MetaWorld through frame-difference progress, not dense flow fields. The preprocessing requires raw MetaWorld frames because the scalar `flow_progress_<traj_id>` and `flow_signal_<traj_id>` datasets are computed before DINO-only embeddings are saved.

```bash
python data_generation/metaworld_generation.py --save_path datasets
python data_preprocessing/metaworld_center_crop.py --video_path datasets --target_path datasets
python data_preprocessing/generate_dino_embeddings.py --video_path_folder datasets --target_path datasets --motion_signal_type frame_diff
```

Expected MetaWorld files:
- Raw trajectories: `datasets/metaworld_generation.h5`
- Cropped frames: `datasets/metaworld_centercrop_32_train.h5`, `datasets/metaworld_centercrop_32_eval.h5`
- Embeddings and flow targets: `datasets/metaworld_embeddings_train.h5`, `datasets/metaworld_embeddings_eval.h5`

The embedding H5 groups should contain trajectory datasets (`0`, `1`, ...), `minilm_lang_embedding`, and matching `flow_progress_<traj_id>` / `flow_signal_<traj_id>` datasets.

### Training Commands

**ReWiND + freeze**
```bash
python train_reward.py --wandb_entity YOUR_WANDB_ENTITY \
--wandb_project rewind-reward-training \
--h5_folder_path datasets \
--openx_embedding_path datasets/full_openx_embeddings_v2_train.h5 \
--rewind \
--use_freeze \
--freeze_ratio 0.4 \
--subsample_video \
--clip_grad \
--cosine_scheduler \
--batch_size 1024 \
--worker 1
```

Important flags: `--use_freeze` enables frozen-frame augmentation, and `--freeze_ratio` controls how often frames are duplicated inside rewound sequences. This uses the default DINO goal-distance progress target. It requires the standard MetaWorld embedding files and the preprocessed OpenX embedding file used by the original ReWiND training loop.

Checkpoints are saved to:
```bash
checkpoints_dino_freeze/rewind_metaworld_epoch_<N>.pth
```

**ReWiND + optical flow**
```bash
python train_reward.py --wandb_entity YOUR_WANDB_ENTITY \
--wandb_project rewind-reward-training \
--h5_folder_path datasets \
--openx_embedding_path datasets/full_openx_embeddings_v2_train.h5 \
--rewind \
--progress_target_type optical_flow \
--subsample_video \
--clip_grad \
--cosine_scheduler \
--batch_size 1024 \
--worker 1
```

Important flags: `--progress_target_type optical_flow` makes the dataset loader read `flow_progress_<traj_id>` from the MetaWorld embedding H5. By default, missing flow targets fall back to linear progress (`--flow_missing_fallback linear`); use `--flow_missing_fallback error` when debugging dataset coverage. This variant requires MetaWorld embeddings generated from raw frames with `generate_dino_embeddings.py`.

Checkpoints are saved to:
```bash
checkpoints_flow/rewind_metaworld_epoch_<N>.pth
```

**ReWiND + freeze + optical flow**
```bash
python train_reward.py --wandb_entity YOUR_WANDB_ENTITY \
--wandb_project rewind-reward-training \
--h5_folder_path datasets \
--openx_embedding_path datasets/full_openx_embeddings_v2_train.h5 \
--rewind \
--use_freeze \
--freeze_ratio 0.4 \
--progress_target_type optical_flow \
--subsample_video \
--clip_grad \
--cosine_scheduler \
--batch_size 1024 \
--worker 1
```

This combines frozen-frame augmentation with frame-difference progress targets. It has the same raw-frame MetaWorld preprocessing requirement as the optical-flow variant.

Checkpoints are saved to:
```bash
checkpoints_flow_freeze/rewind_metaworld_epoch_<N>.pth
```



## ReWiND Metaworld Policy Training

### Label Offline Dataset (Recommend run with default path)
```bash
# Relabel the dataset we collect with ReWiND reward model
python data_preprocessing/metaworld_label_reward.py --reward_model_path CHECKPOINT_PATH --h5_video_path GENERATION_PATH --h5_embedding_path EMBEDDING_TARGET_PATH --output_path OUTPUT_PATH
```

Note:
- `OUTPUT_PATH`: The labeled dataset file path (default: `datasets/metaworld_labeled.h5`). This will be used as `<OUTPUT_PATH>` in [Offline Training](#offline-training) and [Online Training](#online-training) below.
- `CHECKPOINT_PATH`: a trained reward model checkpoint, for example `checkpoints_flow_freeze/rewind_metaworld_epoch_19.pth`.
- `GENERATION_PATH`: raw MetaWorld trajectories, normally `datasets/metaworld_generation.h5`.
- `EMBEDDING_TARGET_PATH`: MetaWorld training embeddings, normally `datasets/metaworld_embeddings_train.h5`; this is used for language embeddings during relabeling.

Example:
```bash
python data_preprocessing/metaworld_label_reward.py \
--reward_model_path checkpoints_flow_freeze/rewind_metaworld_epoch_19.pth \
--h5_video_path datasets/metaworld_generation.h5 \
--h5_embedding_path datasets/metaworld_embeddings_train.h5 \
--output_path datasets/metaworld_labeled_flow_freeze.h5
```

The relabeled H5 contains `action`, `rewards`, `done`, `policy_lang_embedding`, `img_embedding`, and `env_id`. Use the output path as `offline_training.offline_h5_path` for policy training.

```bash
cd metaworld_policy_training
```


### Policy Offline to Online RL Training
```bash
python train_policy.py metaworld=off_on_15 \
algorithm=wsrl_iql \
reward=rewind_metaworld \
offline_training.offline_training_steps=15000 \
general_training.seed=42 \
environment.env_id=<ENV_ID> \
offline_training.offline_h5_path=<OUTPUT_PATH> \
reward_model.model_path=<CHECKPOINT_PATH>
```

- `<ENV_ID>`: the Metaworld task you want to train online, e.g., `button-press-wall-v2`, `window-close-v2`. Full list of our (not in training data) evaluation tasks in the paper is: [`window-close-v2`, `reach-wall-v2`, `faucet-close-v2`, `coffee-button-v2`, `button-press-wall-v2`, `door-lock-v2`, `handle-press-side-v2`, `sweep-into-v2`]
- `<OFFLINE_CKPT_PATH>`: path to your offline-trained checkpoint directory (often contains `last_offline`) to warm-start online training. If set to `null`, the run will first execute the offline phase for `offline_training.offline_training_steps` steps on the dataset, and then proceed to the online phase. 
- To skip offline learning entirely, set `offline_training.offline_training_steps=0`.


### Optional: Policy Offline Training
We also provide code to just train the policy offline, so that you can load the same offline policy checkpoint for online RL to multiple new tasks downstream.
You only need to set `online_training.total_time_steps=0`. 

After offline training completes, check the `model_dir` in your wandb log to find the `<OFFLINE_CKPT_PATH>` for online training (see [Online Training](#online-training) below).

Then, run the above offline to online RL training command with `offline_training.ckpt_path=<OFFLINE_CKPT_PATH>` as an extra argument to perform online RL directly with the same offline policy.

Note:
- In offline training, `environment.env_id` is not important; the agent is trained over all training tasks found in your offline dataset.
- `<OUTPUT_PATH>` should point to your labeled offline dataset (see [Label Offline Dataset](#label-offline-dataset-recommend-run-with-default-path) above).


## FAQ & Debugging

### Freeze / Optical-Flow Notes

- Optical-flow reward training requires raw MetaWorld frames before DINO embedding generation. Embeddings-only datasets cannot be retrofitted unless the corresponding raw frames are available.
- OpenX embeddings-only datasets do not fully support optical-flow targets yet. OpenX optical-flow preprocessing is experimental/WIP.
- Keep large raw H5 files and generated embeddings under `datasets/` by default, or point commands to a scratch-backed directory with `--save_path`, `--target_path`, `--h5_folder_path`, and `--output_path`.
- H5 naming matters: each trajectory `N` should have matching `flow_progress_N` and `flow_signal_N` datasets when using `--progress_target_type optical_flow`.
- Reward checkpoints are written under `checkpoints_dino/`, `checkpoints_dino_freeze/`, `checkpoints_flow/`, `checkpoints_flow_freeze/`, `checkpoints/`, or `checkpoints_freeze/` depending on `--progress_target_type` and `--use_freeze`.

### Mujoco Installation

1. Download mujoco210 from [mujoco-py installation guide](https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco)
2. Extract the downloaded mujoco210 directory into `~/.mujoco/mujoco210`
3. Add the following lines to `~/.bashrc`:
   ```bash
   export LD_LIBRARY_PATH=~/.mujoco/mujoco210/bin
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
   ```
4. Reload your shell configuration:
   ```bash
   source ~/.bashrc

### Debug
```
fatal error: GL/glew.h: No such file or directory 4 | #include <GL/glew.h>
```
Solution:
check https://github.com/openai/mujoco-py/issues/745


## 📄 Citation
```bibtex
  @inproceedings{
      zhang2025rewind,
      title={ReWi{ND}: Language-Guided Rewards Teach Robot Policies without New Demonstrations},
      author={Jiahui Zhang and Yusen Luo and Abrar Anwar and Sumedh Anand Sontakke and Joseph J Lim and Jesse Thomason and Erdem Biyik and Jesse Zhang},
      booktitle={9th Annual Conference on Robot Learning},
      year={2025},
      url={https://openreview.net/forum?id=XjjXLxfPou}
    }
```
