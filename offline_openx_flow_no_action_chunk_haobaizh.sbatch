#!/usr/bin/env bash
# Train no-action-chunk WSRL/IQL offline policies from raw OpenX-flow ReWiND labels.

#SBATCH --job-name=rw_off_flow
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch1/haobaizh/rewind_incremental/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch1/haobaizh/rewind_incremental/slurm_logs/%x_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

NO_ACTION_REPO="${NO_ACTION_REPO:-/project2/biyik_1165/haobaizh/rewind_no-action-chunk}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch1/haobaizh/rewind_incremental}"
CONDA_ENV="${CONDA_ENV:-${NO_ACTION_REPO}/conda_envs/rewind_nochunk}"
LABELED_H5="${LABELED_H5:-${SCRATCH_ROOT}/datasets/metaworld_labeled_openx_flow_epoch19.h5}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${SCRATCH_ROOT}/checkpoints_flow/openx_flow_20ep/rewind_metaworld_epoch_19.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_ROOT}/no_action_offline/openx_flow_epoch19}"
OFFLINE_STEPS="${OFFLINE_STEPS:-100000}"
SUCCESS_BONUS="${SUCCESS_BONUS:-200}"
REWARD_DIVISOR="${REWARD_DIVISOR:-1.0}"
STAGE_DATA="${STAGE_DATA:-1}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-false}"

# Match the verified no-action-chunk baseline seeds.
SEEDS=(42 32 0)
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

if ! [[ "${ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || (( ARRAY_TASK_ID >= ${#SEEDS[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID must be in [0, $((${#SEEDS[@]} - 1))]" >&2
  exit 2
fi
if ! [[ "${OFFLINE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: OFFLINE_STEPS must be a positive integer: ${OFFLINE_STEPS}" >&2
  exit 2
fi
if [[ "${SUCCESS_BONUS}" != "200" && "${SUCCESS_BONUS}" != "200.0" ]]; then
  echo "ERROR: the matched ReWiND comparison requires SUCCESS_BONUS=200" >&2
  exit 2
fi
if [[ "${REWARD_DIVISOR}" != "1" && "${REWARD_DIVISOR}" != "1.0" ]]; then
  echo "ERROR: this raw-reward offline job requires REWARD_DIVISOR=1.0" >&2
  exit 2
fi

SEED="${SEEDS[$ARRAY_TASK_ID]}"
LOG_GROUP="offline_rewind_openx_flow_epoch19_bonus200_10critics"
LOG_DIR="${OUTPUT_ROOT}/${LOG_GROUP}_seed${SEED}"
STAGE_DIR="${TMPDIR:-/tmp/SLURM_${SLURM_JOB_ID}}/${USER}/rewind_offline_flow_seed${SEED}"
TRAIN_H5="${LABELED_H5}"

required_files=(
  "${NO_ACTION_REPO}/test_scripts/test_iql.py"
  "${NO_ACTION_REPO}/offline_rl_algorithms/transition_utils.py"
  "${LABELED_H5}"
  "${REWARD_MODEL_PATH}"
)

for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: missing required file: ${required_file}" >&2
    exit 2
  fi
done
if [[ ! -d "${CONDA_ENV}" ]]; then
  echo "ERROR: conda environment not found: ${CONDA_ENV}" >&2
  exit 2
fi

mkdir -p \
  "${SCRATCH_ROOT}/slurm_logs" \
  "${OUTPUT_ROOT}" \
  "${SCRATCH_ROOT}/wandb" \
  "${SCRATCH_ROOT}/wandb_cache" \
  "${SCRATCH_ROOT}/cache" \
  "/scratch1/haobaizh/hf_cache" \
  "/scratch1/haobaizh/torch_cache"

if [[ -e "${LOG_DIR}" ]]; then
  if [[ "${OVERWRITE_OUTPUT}" != "true" ]]; then
    echo "ERROR: refusing to overwrite existing output: ${LOG_DIR}" >&2
    echo "Set OVERWRITE_OUTPUT=true only if replacement is intentional." >&2
    exit 2
  fi
  rm -rf -- "${LOG_DIR}"
fi

if [[ "${STAGE_DATA}" == "1" ]]; then
  mkdir -p "${STAGE_DIR}"
  TRAIN_H5="${STAGE_DIR}/$(basename "${LABELED_H5}")"
  echo "Staging ${LABELED_H5} to ${TRAIN_H5}"
  cp -L -p "${LABELED_H5}" "${TRAIN_H5}"
fi

module purge
module load gcc/13.3.0
module load mesa/24.2.3
module load mesa-glu/9.0.2
module load glew/2.2.0
module load glx/1.4

# The cloned Qt activation hook reads this under `set -u`.
export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"

source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
GCC_LIBDIR="$(dirname "$(g++ -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="${NO_ACTION_REPO}:${PYTHONPATH:-}"
export HF_HOME="/scratch1/haobaizh/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="/scratch1/haobaizh/torch_cache"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/cache"
export WANDB_CONFIG_DIR="${HOME}/.config/wandb"
export WANDB_CACHE_DIR="${SCRATCH_ROOT}/wandb_cache"
export WANDB_DIR="${SCRATCH_ROOT}/wandb"
export WANDB_MODE=online
export WANDB_RUN_NAME="${LOG_GROUP}_seed${SEED}"
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python - "${LABELED_H5}" "${TRAIN_H5}" <<'PY'
import os
import sys

import h5py
import numpy as np
import torch

from offline_rl_algorithms.offline_replay_buffers import H5ReplayBuffer

source_path, train_path = sys.argv[1:3]
if os.path.getsize(source_path) != os.path.getsize(train_path):
    raise SystemExit("ERROR: staged H5 size does not match the source")

required = [
    "state",
    "action",
    "rewards",
    "done",
    "policy_lang_embedding",
    "img_embedding",
    "env_id",
]
with h5py.File(train_path, "r") as h5_file:
    missing = [key for key in required if key not in h5_file]
    if missing:
        raise SystemExit(f"ERROR: missing H5 datasets: {missing}")

    lengths = {key: int(h5_file[key].shape[0]) for key in required}
    if len(set(lengths.values())) != 1 or lengths["rewards"] == 0:
        raise SystemExit(f"ERROR: inconsistent H5 dataset lengths: {lengths}")

    rewards = np.asarray(h5_file["rewards"], dtype=np.float32)
    dones = np.asarray(h5_file["done"], dtype=np.float32)
    if not np.isfinite(rewards).all():
        raise SystemExit("ERROR: rewards contain non-finite values")
    if bool(h5_file.attrs.get("use_progress_diff", False)):
        raise SystemExit("ERROR: labeled H5 unexpectedly uses progress-difference rewards")

    print(
        "OFFLINE_INPUT_OK "
        f"rows={len(rewards)} min={float(rewards.min()):.6f} "
        f"max={float(rewards.max()):.6f} mean={float(rewards.mean()):.6f} "
        "use_progress_diff=False"
    )

buffer = H5ReplayBuffer(
    train_path,
    success_bonus=200.0,
    use_language_embeddings=True,
    is_state_based=False,
    use_proprio=True,
    reward_divisor=1.0,
)
nonterminal_indices = np.flatnonzero(buffer.dones < 0.5)
terminal_indices = np.flatnonzero(buffer.dones >= 0.5)
if len(nonterminal_indices) == 0 or len(terminal_indices) == 0:
    raise SystemExit("ERROR: labeled H5 must contain terminal and non-terminal rows")

np.testing.assert_array_equal(
    buffer.next_observations[nonterminal_indices],
    buffer.observations[nonterminal_indices + 1],
)
np.testing.assert_allclose(
    buffer.rewards[terminal_indices],
    rewards[terminal_indices] + 200.0,
)
print(
    "OFFLINE_TRANSITIONS_OK "
    f"nonterminal={len(nonterminal_indices)} terminal={len(terminal_indices)} "
    "success_bonus=200"
)

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable")
print(f"CUDA_OK device={torch.cuda.get_device_name(0)}")
PY

mkdir -p "${LOG_DIR}"
cd "${NO_ACTION_REPO}"

echo "========== Train no-action OpenX-flow offline policy =========="
echo "NO_ACTION_REPO=${NO_ACTION_REPO}"
echo "TRAIN_H5=${TRAIN_H5}"
echo "REWARD_MODEL_PATH=${REWARD_MODEL_PATH}"
echo "LOG_DIR=${LOG_DIR}"
echo "WANDB_ENTITY=rewind-valuemodel"
echo "WANDB_PROJECT=rewind-policy-training"
echo "WANDB_GROUP=${LOG_GROUP}"
echo "SEED=${SEED}"
echo "OFFLINE_STEPS=${OFFLINE_STEPS}"
echo "N_CRITICS=10"
echo "N_CRITICS_TO_SAMPLE=2"
echo "SUCCESS_BONUS=${SUCCESS_BONUS}"
echo "REWARD_DIVISOR=${REWARD_DIVISOR}"
echo "USE_PROGRESS_DIFF=false"
echo "USE_BASE_REWARD=false"
echo "ONLINE_STEPS=0"
echo "==============================================================="

srun python -u test_scripts/test_iql.py \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  reward_model.model_path="${REWARD_MODEL_PATH}" \
  reward_model.success_bonus="${SUCCESS_BONUS}" \
  reward_model.reward_divisor="${REWARD_DIVISOR}" \
  reward_model.use_progress_diff=false \
  reward_model.use_reverse_progress_diff=false \
  reward_model.use_base_reward=false \
  reward_model.use_exponential_progress=false \
  offline_training.offline_h5_path="${TRAIN_H5}" \
  offline_training.offline_training_steps="${OFFLINE_STEPS}" \
  online_training.total_time_steps=0 \
  online_training.mix_buffers_ratio=0.0 \
  general_training.seed="${SEED}" \
  general_training.n_critics=10 \
  general_training.n_critics_to_sample=2 \
  environment.env_id=coffee-button-v2 \
  logging.wandb_entity_name=rewind-valuemodel \
  logging.wandb_project_name=rewind-policy-training \
  logging.wandb_group_name="${LOG_GROUP}" \
  logging.log_dir="${LOG_DIR}" \
  hydra.run.dir=.

expected_outputs=(
  "${LOG_DIR}/last_offline.zip"
  "${LOG_DIR}/last_offline_rlpd_offline.zip"
)
for output_path in "${expected_outputs[@]}"; do
  if [[ ! -s "${output_path}" ]]; then
    echo "ERROR: expected offline checkpoint is missing or empty: ${output_path}" >&2
    exit 2
  fi
  ls -lh "${output_path}"
done

echo "OFFLINE_TRAINING_OK seed=${SEED} log_dir=${LOG_DIR}"
