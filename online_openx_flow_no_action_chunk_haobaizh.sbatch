#!/usr/bin/env bash
# Run no-action-chunk online WSRL from the pixel-difference ReWiND offline
# checkpoints. This file is also the shared runner for the matched-linear arm.

#SBATCH --job-name=rw_on_flow
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch1/haobaizh/rewind_incremental/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch1/haobaizh/rewind_incremental/slurm_logs/%x_%A_%a.err
#SBATCH --array=0-23

set -euo pipefail

NO_ACTION_REPO="${NO_ACTION_REPO:-/project2/biyik_1165/haobaizh/rewind_no-action-chunk}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch1/haobaizh/rewind_incremental}"
CONDA_ENV="${CONDA_ENV:-${NO_ACTION_REPO}/conda_envs/rewind_nochunk}"
LABELED_H5="${LABELED_H5:-${SCRATCH_ROOT}/datasets/metaworld_labeled_openx_flow_epoch19.h5}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${SCRATCH_ROOT}/checkpoints_flow/openx_flow_20ep/rewind_metaworld_epoch_19.pth}"
OFFLINE_OUTPUT_ROOT="${OFFLINE_OUTPUT_ROOT:-${SCRATCH_ROOT}/no_action_offline/openx_flow_epoch19}"
OFFLINE_EXPERIMENT_NAME="${OFFLINE_EXPERIMENT_NAME:-rewind_openx_flow_epoch19_bonus200_10critics}"
ONLINE_OUTPUT_ROOT="${ONLINE_OUTPUT_ROOT:-${SCRATCH_ROOT}/no_action_online/openx_flow_epoch19}"
EXPECTED_PROGRESS_TARGET_TYPE="${EXPECTED_PROGRESS_TARGET_TYPE:-optical_flow}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rewind_openx_flow_epoch19_bonus200_no_base_10critics}"
RUN_DESCRIPTION="${RUN_DESCRIPTION:-no-action OpenX pixel-difference online policy}"
ONLINE_STEPS="${ONLINE_STEPS:-100000}"
SUCCESS_BONUS="${SUCCESS_BONUS:-200}"
REWARD_DIVISOR="${REWARD_DIVISOR:-1.0}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-false}"

ENVS=(
  window-close-v2
  reach-wall-v2
  faucet-close-v2
  coffee-button-v2
  button-press-wall-v2
  door-lock-v2
  handle-press-side-v2
  sweep-into-v2
)
SEEDS=(42 32 0)

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
NUM_SEEDS="${#SEEDS[@]}"
TOTAL_JOBS=$(( ${#ENVS[@]} * NUM_SEEDS ))

if ! [[ "${ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || (( ARRAY_TASK_ID >= TOTAL_JOBS )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID must be in [0, $((TOTAL_JOBS - 1))]" >&2
  exit 2
fi
if [[ "${ONLINE_STEPS}" != "100000" ]]; then
  echo "ERROR: the matched comparison requires ONLINE_STEPS=100000" >&2
  exit 2
fi
if [[ "${SUCCESS_BONUS}" != "200" && "${SUCCESS_BONUS}" != "200.0" ]]; then
  echo "ERROR: the matched comparison requires SUCCESS_BONUS=200" >&2
  exit 2
fi
if [[ "${REWARD_DIVISOR}" != "1" && "${REWARD_DIVISOR}" != "1.0" ]]; then
  echo "ERROR: the matched comparison requires REWARD_DIVISOR=1.0" >&2
  exit 2
fi

ENV_IDX=$(( ARRAY_TASK_ID / NUM_SEEDS ))
SEED_IDX=$(( ARRAY_TASK_ID % NUM_SEEDS ))
ENV_ID="${ENVS[$ENV_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

OFFLINE_GROUP="offline_${OFFLINE_EXPERIMENT_NAME}"
OFFLINE_DIR="${OFFLINE_OUTPUT_ROOT}/${OFFLINE_GROUP}_seed${SEED}"
OFFLINE_CKPT="${OFFLINE_DIR}/last_offline"
ONLINE_GROUP="${ENV_ID}_online_${EXPERIMENT_NAME}"
LOG_DIR="${ONLINE_OUTPUT_ROOT}/${ENV_ID}/${ONLINE_GROUP}_seed${SEED}"

required_files=(
  "${NO_ACTION_REPO}/test_scripts/test_iql.py"
  "${LABELED_H5}"
  "${REWARD_MODEL_PATH}"
  "${OFFLINE_CKPT}.zip"
  "${OFFLINE_CKPT}_rlpd_offline.zip"
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
  "${ONLINE_OUTPUT_ROOT}" \
  "${SCRATCH_ROOT}/wandb" \
  "${SCRATCH_ROOT}/wandb_cache" \
  "${SCRATCH_ROOT}/cache" \
  "/scratch1/haobaizh/hf_cache" \
  "/scratch1/haobaizh/torch_cache"

if [[ -e "${LOG_DIR}" || -e "${LOG_DIR}.zip" ]]; then
  if [[ "${OVERWRITE_OUTPUT}" != "true" ]]; then
    echo "ERROR: refusing to overwrite existing online output: ${LOG_DIR}" >&2
    echo "Set OVERWRITE_OUTPUT=true only if replacement is intentional." >&2
    exit 2
  fi
  rm -rf -- "${LOG_DIR}" "${LOG_DIR}.zip"
fi

module purge
module load gcc/13.3.0
module load mesa/24.2.3
module load mesa-glu/9.0.2
module load glew/2.2.0
module load glx/1.4

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
export WANDB_RUN_NAME="${ONLINE_GROUP}_seed${SEED}"
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python - \
  "${REWARD_MODEL_PATH}" \
  "${EXPECTED_PROGRESS_TARGET_TYPE}" \
  "${LABELED_H5}" \
  "${OFFLINE_CKPT}.zip" \
  "${OFFLINE_CKPT}_rlpd_offline.zip" <<'PY'
import sys
from pathlib import Path
import zipfile

import h5py
import torch

reward_path, expected_target, label_path, policy_path, iql_path = sys.argv[1:6]

checkpoint = torch.load(reward_path, map_location="cpu", weights_only=False)
checkpoint_args = checkpoint.get("args")
actual_target = getattr(checkpoint_args, "progress_target_type", None)
if actual_target != expected_target:
    raise SystemExit(
        "ERROR: reward checkpoint target mismatch: "
        f"expected={expected_target} actual={actual_target}"
    )
if not bool(getattr(checkpoint_args, "rewind", False)):
    raise SystemExit("ERROR: reward checkpoint is not a ReWiND checkpoint")
openx_file = Path(getattr(checkpoint_args, "openx_embedding_path", "")).name
if openx_file != "full_openx_flow_embeddings_train.h5":
    raise SystemExit(
        "ERROR: reward checkpoint was not trained on the matched OpenX file: "
        f"{openx_file}"
    )

with h5py.File(label_path, "r") as h5_file:
    rows = int(h5_file["rewards"].shape[0])
    if rows != 35210:
        raise SystemExit(f"ERROR: expected 35210 labeled rows, found {rows}")
    if bool(h5_file.attrs.get("use_progress_diff", False)):
        raise SystemExit("ERROR: online provenance H5 unexpectedly uses reward differences")

for path in (policy_path, iql_path):
    if not zipfile.is_zipfile(path):
        raise SystemExit(f"ERROR: invalid offline checkpoint zip: {path}")

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable")

print(
    "ONLINE_PREFLIGHT_OK "
    f"progress_target_type={actual_target} openx_file={openx_file} "
    f"label_rows={rows} success_bonus=200 use_progress_diff=False "
    "use_base_reward=False"
)
print(f"CUDA_OK device={torch.cuda.get_device_name(0)}")
PY

mkdir -p "${LOG_DIR}"
cd "${LOG_DIR}"

echo "========== Train ${RUN_DESCRIPTION} =========="
echo "NO_ACTION_REPO=${NO_ACTION_REPO}"
echo "ENV_ID=${ENV_ID}"
echo "SEED=${SEED}"
echo "REWARD_MODEL_PATH=${REWARD_MODEL_PATH}"
echo "OFFLINE_CKPT=${OFFLINE_CKPT}"
echo "LOG_DIR=${LOG_DIR}"
echo "WANDB_ENTITY=rewind-valuemodel"
echo "WANDB_PROJECT=rewind-policy-training"
echo "WANDB_GROUP=${ONLINE_GROUP}"
echo "ONLINE_STEPS=${ONLINE_STEPS}"
echo "OFFLINE_STEPS=0"
echo "MIX_BUFFERS_RATIO=0.0"
echo "WARM_START_ONLINE_RL=true"
echo "LEARNING_STARTS=4000"
echo "ONLINE_CRITIC_UPDATE_RATIO=4"
echo "N_CRITICS=10"
echo "N_CRITICS_TO_SAMPLE=2"
echo "SUCCESS_BONUS=${SUCCESS_BONUS}"
echo "REWARD_DIVISOR=${REWARD_DIVISOR}"
echo "USE_PROGRESS_DIFF=false"
echo "USE_BASE_REWARD=false"
echo "EXPECTED_PROGRESS_TARGET_TYPE=${EXPECTED_PROGRESS_TARGET_TYPE}"
echo "==============================================================="

srun python -u "${NO_ACTION_REPO}/test_scripts/test_iql.py" \
  metaworld=off_on_15 \
  algorithm=wsrl_iql \
  reward=rewind \
  reward_model.model_path="${REWARD_MODEL_PATH}" \
  reward_model.success_bonus="${SUCCESS_BONUS}" \
  reward_model.reward_divisor="${REWARD_DIVISOR}" \
  reward_model.reward_at_every_step=true \
  reward_model.sum_reward=false \
  reward_model.use_progress_diff=false \
  reward_model.use_reverse_progress_diff=false \
  reward_model.use_base_reward=false \
  reward_model.use_exponential_progress=false \
  offline_training.offline_h5_path="${LABELED_H5}" \
  offline_training.ckpt_path="${OFFLINE_CKPT}" \
  offline_training.offline_training_steps=0 \
  offline_training.critic_update_ratio=1 \
  online_training.total_time_steps="${ONLINE_STEPS}" \
  online_training.mix_buffers_ratio=0.0 \
  online_training.warm_start_online_rl=true \
  online_training.learning_starts=4000 \
  online_training.critic_update_ratio=4 \
  general_training.seed="${SEED}" \
  general_training.learning_rate=1e-3 \
  general_training.n_critics=10 \
  general_training.n_critics_to_sample=2 \
  general_training.normalize_reward=false \
  general_training.terminate_on_success=true \
  model.policy_layer_norm=true \
  model.critic_layer_norm=true \
  environment.env_id="${ENV_ID}" \
  environment.n_envs=1 \
  logging.eval_freq=10000 \
  logging.video_freq=20000 \
  logging.wandb_entity_name=rewind-valuemodel \
  logging.wandb_project_name=rewind-policy-training \
  logging.wandb_group_name="${ONLINE_GROUP}" \
  logging.log_dir="${LOG_DIR}" \
  hydra.run.dir=.

if [[ ! -s "${LOG_DIR}.zip" ]]; then
  echo "ERROR: final online checkpoint is missing or empty: ${LOG_DIR}.zip" >&2
  exit 2
fi
if [[ ! -s "${LOG_DIR}/evaluations.npz" ]]; then
  echo "ERROR: online evaluation history is missing: ${LOG_DIR}/evaluations.npz" >&2
  exit 2
fi

ls -lh "${LOG_DIR}.zip" "${LOG_DIR}/evaluations.npz"
echo "ONLINE_TRAINING_OK env=${ENV_ID} seed=${SEED} log_dir=${LOG_DIR}"
