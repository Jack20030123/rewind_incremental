#!/usr/bin/env bash
# Train a linear-progress ReWiND control on the exact OpenX subset used by the
# pixel-difference run. The input H5 contains DINO embeddings plus flow labels;
# --progress_target_type linear ignores the flow labels.

#SBATCH --job-name=rw_linear
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch1/haobaizh/rewind_incremental/slurm_logs/%x_%j.out
#SBATCH --error=/scratch1/haobaizh/rewind_incremental/slurm_logs/%x_%j.err

set -euo pipefail

CODE_DIR="${CODE_DIR:-/project2/biyik_1165/haobaizh/rewind_incremental}"
METAWORLD_DATA_DIR="${METAWORLD_DATA_DIR:-/scratch1/haobaizh/rewind_incremental/datasets_rewind_scratch}"
OPENX_MATCHED_PATH="${OPENX_MATCHED_PATH:-/scratch1/haobaizh/rewind_incremental/datasets/full_openx_flow_embeddings_train.h5}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch1/haobaizh/rewind_incremental}"
CONDA_ENV="${CONDA_ENV:-/scratch1/haobaizh/conda_envs/rewind}"

TARGET_WANDB_ENTITY="${REWARD_WANDB_ENTITY:-rewind-valuemodel}"
TARGET_WANDB_PROJECT="${REWARD_WANDB_PROJECT:-rewind-reward-training}"
TARGET_WANDB_MODE="${REWARD_WANDB_MODE:-online}"
EPOCHS="${EPOCHS:-20}"
RUN_SUFFIX="${RUN_SUFFIX:-openx_linear_matched_20ep}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
WORKER="${WORKER:-1}"
SEED="${SEED:-42}"
STAGE_DATA="${STAGE_DATA:-1}"

CHECKPOINT_DIR="${SCRATCH_ROOT}/checkpoints_linear_matched/${RUN_SUFFIX}"
SLURM_LOG_DIR="${SCRATCH_ROOT}/slurm_logs"

if ! [[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EPOCHS must be a positive integer: ${EPOCHS}" >&2
  exit 2
fi

required_files=(
  "${CODE_DIR}/train_reward.py"
  "${METAWORLD_DATA_DIR}/metaworld_embeddings_train.h5"
  "${METAWORLD_DATA_DIR}/metaworld_embeddings_eval.h5"
  "${METAWORLD_DATA_DIR}/metaworld_dino_embeddings_eval_close_succ.h5"
  "${METAWORLD_DATA_DIR}/metaworld_dino_embeddings_eval_all_fail.h5"
  "${OPENX_MATCHED_PATH}"
)

for required_file in "${required_files[@]}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: missing required file: ${required_file}" >&2
    exit 2
  fi
done

if compgen -G "${CHECKPOINT_DIR}/rewind_metaworld_epoch_*.pth" > /dev/null; then
  echo "ERROR: refusing to overwrite existing checkpoints in ${CHECKPOINT_DIR}" >&2
  echo "Choose a different RUN_SUFFIX for a new run." >&2
  exit 2
fi

mkdir -p \
  "${CHECKPOINT_DIR}" \
  "${SLURM_LOG_DIR}" \
  "${SCRATCH_ROOT}/wandb" \
  "${SCRATCH_ROOT}/.cache/wandb" \
  "${SCRATCH_ROOT}/.config/wandb"

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

export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WANDB_ENV_FILE="${WANDB_ENV_FILE:-${HOME}/.config/wandb/env}"
if [[ -f "${WANDB_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${WANDB_ENV_FILE}"
fi
export WANDB_MODE="${TARGET_WANDB_MODE}"
export WANDB_DIR="${SCRATCH_ROOT}/wandb"
export WANDB_CACHE_DIR="${SCRATCH_ROOT}/.cache/wandb"
export WANDB_CONFIG_DIR="${SCRATCH_ROOT}/.config/wandb"
export WANDB_ENTITY="${TARGET_WANDB_ENTITY}"
export WANDB_PROJECT="${TARGET_WANDB_PROJECT}"

TRAIN_METAWORLD_DIR="${METAWORLD_DATA_DIR}"
TRAIN_OPENX_MATCHED_PATH="${OPENX_MATCHED_PATH}"

if [[ "${STAGE_DATA}" == "1" ]]; then
  LOCAL_DATA_DIR="${TMPDIR:-/tmp}/${USER}/rewind_linear_matched_${SLURM_JOB_ID}"
  mkdir -p "${LOCAL_DATA_DIR}"

  echo "Staging reward-training HDF5 files to ${LOCAL_DATA_DIR}"
  for file_name in \
    metaworld_embeddings_train.h5 \
    metaworld_embeddings_eval.h5 \
    metaworld_dino_embeddings_eval_close_succ.h5 \
    metaworld_dino_embeddings_eval_all_fail.h5
  do
    cp -L -p "${METAWORLD_DATA_DIR}/${file_name}" "${LOCAL_DATA_DIR}/${file_name}"
  done
  cp -L -p "${OPENX_MATCHED_PATH}" "${LOCAL_DATA_DIR}/full_openx_flow_embeddings_train.h5"

  TRAIN_METAWORLD_DIR="${LOCAL_DATA_DIR}"
  TRAIN_OPENX_MATCHED_PATH="${LOCAL_DATA_DIR}/full_openx_flow_embeddings_train.h5"
  du -sh "${LOCAL_DATA_DIR}"
fi

cd "${CODE_DIR}"

echo "========== ReWiND matched linear reward training =========="
echo "CODE_DIR=${CODE_DIR}"
echo "TRAIN_METAWORLD_DIR=${TRAIN_METAWORLD_DIR}"
echo "TRAIN_OPENX_MATCHED_PATH=${TRAIN_OPENX_MATCHED_PATH}"
echo "PROGRESS_TARGET_TYPE=linear"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "WANDB_ENTITY=${WANDB_ENTITY}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "EPOCHS=${EPOCHS}"
echo "RUN_SUFFIX=${RUN_SUFFIX}"
echo "SEED=${SEED}"
echo "STAGE_DATA=${STAGE_DATA}"
echo "==========================================================="

srun python -u train_reward.py \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_project "${WANDB_PROJECT}" \
  --h5_folder_path "${TRAIN_METAWORLD_DIR}" \
  --openx_embedding_path "${TRAIN_OPENX_MATCHED_PATH}" \
  --rewind \
  --progress_target_type linear \
  --subsample_video \
  --clip_grad \
  --cosine_scheduler \
  --batch_size "${BATCH_SIZE}" \
  --worker "${WORKER}" \
  --epochs "${EPOCHS}" \
  --eval_interval 1 \
  --seed "${SEED}" \
  --log_pred_target_plot \
  --run_suffix "${RUN_SUFFIX}" \
  --checkpoint_dir "${CHECKPOINT_DIR}"

FINAL_EPOCH=$((EPOCHS - 1))
FINAL_CHECKPOINT="${CHECKPOINT_DIR}/rewind_metaworld_epoch_${FINAL_EPOCH}.pth"
if [[ ! -s "${FINAL_CHECKPOINT}" ]]; then
  echo "ERROR: final checkpoint is missing or empty: ${FINAL_CHECKPOINT}" >&2
  exit 2
fi

CHECKPOINT_COUNT="$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'rewind_metaworld_epoch_*.pth' | wc -l | tr -d ' ')"
if [[ "${CHECKPOINT_COUNT}" != "${EPOCHS}" ]]; then
  echo "ERROR: expected ${EPOCHS} checkpoints, found ${CHECKPOINT_COUNT}" >&2
  exit 2
fi

echo "LINEAR_MATCHED_TRAINING_OK checkpoint=${FINAL_CHECKPOINT} count=${CHECKPOINT_COUNT}"
