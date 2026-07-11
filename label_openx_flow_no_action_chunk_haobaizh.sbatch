#!/usr/bin/env bash
#SBATCH --job-name=rw_lbl_flow
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch1/haobaizh/rewind_incremental/slurm_logs/%x_%j.out
#SBATCH --error=/scratch1/haobaizh/rewind_incremental/slurm_logs/%x_%j.err

set -euo pipefail

NO_ACTION_REPO="${NO_ACTION_REPO:-/project2/biyik_1165/haobaizh/rewind_no-action-chunk}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-/scratch1/haobaizh/rewind_incremental/datasets_rewind_scratch}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch1/haobaizh/rewind_incremental}"
CONDA_ENV="${CONDA_ENV:-${NO_ACTION_REPO}/conda_envs/rewind_nochunk}"
REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${SCRATCH_ROOT}/checkpoints_flow/openx_flow_20ep/rewind_metaworld_epoch_19.pth}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRATCH_ROOT}/datasets/metaworld_labeled_openx_flow_epoch19.h5}"
TORCH_CACHE="${TORCH_CACHE:-/scratch1/haobaizh/torch_cache}"
STAGE_DATA="${STAGE_DATA:-1}"
OVERWRITE_OUTPUT="${OVERWRITE_OUTPUT:-false}"

required_files=(
  "${NO_ACTION_REPO}/scripts/generate_labeled_dataset.py"
  "${SOURCE_DATA_DIR}/metaworld_generation.h5"
  "${SOURCE_DATA_DIR}/metaworld_embeddings_train.h5"
  "${REWARD_MODEL_PATH}"
)

for required_file in "${required_files[@]}"; do
  if [ ! -f "${required_file}" ]; then
    echo "ERROR: missing required file: ${required_file}" >&2
    exit 2
  fi
done

if [ -e "${OUTPUT_PATH}" ] && [ "${OVERWRITE_OUTPUT}" != "true" ]; then
  echo "ERROR: refusing to overwrite existing output: ${OUTPUT_PATH}" >&2
  echo "Set OVERWRITE_OUTPUT=true only if replacement is intentional." >&2
  exit 2
fi

mkdir -p \
  "$(dirname "${OUTPUT_PATH}")" \
  "${SCRATCH_ROOT}/slurm_logs" \
  "${TORCH_CACHE}"

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
export TORCH_HOME="${TORCH_CACHE}"
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -c "import h5py, numpy, torch; assert torch.cuda.is_available(); print('label environment and CUDA ok')"

LABEL_DATA_DIR="${SOURCE_DATA_DIR}"
if [ "${STAGE_DATA}" = "1" ]; then
  LABEL_DATA_DIR="${TMPDIR:-/tmp}/${USER}/rewind_label_flow_${SLURM_JOB_ID}"
  mkdir -p "${LABEL_DATA_DIR}"

  echo "Staging label inputs to ${LABEL_DATA_DIR}"
  for file_name in metaworld_generation.h5 metaworld_embeddings_train.h5; do
    cp -L -p "${SOURCE_DATA_DIR}/${file_name}" "${LABEL_DATA_DIR}/${file_name}"
    source_size="$(stat -Lc %s "${SOURCE_DATA_DIR}/${file_name}")"
    staged_size="$(stat -Lc %s "${LABEL_DATA_DIR}/${file_name}")"
    if [ "${source_size}" != "${staged_size}" ]; then
      echo "ERROR: staged size mismatch for ${file_name}" >&2
      exit 2
    fi
  done
  du -sh "${LABEL_DATA_DIR}"
fi

cd "${NO_ACTION_REPO}"

echo "========== Generate no-action OpenX-flow ReWiND labels =========="
echo "NO_ACTION_REPO=${NO_ACTION_REPO}"
echo "LABEL_DATA_DIR=${LABEL_DATA_DIR}"
echo "REWARD_MODEL_PATH=${REWARD_MODEL_PATH}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "TORCH_HOME=${TORCH_HOME}"
echo "STAGE_DATA=${STAGE_DATA}"
echo "REWARD_MODE=raw_progress_output"
echo "================================================================="

srun python -u scripts/generate_labeled_dataset.py \
  --h5_video_path "${LABEL_DATA_DIR}/metaworld_generation.h5" \
  --h5_embedding_path "${LABEL_DATA_DIR}/metaworld_embeddings_train.h5" \
  --reward_model_path "${REWARD_MODEL_PATH}" \
  --output_path "${OUTPUT_PATH}"

export OUTPUT_PATH
python - <<'PY'
import os

import h5py
import numpy as np

path = os.environ["OUTPUT_PATH"]
required = [
    "state",
    "action",
    "rewards",
    "done",
    "policy_lang_embedding",
    "img_embedding",
    "env_id",
]

with h5py.File(path, "r") as h5_file:
    missing = [key for key in required if key not in h5_file]
    if missing:
        raise RuntimeError(f"Missing output datasets: {missing}")

    lengths = {key: len(h5_file[key]) for key in required}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values())) == 0:
        raise RuntimeError(f"Invalid output lengths: {lengths}")

    rewards = np.asarray(h5_file["rewards"], dtype=np.float32)
    if not np.isfinite(rewards).all():
        raise RuntimeError("Output rewards contain non-finite values")

    use_progress_diff = bool(h5_file.attrs.get("use_progress_diff", False))
    if use_progress_diff:
        raise RuntimeError("Output unexpectedly uses progress-difference rewards")

    print(f"LABEL_OUTPUT_OK path={path}")
    print(f"timesteps={len(rewards)}")
    print(
        "reward_stats "
        f"min={rewards.min():.6f} max={rewards.max():.6f} "
        f"mean={rewards.mean():.6f} std={rewards.std():.6f}"
    )
    print(f"use_progress_diff={use_progress_diff}")
PY
