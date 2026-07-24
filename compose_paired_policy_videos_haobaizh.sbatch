#!/usr/bin/env bash
# Stack the existing Linear and Pixel-Difference rollout videos for reporting.

#SBATCH --job-name=rw_pair
#SBATCH --account=biyik_1165
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch1/haobaizh/rewind_incremental/slurm_logs/%x_%j.out
#SBATCH --error=/scratch1/haobaizh/rewind_incremental/slurm_logs/%x_%j.err

set -euo pipefail

CODE_DIR="${CODE_DIR:-/project2/biyik_1165/haobaizh/rewind_incremental}"
NO_ACTION_REPO="${NO_ACTION_REPO:-/project2/biyik_1165/haobaizh/rewind_no-action-chunk}"
CONDA_ENV="${CONDA_ENV:-${NO_ACTION_REPO}/conda_envs/rewind_nochunk}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch1/haobaizh/rewind_incremental}"

INPUT_ROOT="${INPUT_ROOT:-${SCRATCH_ROOT}/reward_rollout_comparison}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_ROOT}/paired_policy_comparison}"
ENVS="${ENVS:-window-close-v2,reach-wall-v2,faucet-close-v2}"
POLICY_SEED="${POLICY_SEED:-42}"
ROLLOUT_SEED="${ROLLOUT_SEED:-450}"
EPISODE="${EPISODE:-0}"

mkdir -p "${SCRATCH_ROOT}/slurm_logs" "${OUTPUT_DIR}"

module purge
module load gcc/13.3.0

export QT_XCB_GL_INTEGRATION="${QT_XCB_GL_INTEGRATION:-none}"
source /apps/conda/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${CODE_DIR}"

echo "========== Paired policy video composition =========="
echo "INPUT_ROOT=${INPUT_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ENVS=${ENVS}"
echo "POLICY_SEED=${POLICY_SEED}"
echo "ROLLOUT_SEED=${ROLLOUT_SEED}"
echo "EPISODE=${EPISODE}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "====================================================="

python -c "import imageio, numpy, PIL; print('VIDEO_ENV_OK')"

srun python -u scripts/compose_paired_policy_videos.py \
  --input-root "${INPUT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --envs "${ENVS}" \
  --policy-seed "${POLICY_SEED}" \
  --rollout-seed "${ROLLOUT_SEED}" \
  --episode "${EPISODE}" \
  --overwrite
