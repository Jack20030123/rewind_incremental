#!/usr/bin/env bash
# Roll out matched linear/pixel policies and score every trajectory prefix with
# both ReWiND reward checkpoints. Defaults: 3 tasks x 2 policy arms = 6 demos.

#SBATCH --job-name=rw_curve
#SBATCH --account=biyik_1165
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch1/haobaizh/rewind_incremental/slurm_logs/%x_%j.out
#SBATCH --error=/scratch1/haobaizh/rewind_incremental/slurm_logs/%x_%j.err

set -euo pipefail

CODE_DIR="${CODE_DIR:-/project2/biyik_1165/haobaizh/rewind_incremental}"
NO_ACTION_REPO="${NO_ACTION_REPO:-/project2/biyik_1165/haobaizh/rewind_no-action-chunk}"
CONDA_ENV="${CONDA_ENV:-${NO_ACTION_REPO}/conda_envs/rewind_nochunk}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch1/haobaizh/rewind_incremental}"

LINEAR_REWARD_CHECKPOINT="${LINEAR_REWARD_CHECKPOINT:-${SCRATCH_ROOT}/checkpoints_linear_matched/openx_linear_matched_20ep/rewind_metaworld_epoch_19.pth}"
PIXEL_REWARD_CHECKPOINT="${PIXEL_REWARD_CHECKPOINT:-${SCRATCH_ROOT}/checkpoints_flow/openx_flow_20ep/rewind_metaworld_epoch_19.pth}"
LINEAR_POLICY_ROOT="${LINEAR_POLICY_ROOT:-${SCRATCH_ROOT}/no_action_online/openx_linear_matched_epoch19}"
PIXEL_POLICY_ROOT="${PIXEL_POLICY_ROOT:-${SCRATCH_ROOT}/no_action_online/openx_flow_epoch19}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_ROOT}/reward_rollout_comparison}"

ENVS="${ENVS:-window-close-v2,reach-wall-v2,faucet-close-v2}"
POLICY_ARMS="${POLICY_ARMS:-linear,pixel}"
POLICY_SEED="${POLICY_SEED:-42}"
ROLLOUT_SEED="${ROLLOUT_SEED:-450}"
EPISODES_PER_POLICY="${EPISODES_PER_POLICY:-1}"

mkdir -p \
  "${SCRATCH_ROOT}/slurm_logs" \
  "${OUTPUT_DIR}" \
  "${SCRATCH_ROOT}/cache" \
  "${SCRATCH_ROOT}/mpl_cache" \
  "/scratch1/haobaizh/hf_cache" \
  "/scratch1/haobaizh/torch_cache"

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

export PYTHONPATH="${NO_ACTION_REPO}:${CODE_DIR}:${PYTHONPATH:-}"
export HF_HOME="/scratch1/haobaizh/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="/scratch1/haobaizh/torch_cache"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/cache"
export MPLCONFIGDIR="${SCRATCH_ROOT}/mpl_cache"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${CODE_DIR}"

echo "========== ReWiND reward rollout comparison =========="
echo "ENVS=${ENVS}"
echo "POLICY_ARMS=${POLICY_ARMS}"
echo "POLICY_SEED=${POLICY_SEED}"
echo "ROLLOUT_SEED=${ROLLOUT_SEED}"
echo "EPISODES_PER_POLICY=${EPISODES_PER_POLICY}"
echo "LINEAR_REWARD_CHECKPOINT=${LINEAR_REWARD_CHECKPOINT}"
echo "PIXEL_REWARD_CHECKPOINT=${PIXEL_REWARD_CHECKPOINT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "======================================================="

srun python -u scripts/compare_rewind_reward_rollouts.py \
  --no-action-repo "${NO_ACTION_REPO}" \
  --linear-reward-checkpoint "${LINEAR_REWARD_CHECKPOINT}" \
  --pixel-reward-checkpoint "${PIXEL_REWARD_CHECKPOINT}" \
  --linear-policy-root "${LINEAR_POLICY_ROOT}" \
  --pixel-policy-root "${PIXEL_POLICY_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --envs "${ENVS}" \
  --policy-arms "${POLICY_ARMS}" \
  --policy-seed "${POLICY_SEED}" \
  --rollout-seed "${ROLLOUT_SEED}" \
  --episodes-per-policy "${EPISODES_PER_POLICY}" \
  --max-steps 128 \
  --score-batch-size 64 \
  --fps 20 \
  --device cuda
