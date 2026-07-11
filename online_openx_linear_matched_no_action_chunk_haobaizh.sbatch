#!/usr/bin/env bash
# Run the matched-linear ReWiND arm through the exact same online runner used
# by the pixel-difference arm. Only model/checkpoint paths and names differ.

#SBATCH --job-name=rw_on_linear
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

CODE_DIR="${CODE_DIR:-/project2/biyik_1165/haobaizh/rewind_incremental}"

export LABELED_H5="${LABELED_H5:-/scratch1/haobaizh/rewind_incremental/datasets/metaworld_labeled_openx_linear_matched_epoch19.h5}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-/scratch1/haobaizh/rewind_incremental/checkpoints_linear_matched/openx_linear_matched_20ep/rewind_metaworld_epoch_19.pth}"
export OFFLINE_OUTPUT_ROOT="${OFFLINE_OUTPUT_ROOT:-/scratch1/haobaizh/rewind_incremental/no_action_offline/openx_linear_matched_epoch19}"
export OFFLINE_EXPERIMENT_NAME="rewind_openx_linear_matched_epoch19_bonus200_10critics"
export ONLINE_OUTPUT_ROOT="${ONLINE_OUTPUT_ROOT:-/scratch1/haobaizh/rewind_incremental/no_action_online/openx_linear_matched_epoch19}"
export EXPECTED_PROGRESS_TARGET_TYPE="linear"
export EXPERIMENT_NAME="rewind_openx_linear_matched_epoch19_bonus200_no_base_10critics"
export RUN_DESCRIPTION="no-action OpenX matched-linear online policy"

exec bash "${CODE_DIR}/online_openx_flow_no_action_chunk_haobaizh.sbatch"
