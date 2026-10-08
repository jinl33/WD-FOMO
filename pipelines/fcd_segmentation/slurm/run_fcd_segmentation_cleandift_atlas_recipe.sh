#!/bin/bash
#SBATCH --job-name=fcd_cd_atlas
#SBATCH --account=wilms99
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # 2026-09-06 FIX: spgpu nodes are 32 CPU / 8 GPU = 4 CPU/GPU; requesting 8 for a
                            # 1-GPU job double-bills against the account's GPU allocation (admin-flagged).
#SBATCH --mem=72G  # 2026-09-06 FIX: measured peak host RSS across 40+ completed runs of this exact
                    # recipe (both the original 2026-07-28 OOM-fix runs and this program's own runs)
                    # clusters at 58-63 GB; 96G was never observed to be needed and, per the admin,
                    # also drives the same over-fair-share GPU billing penalty as excess CPUs. 72G
                    # keeps ~9-14 GB of margin above every observed peak. 48G (the pre-2026-07-28
                    # default) is NOT safe -- that is the value that originally OOM-killed this job.
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/turbo/umms-wilms1/FOMO/experiments/jinhlee/logs/%x-%j.log

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-cleandift_s23}"
SPLIT_METHOD="${SPLIT_METHOD:-nested_balanced_qc76_kfold}"
PATCH_D="${PATCH_D:-192}"
PATCH_H="${PATCH_H:-256}"
PATCH_W="${PATCH_W:-192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-basic}"
SEGMENTATION_LOSS="${SEGMENTATION_LOSS:-dicece}"
SEG_HEAD_VARIANT="${SEG_HEAD_VARIANT:-waveletavg_refine}"
SEG_ATTENTION="${SEG_ATTENTION:-se}"
P_OVERSAMPLE_FOREGROUND="${P_OVERSAMPLE_FOREGROUND:-0.33}"
BEST_CHECKPOINT_MONITOR="${BEST_CHECKPOINT_MONITOR:-val/dice}"
BEST_CHECKPOINT_MODE="${BEST_CHECKPOINT_MODE:-max}"
EARLY_STOPPING_MONITOR="${EARLY_STOPPING_MONITOR:-val/dice}"
EARLY_STOPPING_MODE="${EARLY_STOPPING_MODE:-max}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"
CHECKPOINT_SELECT="${CHECKPOINT_SELECT:-best}"
EVAL_MODE="${EVAL_MODE:-argmax_no_tune}"
MIRROR_TTA="${MIRROR_TTA:-0}"
INPUT_MODALITY_INDICES="${INPUT_MODALITY_INDICES:-}"
NUM_WORKERS="${NUM_WORKERS:-2}"

export MODEL_NAME
export SPLIT_METHOD
export PATCH_D
export PATCH_H
export PATCH_W
export BATCH_SIZE
export ACCUMULATE_GRAD_BATCHES
export AUGMENTATION_PRESET
export SEGMENTATION_LOSS
export SEG_HEAD_VARIANT
export SEG_ATTENTION
export P_OVERSAMPLE_FOREGROUND
export BEST_CHECKPOINT_MONITOR
export BEST_CHECKPOINT_MODE
export EARLY_STOPPING_MONITOR
export EARLY_STOPPING_MODE
export EARLY_STOPPING_PATIENCE
export CHECKPOINT_SELECT
export EVAL_MODE
export MIRROR_TTA
export INPUT_MODALITY_INDICES
export NUM_WORKERS

if [[ -n "${LIGHTWEIGHT_FOMO_ROOT:-}" ]]; then
  PIPELINE_ROOT="${LIGHTWEIGHT_FOMO_ROOT}/pipelines/fcd_segmentation/slurm"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/lightweight-fomo/pipelines/fcd_segmentation/slurm/run_fcd_segmentation_nested_tuned.sh" ]]; then
  PIPELINE_ROOT="${SLURM_SUBMIT_DIR}/lightweight-fomo/pipelines/fcd_segmentation/slurm"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pipelines/fcd_segmentation/slurm/run_fcd_segmentation_nested_tuned.sh" ]]; then
  PIPELINE_ROOT="${SLURM_SUBMIT_DIR}/pipelines/fcd_segmentation/slurm"
else
  PIPELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

SCRIPT_PATH="${PIPELINE_ROOT}/run_fcd_segmentation_nested_tuned.sh"
exec bash "${SCRIPT_PATH}"
