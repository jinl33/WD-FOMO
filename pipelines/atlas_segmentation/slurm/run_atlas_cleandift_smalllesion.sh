#!/bin/bash
#SBATCH --job-name=atlas_cd_small
#SBATCH --account=wilms99
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # 2026-09-06 FIX: spgpu nodes are 32 CPU / 8 GPU = 4 CPU/GPU; requesting 8 for a
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/turbo/umms-wilms1/FOMO/experiments/jinhlee/logs/%x-%j.log

set -euo pipefail

MODEL_NAME="${MODEL_NAME:-cleandift_s23}"
RUN_STYLE="${RUN_STYLE:-cleandift_explore}"
PATCH_D="${PATCH_D:-192}"
PATCH_H="${PATCH_H:-256}"
PATCH_W="${PATCH_W:-192}"
BATCH_SIZE="${BATCH_SIZE:-4}"
AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-basic}"
SEGMENTATION_LOSS="${SEGMENTATION_LOSS:-dicece_sizeaware}"
SEG_ATTENTION="${SEG_ATTENTION:-se}"
SEG_HEAD_VARIANT="${SEG_HEAD_VARIANT:-waveletavg_refine}"
SEG_AUX_LOSS_WEIGHT="${SEG_AUX_LOSS_WEIGHT:-0.15}"
SEG_AUX_TARGET="${SEG_AUX_TARGET:-distance_shell}"
SEG_BOUNDARY_RADIUS="${SEG_BOUNDARY_RADIUS:-2}"
SEG_SMALL_LESION_WEIGHT="${SEG_SMALL_LESION_WEIGHT:-2.0}"
SEG_SMALL_THRESHOLDS="${SEG_SMALL_THRESHOLDS:-1000 10000 50000}"
P_OVERSAMPLE_FOREGROUND="${P_OVERSAMPLE_FOREGROUND:-0.33}"
EVAL_MODE="${EVAL_MODE:-argmax_no_tune}"

export MODEL_NAME
export RUN_STYLE
export PATCH_D
export PATCH_H
export PATCH_W
export BATCH_SIZE
export AUGMENTATION_PRESET
export SEGMENTATION_LOSS
export SEG_ATTENTION
export SEG_HEAD_VARIANT
export SEG_AUX_LOSS_WEIGHT
export SEG_AUX_TARGET
export SEG_BOUNDARY_RADIUS
export SEG_SMALL_LESION_WEIGHT
export SEG_SMALL_THRESHOLDS
export P_OVERSAMPLE_FOREGROUND
export EVAL_MODE

if [[ -n "${LIGHTWEIGHT_FOMO_ROOT:-}" ]]; then
  SCRIPT_ROOT="${LIGHTWEIGHT_FOMO_ROOT}/pipelines/atlas_segmentation/slurm"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/lightweight-fomo/pipelines/atlas_segmentation/slurm/run_atlas_segmentation_fullvol.sh" ]]; then
  SCRIPT_ROOT="${SLURM_SUBMIT_DIR}/lightweight-fomo/pipelines/atlas_segmentation/slurm"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pipelines/atlas_segmentation/slurm/run_atlas_segmentation_fullvol.sh" ]]; then
  SCRIPT_ROOT="${SLURM_SUBMIT_DIR}/pipelines/atlas_segmentation/slurm"
else
  SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

SCRIPT_PATH="${SCRIPT_ROOT}/run_atlas_segmentation_fullvol.sh"
exec bash "${SCRIPT_PATH}"
