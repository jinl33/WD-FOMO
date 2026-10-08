#!/bin/bash
#SBATCH --job-name=atlas_cd
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
RUN_STYLE="${RUN_STYLE:-cleandift_control}"
PATCH_D="${PATCH_D:-196}"
PATCH_H="${PATCH_H:-256}"
PATCH_W="${PATCH_W:-196}"
BATCH_SIZE="${BATCH_SIZE:-4}"
AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-basic}"
SEGMENTATION_LOSS="${SEGMENTATION_LOSS:-dicece}"
SEG_ATTENTION="${SEG_ATTENTION:-se}"
SEG_HEAD_VARIANT="${SEG_HEAD_VARIANT:-image_refine_v1}"
SEG_AUX_LOSS_WEIGHT="${SEG_AUX_LOSS_WEIGHT:-0.0}"
P_OVERSAMPLE_FOREGROUND="${P_OVERSAMPLE_FOREGROUND:-0.33}"
EVAL_MODE="${EVAL_MODE:-argmax_no_tune}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_atlas_segmentation_fullvol.sh" ]]; then
	SCRIPT_PATH="${SLURM_SUBMIT_DIR}/run_atlas_segmentation_fullvol.sh"
elif [[ -n "${LIGHTWEIGHT_FOMO_ROOT:-}" && -f "${LIGHTWEIGHT_FOMO_ROOT}/pipelines/atlas_segmentation/slurm/run_atlas_segmentation_fullvol.sh" ]]; then
	SCRIPT_PATH="${LIGHTWEIGHT_FOMO_ROOT}/pipelines/atlas_segmentation/slurm/run_atlas_segmentation_fullvol.sh"
else
	SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_atlas_segmentation_fullvol.sh"
fi
exec "${SCRIPT_PATH}"
