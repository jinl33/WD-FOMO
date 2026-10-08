#!/bin/bash
#SBATCH --job-name=atlas_base
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

MODEL_NAME="${MODEL_NAME:-unet_b}"
RUN_STYLE="${RUN_STYLE:-local_wrapper}"
PATCH_D="${PATCH_D:-96}"
PATCH_H="${PATCH_H:-96}"
PATCH_W="${PATCH_W:-96}"
BATCH_SIZE="${BATCH_SIZE:-2}"
AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-basic}"
SEGMENTATION_LOSS="${SEGMENTATION_LOSS:-dicece}"
SEG_HEAD_VARIANT="${SEG_HEAD_VARIANT:-earlyfusion}"
SEG_ATTENTION="${SEG_ATTENTION:-none}"
EVAL_MODE="${EVAL_MODE:-argmax_no_tune}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/run_atlas_segmentation_fullvol.sh" ]]; then
	SCRIPT_PATH="${SLURM_SUBMIT_DIR}/run_atlas_segmentation_fullvol.sh"
elif [[ -n "${LIGHTWEIGHT_FOMO_ROOT:-}" && -f "${LIGHTWEIGHT_FOMO_ROOT}/pipelines/atlas_segmentation/slurm/run_atlas_segmentation_fullvol.sh" ]]; then
	SCRIPT_PATH="${LIGHTWEIGHT_FOMO_ROOT}/pipelines/atlas_segmentation/slurm/run_atlas_segmentation_fullvol.sh"
else
	SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_atlas_segmentation_fullvol.sh"
fi
exec "${SCRIPT_PATH}"
