#!/bin/bash
#SBATCH --job-name=atlas_task12_single
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

if [[ -n "${LIGHTWEIGHT_FOMO_ROOT:-}" ]]; then
  REPO_ROOT="${LIGHTWEIGHT_FOMO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/lightweight-fomo/src/downstream/finetune.py" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}/lightweight-fomo"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/src/downstream/finetune.py" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

module load python/3.10.4
source "${FOMO_ENV:-/scratch/wilms_root/wilms0/jinhlee/fomo-diffusion/fomo_env/bin/activate}"

TASK_NAME="${TASK_NAME:-Task012_ATLAS2_StrokeLesion_T1_1mm}"
DATA_ROOT="${DATA_ROOT:-/scratch/wilms_root/wilms0/jinhlee/fomo-diffusion/atlas2_stroke_t1seg_1mm_all955_20260528}"
TASK_DIR="${DATA_ROOT}/${TASK_NAME}"
REFRESH_SPLIT="${REFRESH_SPLIT:-0}"

if [[ "${REFRESH_SPLIT}" == "1" ]]; then
  python3 "${REPO_ROOT}/pipelines/atlas_segmentation/scripts/refresh_atlas_single_split.py" \
    --task_dir "${TASK_DIR}" \
    --delete_stale_zero_byte_npys
else
  python3 << PYEOF
import csv
import pickle
from pathlib import Path

task_dir = Path("${TASK_DIR}")
manifest_rows = list(csv.DictReader(open(task_dir / "manifest_atlas2_t1seg.tsv"), delimiter="\t"))
n_cases = len(manifest_rows)
splits = pickle.load(open(task_dir / "splits.pkl", "rb"))
fold = splits["stratified_train_val_test_split"][0.8][0]
assert len(fold["train"]) + len(fold["val"]) + len(fold["test"]) == n_cases, (n_cases, fold)
assert not (set(fold["train"]) & set(fold["val"]))
assert not (set(fold["train"]) & set(fold["test"]))
assert not (set(fold["val"]) & set(fold["test"]))
print({
    "split": "stratified_train_val_test_split[0.8]",
    "n_cases": n_cases,
    "counts": {k: len(v) for k, v in fold.items()},
})
PYEOF
fi

SPLIT_METHOD="${SPLIT_METHOD:-stratified_train_val_test_split}"
SPLIT_PARAM="${SPLIT_PARAM:-0.8}"
FOLD_IDX="${FOLD_IDX:-0}"
EPOCHS="${EPOCHS:-}"
TRAIN_BATCHES="${TRAIN_BATCHES:-}"
AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-}"
SEGMENTATION_LOSS="${SEGMENTATION_LOSS:-dicece}"
BEST_CHECKPOINT_MONITOR="${BEST_CHECKPOINT_MONITOR:-}"
BEST_CHECKPOINT_MODE="${BEST_CHECKPOINT_MODE:-}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-}"
EARLY_STOPPING_MONITOR="${EARLY_STOPPING_MONITOR:-}"
EARLY_STOPPING_MODE="${EARLY_STOPPING_MODE:-}"
CHECKPOINT_SELECT="${CHECKPOINT_SELECT:-}"
EVALUATE_BOTH_CHECKPOINTS="${EVALUATE_BOTH_CHECKPOINTS:-1}"
RUN_STYLE="${RUN_STYLE:-}"
SAVE_ROOT="${SAVE_ROOT:-/scratch/wilms_root/wilms0/jinhlee/fomo-diffusion/.atlas_single_split_runs_all955_20260528}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/turbo/umms-wilms1/FOMO/inference_outputs/task12_single_run_all955_20260528}"

case "${MODEL_NAME:-cleandift_s23}" in
  cleandift_s23)
    RUN_TAG="${RUN_TAG:-atlas955_single80_10_10_waveletavg_refine}"
    RUN_STYLE="${RUN_STYLE:-cleandift_control}"
    EPOCHS="${EPOCHS:-100}"
    TRAIN_BATCHES="${TRAIN_BATCHES:-100}"
    AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-basic}"
    BEST_CHECKPOINT_MONITOR="${BEST_CHECKPOINT_MONITOR:-val/dice}"
    BEST_CHECKPOINT_MODE="${BEST_CHECKPOINT_MODE:-max}"
    EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"
    EARLY_STOPPING_MONITOR="${EARLY_STOPPING_MONITOR:-val/dice}"
    EARLY_STOPPING_MODE="${EARLY_STOPPING_MODE:-max}"
    CHECKPOINT_SELECT="${CHECKPOINT_SELECT:-best}"
    SEG_HEAD_VARIANT="${SEG_HEAD_VARIANT:-waveletavg_refine}"
    SEG_ATTENTION="${SEG_ATTENTION:-se}"
    P_OVERSAMPLE_FOREGROUND="${P_OVERSAMPLE_FOREGROUND:-0.33}"
    PATCH_D="${PATCH_D:-192}"
    PATCH_H="${PATCH_H:-256}"
    PATCH_W="${PATCH_W:-192}"
    MIRROR_TTA="${MIRROR_TTA:-0}"
    ;;
  unet_b)
    RUN_TAG="${RUN_TAG:-atlas955_single80_10_10_unetb}"
    RUN_STYLE="${RUN_STYLE:-repo_hparams_local}"
    EPOCHS="${EPOCHS:-500}"
    TRAIN_BATCHES="${TRAIN_BATCHES:-100}"
    AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-basic}"
    BEST_CHECKPOINT_MONITOR="${BEST_CHECKPOINT_MONITOR:-val/loss}"
    BEST_CHECKPOINT_MODE="${BEST_CHECKPOINT_MODE:-min}"
    EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
    EARLY_STOPPING_MONITOR="${EARLY_STOPPING_MONITOR:-val/loss}"
    EARLY_STOPPING_MODE="${EARLY_STOPPING_MODE:-min}"
    CHECKPOINT_SELECT="${CHECKPOINT_SELECT:-last}"
    MIRROR_TTA="${MIRROR_TTA:-0}"
    ;;
  unet_xl)
    RUN_TAG="${RUN_TAG:-atlas955_single80_10_10_unetxl}"
    RUN_STYLE="${RUN_STYLE:-repo_hparams_local}"
    EPOCHS="${EPOCHS:-500}"
    TRAIN_BATCHES="${TRAIN_BATCHES:-100}"
    AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-basic}"
    BEST_CHECKPOINT_MONITOR="${BEST_CHECKPOINT_MONITOR:-val/loss}"
    BEST_CHECKPOINT_MODE="${BEST_CHECKPOINT_MODE:-min}"
    EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
    EARLY_STOPPING_MONITOR="${EARLY_STOPPING_MONITOR:-val/loss}"
    EARLY_STOPPING_MODE="${EARLY_STOPPING_MODE:-min}"
    CHECKPOINT_SELECT="${CHECKPOINT_SELECT:-last}"
    MIRROR_TTA="${MIRROR_TTA:-0}"
    ;;
  mmunetvae)
    RUN_TAG="${RUN_TAG:-atlas955_single80_10_10_mmunetvae}"
    RUN_STYLE="${RUN_STYLE:-local_wrapper}"
    EPOCHS="${EPOCHS:-100}"
    TRAIN_BATCHES="${TRAIN_BATCHES:-100}"
    AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-basic}"
    BEST_CHECKPOINT_MONITOR="${BEST_CHECKPOINT_MONITOR:-val/dice}"
    BEST_CHECKPOINT_MODE="${BEST_CHECKPOINT_MODE:-max}"
    EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"
    EARLY_STOPPING_MONITOR="${EARLY_STOPPING_MONITOR:-val/dice}"
    EARLY_STOPPING_MODE="${EARLY_STOPPING_MODE:-max}"
    CHECKPOINT_SELECT="${CHECKPOINT_SELECT:-best}"
    MIRROR_TTA="${MIRROR_TTA:-0}"
    ;;
esac

export SPLIT_METHOD SPLIT_PARAM FOLD_IDX EPOCHS TRAIN_BATCHES AUGMENTATION_PRESET
export SEGMENTATION_LOSS BEST_CHECKPOINT_MONITOR BEST_CHECKPOINT_MODE
export EARLY_STOPPING_PATIENCE EARLY_STOPPING_MONITOR EARLY_STOPPING_MODE
export CHECKPOINT_SELECT EVALUATE_BOTH_CHECKPOINTS RUN_STYLE SAVE_ROOT OUTPUT_ROOT RUN_TAG
export SEG_HEAD_VARIANT SEG_ATTENTION P_OVERSAMPLE_FOREGROUND PATCH_D PATCH_H PATCH_W MIRROR_TTA

exec bash "${REPO_ROOT}/pipelines/atlas_segmentation/slurm/run_atlas_segmentation_fullvol.sh"
