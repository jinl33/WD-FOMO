#!/bin/bash
#SBATCH --job-name=atlas_eval
#SBATCH --account=wilms99
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # 2026-09-06 FIX: spgpu nodes are 32 CPU / 8 GPU = 4 CPU/GPU; requesting 8 for a
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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

EVAL_PY="${REPO_ROOT}/pipelines/atlas_segmentation/scripts/evaluate_atlas_segmentation_fold.py"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src/downstream:${REPO_ROOT}/src/pretraining"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

DATA_ROOT="${DATA_ROOT:-/nfs/turbo/umms-wilms1/FOMO/Data/atlas2_stroke_t1seg_1mm_20260522}"
TASK_NAME="${TASK_NAME:-Task012_ATLAS2_StrokeLesion_T1_1mm}"
TASK_DIR="${DATA_ROOT}/${TASK_NAME}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME is required}"
FOLD_IDX="${FOLD_IDX:?FOLD_IDX is required}"
SPLIT_METHOD="${SPLIT_METHOD:-stratified_train_val_test_split}"
SPLIT_PARAM="${SPLIT_PARAM:-0.8}"
PATCH_D="${PATCH_D:?PATCH_D is required}"
PATCH_H="${PATCH_H:?PATCH_H is required}"
PATCH_W="${PATCH_W:?PATCH_W is required}"
OUTPUT_JSON="${OUTPUT_JSON:?OUTPUT_JSON is required}"
SEG_HEAD_VARIANT="${SEG_HEAD_VARIANT:-earlyfusion}"
MIRROR_TTA="${MIRROR_TTA:-1}"
ARGMAX_NO_TUNE="${ARGMAX_NO_TUNE:-1}"
FIXED_THRESHOLD="${FIXED_THRESHOLD:-}"
FIXED_MIN_COMPONENT_VOXELS="${FIXED_MIN_COMPONENT_VOXELS:-}"
FIXED_KEEP_LARGEST="${FIXED_KEEP_LARGEST:-}"
RESTRICT_TO_IMAGE_FOREGROUND="${RESTRICT_TO_IMAGE_FOREGROUND:-0}"

echo "=========================================================="
echo "[$(date)] ATLAS evaluation-only run"
echo "Model: ${MODEL_NAME}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Split index: ${FOLD_IDX}"
echo "Patch size: ${PATCH_D} ${PATCH_H} ${PATCH_W}"
echo "Seg head variant: ${SEG_HEAD_VARIANT}"
echo "Task dir: ${TASK_DIR}"
echo "Output JSON: ${OUTPUT_JSON}"
echo "=========================================================="

if [[ ! -d "${TASK_DIR}" ]]; then
  echo "[ERROR] Missing task directory: ${TASK_DIR}"
  exit 3
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[ERROR] Missing checkpoint: ${CHECKPOINT}"
  exit 4
fi

mkdir -p "$(dirname "${OUTPUT_JSON}")"

MIRROR_FLAG=()
if [[ "${MIRROR_TTA}" == "1" ]]; then
  MIRROR_FLAG=(--mirror)
fi

ARGMAX_FLAG=()
if [[ "${ARGMAX_NO_TUNE}" == "1" ]]; then
  ARGMAX_FLAG=(--argmax_no_tune)
fi

IMAGE_FG_FLAG=()
if [[ "${RESTRICT_TO_IMAGE_FOREGROUND}" == "1" ]]; then
  IMAGE_FG_FLAG=(--restrict_to_image_foreground)
fi

FIXED_FLAGS=()
if [[ -n "${FIXED_THRESHOLD}" ]]; then
  : "${FIXED_MIN_COMPONENT_VOXELS:?FIXED_MIN_COMPONENT_VOXELS is required with FIXED_THRESHOLD}"
  : "${FIXED_KEEP_LARGEST:?FIXED_KEEP_LARGEST is required with FIXED_THRESHOLD}"
  FIXED_FLAGS=(
    --fixed_threshold "${FIXED_THRESHOLD}"
    --fixed_min_component_voxels "${FIXED_MIN_COMPONENT_VOXELS}"
    --fixed_keep_largest "${FIXED_KEEP_LARGEST}"
  )
fi

python3 -u "${EVAL_PY}" \
  --task_dir "${TASK_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --model_name "${MODEL_NAME}" \
  --seg_head_variant "${SEG_HEAD_VARIANT}" \
  --fold_idx "${FOLD_IDX}" \
  --split_method "${SPLIT_METHOD}" \
  --split_param "${SPLIT_PARAM}" \
  --patch_size_dhw "${PATCH_D}" "${PATCH_H}" "${PATCH_W}" \
  --output_json "${OUTPUT_JSON}" \
  "${MIRROR_FLAG[@]}" \
  "${IMAGE_FG_FLAG[@]}" \
  "${ARGMAX_FLAG[@]}" \
  "${FIXED_FLAGS[@]}"

echo "[$(date)] Evaluation-only run complete"
