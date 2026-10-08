#!/bin/bash
# Evaluate an existing FCD segmentation checkpoint.

#SBATCH --job-name=fcdseg_eval_argmax
#SBATCH --account=wilms99
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/turbo/umms-wilms1/FOMO/experiments/jinhlee/logs/%x-%j.log

set -euo pipefail

if [[ -n "${LIGHTWEIGHT_FOMO_ROOT:-}" ]]; then
  REPO_ROOT="${LIGHTWEIGHT_FOMO_ROOT}"
elif [[ -f "/scratch/wilms_root/wilms0/jinhlee/fomo-diffusion/lightweight-fomo/src/downstream/finetune.py" ]]; then
  REPO_ROOT="/scratch/wilms_root/wilms0/jinhlee/fomo-diffusion/lightweight-fomo"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/src/downstream/finetune.py" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi

module load python/3.10.4
source "${FOMO_ENV:-/scratch/wilms_root/wilms0/jinhlee/fomo-diffusion/fomo_env/bin/activate}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src/downstream:${REPO_ROOT}/src/pretraining"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

DATA_ROOT="${DATA_ROOT:-/nfs/turbo/umms-wilms1/FOMO/Data/openneuro_task10_t1flairseg_1mm_20260507}"
TASK_NAME="${TASK_NAME:-Task010_OpenNeuro_ds004199_FCDSeg_T1FLAIR_1mm}"
TASK_DIR="${DATA_ROOT}/${TASK_NAME}"
EVAL_PY="${REPO_ROOT}/pipelines/fcd_segmentation/scripts/evaluate_fcd_segmentation_tuned_fold.py"

: "${MODEL_NAME:?MODEL_NAME is required}"
: "${FOLD_IDX:?FOLD_IDX is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${OUTPUT_JSON:?OUTPUT_JSON is required}"

SPLIT_METHOD="${SPLIT_METHOD:-nested_balanced_qc76_kfold}"
FOLDS="${FOLDS:-5}"
SEG_HEAD_VARIANT="${SEG_HEAD_VARIANT:-earlyfusion}"
PATCH_D="${PATCH_D:-96}"
PATCH_H="${PATCH_H:-96}"
PATCH_W="${PATCH_W:-96}"
INPUT_MODALITY_INDICES="${INPUT_MODALITY_INDICES:-}"
EVAL_MODE="${EVAL_MODE:-argmax_no_tune}"
MIRROR_TTA="${MIRROR_TTA:-0}"
ALLOW_BACKGROUND_PREDICTIONS="${ALLOW_BACKGROUND_PREDICTIONS:-0}"
LEGACY_AXIS_TRANSPOSE="${LEGACY_AXIS_TRANSPOSE:-0}"
FIXED_THRESHOLD="${FIXED_THRESHOLD:-}"
FIXED_MIN_COMPONENT_VOXELS="${FIXED_MIN_COMPONENT_VOXELS:-}"
FIXED_KEEP_LARGEST="${FIXED_KEEP_LARGEST:-}"

INPUT_MODALITY_FLAG=()
if [[ -n "${INPUT_MODALITY_INDICES}" ]]; then
  read -r -a INPUT_MODALITY_ARRAY <<< "${INPUT_MODALITY_INDICES//,/ }"
  INPUT_MODALITY_FLAG=(--input_modality_indices "${INPUT_MODALITY_ARRAY[@]}")
fi

echo "=========================================================="
echo "[$(date)] FCD segmentation existing-checkpoint argmax eval"
echo "Model: ${MODEL_NAME}"
echo "Fold: ${FOLD_IDX}/${FOLDS}"
echo "Split method: ${SPLIT_METHOD}"
echo "Patch: ${PATCH_D} ${PATCH_H} ${PATCH_W}"
echo "Seg head variant: ${SEG_HEAD_VARIANT}"
echo "Input modality indices: ${INPUT_MODALITY_INDICES:-<all>}"
echo "Evaluation mode: ${EVAL_MODE}"
echo "Mirror TTA: ${MIRROR_TTA}"
echo "Allow background predictions: ${ALLOW_BACKGROUND_PREDICTIONS}"
echo "Legacy axis transpose: ${LEGACY_AXIS_TRANSPOSE}"
echo "Fixed threshold: ${FIXED_THRESHOLD:-<none>}"
echo "Fixed min component voxels: ${FIXED_MIN_COMPONENT_VOXELS:-<none>}"
echo "Fixed keep largest: ${FIXED_KEEP_LARGEST:-<none>}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output JSON: ${OUTPUT_JSON}"
echo "=========================================================="

EVAL_MODE_FLAG=()
case "${EVAL_MODE}" in
  tuned)
    EVAL_MODE_FLAG=()
    ;;
  argmax_no_tune)
    EVAL_MODE_FLAG=(--argmax_no_tune)
    ;;
  fixed_postprocess)
    : "${FIXED_THRESHOLD:?FIXED_THRESHOLD is required when EVAL_MODE=fixed_postprocess}"
    : "${FIXED_MIN_COMPONENT_VOXELS:?FIXED_MIN_COMPONENT_VOXELS is required when EVAL_MODE=fixed_postprocess}"
    : "${FIXED_KEEP_LARGEST:?FIXED_KEEP_LARGEST is required when EVAL_MODE=fixed_postprocess}"
    EVAL_MODE_FLAG=(
      --fixed_threshold "${FIXED_THRESHOLD}"
      --fixed_min_component_voxels "${FIXED_MIN_COMPONENT_VOXELS}"
      --fixed_keep_largest "${FIXED_KEEP_LARGEST}"
    )
    ;;
  *)
    echo "[ERROR] Unsupported EVAL_MODE: ${EVAL_MODE}"
    exit 9
    ;;
esac

MIRROR_FLAG=()
if [[ "${MIRROR_TTA}" == "1" ]]; then
  MIRROR_FLAG=(--mirror)
fi

BACKGROUND_FLAG=()
if [[ "${ALLOW_BACKGROUND_PREDICTIONS}" == "1" ]]; then
  BACKGROUND_FLAG=(--allow_background_predictions)
fi

LEGACY_AXIS_FLAG=()
if [[ "${LEGACY_AXIS_TRANSPOSE}" == "1" ]]; then
  LEGACY_AXIS_FLAG=(--legacy_axis_transpose)
fi

python3 "${EVAL_PY}" \
  --task_dir "${TASK_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --model_name "${MODEL_NAME}" \
  --seg_head_variant "${SEG_HEAD_VARIANT}" \
  "${INPUT_MODALITY_FLAG[@]}" \
  --fold_idx "${FOLD_IDX}" \
  --split_method "${SPLIT_METHOD}" \
  --split_param "${FOLDS}" \
  --patch_size_dhw "${PATCH_D}" "${PATCH_H}" "${PATCH_W}" \
  --output_json "${OUTPUT_JSON}" \
  "${EVAL_MODE_FLAG[@]}" \
  "${MIRROR_FLAG[@]}" \
  "${BACKGROUND_FLAG[@]}" \
  "${LEGACY_AXIS_FLAG[@]}"

echo "[$(date)] Done"
