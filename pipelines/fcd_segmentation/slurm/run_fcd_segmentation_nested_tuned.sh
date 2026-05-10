#!/bin/bash
#SBATCH --job-name=ft_task10_nested_tuned
#SBATCH --account=wilms99
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
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

FINETUNE_PY="${REPO_ROOT}/src/downstream/finetune.py"
EVAL_PY="${REPO_ROOT}/pipelines/fcd_segmentation/scripts/evaluate_fcd_segmentation_tuned_fold.py"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src/downstream:${REPO_ROOT}/src/pretraining"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

DATA_ROOT="${DATA_ROOT:-/nfs/turbo/umms-wilms1/FOMO/Data/openneuro_task10_t1flairseg_1mm_20260507}"
TASK_NAME="${TASK_NAME:-Task010_OpenNeuro_ds004199_FCDSeg_T1FLAIR_1mm}"
TASK_DIR="${DATA_ROOT}/${TASK_NAME}"
SAVE_ROOT="${SAVE_ROOT:-/nfs/turbo/umms-wilms1/FOMO/benchmark_results_task10_nested_tuned_20260509}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/turbo/umms-wilms1/FOMO/inference_outputs/task10_nested_tuned_20260509}"
RUN_TAG="${RUN_TAG:-}"

MODEL_NAME="${MODEL_NAME:-cleandift_s23}"
FOLD_IDX="${FOLD_IDX:-0}"
FOLDS="${FOLDS:-5}"
SPLIT_METHOD="${SPLIT_METHOD:-nested_kfold}"
AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-all}"
EPOCHS="${EPOCHS:-100}"
TRAIN_BATCHES="${TRAIN_BATCHES:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEGMENTATION_LOSS="${SEGMENTATION_LOSS:-focaltversky}"
LINEAR_PROBING="${LINEAR_PROBING:-0}"
MIRROR_TTA="${MIRROR_TTA:-1}"

case "${MODEL_NAME}" in
  unet_b)
    PRETRAINED="${PRETRAINED:-${REPO_ROOT}/baseline_pretrained_models/unet_b.ckpt}"
    PATCH_D="${PATCH_D:-96}"; PATCH_H="${PATCH_H:-96}"; PATCH_W="${PATCH_W:-96}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    ;;
  unet_xl)
    PRETRAINED="${PRETRAINED:-${REPO_ROOT}/baseline_pretrained_models/unet_xl.ckpt}"
    PATCH_D="${PATCH_D:-96}"; PATCH_H="${PATCH_H:-96}"; PATCH_W="${PATCH_W:-96}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    ;;
  mmunetvae)
    PRETRAINED="${PRETRAINED:-/nfs/turbo/umms-wilms1/FOMO/checkpoints/fomo25_mmunetvae_pretrained.ckpt}"
    PATCH_D="${PATCH_D:-96}"; PATCH_H="${PATCH_H:-96}"; PATCH_W="${PATCH_W:-96}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    ;;
  cleandift_s23)
    PRETRAINED="${PRETRAINED:-/nfs/turbo/umms-wilms1/FOMO/experiments/jinhlee/checkpoints_scaled/fomo_s23_72h_e100/fomo_s23_72h_e100/fomo_s23_72h_e100_epoch_100.pt}"
    PATCH_D="${PATCH_D:-192}"; PATCH_H="${PATCH_H:-256}"; PATCH_W="${PATCH_W:-192}"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    ;;
  *)
    echo "[ERROR] Unsupported MODEL_NAME: ${MODEL_NAME}"
    exit 2
    ;;
esac

RUN_SUFFIX=""
if [[ -n "${RUN_TAG}" ]]; then
  RUN_SUFFIX="_${RUN_TAG}"
fi

SAVE_DIR="${SAVE_ROOT}${RUN_SUFFIX}/fold${FOLD_IDX}"
OUTPUT_JSON="${OUTPUT_ROOT}/${MODEL_NAME}${RUN_SUFFIX}_fold${FOLD_IDX}_test.json"
TIMING_JSON="${OUTPUT_ROOT}/${MODEL_NAME}${RUN_SUFFIX}_fold${FOLD_IDX}_timing.json"
mkdir -p "${OUTPUT_ROOT}"

echo "=========================================================="
echo "[$(date)] Task10 nested tuned run"
echo "Model: ${MODEL_NAME}"
echo "Fold: ${FOLD_IDX}/${FOLDS}"
echo "Split method: ${SPLIT_METHOD}"
echo "Patch: ${PATCH_D} ${PATCH_H} ${PATCH_W}"
echo "Batch size: ${BATCH_SIZE}"
echo "Augmentation: ${AUGMENTATION_PRESET}"
echo "Segmentation loss: ${SEGMENTATION_LOSS}"
echo "Linear probing: ${LINEAR_PROBING}"
echo "Mirror TTA: ${MIRROR_TTA}"
echo "Run tag: ${RUN_TAG:-<none>}"
echo "Task dir: ${TASK_DIR}"
echo "Save dir: ${SAVE_DIR}"
echo "Output JSON: ${OUTPUT_JSON}"
echo "=========================================================="

if [[ ! -d "${TASK_DIR}" ]]; then
  echo "[ERROR] Missing Task10 directory: ${TASK_DIR}"
  exit 3
fi
if [[ ! -f "${TASK_DIR}/splits.pkl" ]]; then
  echo "[ERROR] Missing splits.pkl in ${TASK_DIR}"
  exit 4
fi
if [[ ! -f "${PRETRAINED}" ]]; then
  echo "[ERROR] Missing pretrained checkpoint: ${PRETRAINED}"
  exit 5
fi

python3 << PYEOF
import pickle
from pathlib import Path

task_dir = Path("${TASK_DIR}")
splits = pickle.load(open(task_dir / "splits.pkl", "rb"))
if "${SPLIT_METHOD}" not in splits or ${FOLDS} not in splits["${SPLIT_METHOD}"]:
    raise SystemExit("Expected nested splits in splits.pkl")
folds = splits["${SPLIT_METHOD}"][${FOLDS}]
if ${FOLD_IDX} >= len(folds):
    raise SystemExit(f"Fold index out of range: ${FOLD_IDX}")
fold = folds[${FOLD_IDX}]
train_ids = fold["train"]
val_ids = fold["val"]
test_ids = fold.get("test", [])
if set(train_ids) & set(val_ids):
    raise SystemExit("Train/val overlap detected")
if set(train_ids) & set(test_ids):
    raise SystemExit("Train/test overlap detected")
if set(val_ids) & set(test_ids):
    raise SystemExit("Val/test overlap detected")
print(f"Fold {${FOLD_IDX}}: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
PYEOF

FT_START=$(date +%s)

LP_FLAG=""
if [[ "${LINEAR_PROBING}" == "1" ]]; then
  LP_FLAG="--linear_probing"
fi

python3 -u "${FINETUNE_PY}" \
  --taskid 10 \
  --data_dir "${DATA_ROOT}" \
  --save_dir "${SAVE_DIR}" \
  --pretrained_weights_path "${PRETRAINED}" \
  --model_name "${MODEL_NAME}" \
  --patch_size_dhw "${PATCH_D}" "${PATCH_H}" "${PATCH_W}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --train_batches_per_epoch "${TRAIN_BATCHES}" \
  --learning_rate "${LEARNING_RATE}" \
  --augmentation_preset "${AUGMENTATION_PRESET}" \
  --segmentation_loss "${SEGMENTATION_LOSS}" \
  --split_method "${SPLIT_METHOD}" \
  --split_param "${FOLDS}" \
  --split_idx "${FOLD_IDX}" \
  --num_workers "${NUM_WORKERS}" \
  --num_devices 1 \
  --precision bf16-mixed \
  --best_checkpoint_monitor val/dice \
  --best_checkpoint_mode max \
  --early_stopping_patience 15 \
  --early_stopping_monitor val/dice \
  --early_stopping_mode max \
  --experiment "task10_nested_${MODEL_NAME}${RUN_SUFFIX}_fold${FOLD_IDX}" \
  --new_version \
  ${LP_FLAG}

FT_END=$(date +%s)
FT_SECS=$(( FT_END - FT_START ))
echo "[$(date)] Fine-tune done: ${FT_SECS}s"

CKPT_DIR="${SAVE_DIR}/${TASK_NAME}/${MODEL_NAME}"
LATEST_VERSION_DIR=$(find "${CKPT_DIR}" -maxdepth 1 -type d -name "version_*" 2>/dev/null | sort -V | tail -1)
if [[ -z "${LATEST_VERSION_DIR}" ]]; then
  echo "[ERROR] No version_* directory found in ${CKPT_DIR}"
  exit 6
fi
CKPT=$(find "${LATEST_VERSION_DIR}/checkpoints" -maxdepth 1 -name "best.ckpt" 2>/dev/null | sort -V | tail -1)
if [[ -z "${CKPT}" ]]; then
  CKPT=$(find "${LATEST_VERSION_DIR}/checkpoints" -maxdepth 1 -name "*.ckpt" 2>/dev/null | sort -V | tail -1)
fi
if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] No checkpoint found in ${LATEST_VERSION_DIR}/checkpoints"
  exit 7
fi
echo "Checkpoint: ${CKPT}"

INF_START=$(date +%s)

MIRROR_FLAG=""
if [[ "${MIRROR_TTA}" == "1" ]]; then
  MIRROR_FLAG="--mirror"
fi

python3 "${EVAL_PY}" \
  --task_dir "${TASK_DIR}" \
  --checkpoint "${CKPT}" \
  --model_name "${MODEL_NAME}" \
  --fold_idx "${FOLD_IDX}" \
  --split_method "${SPLIT_METHOD}" \
  --split_param "${FOLDS}" \
  --patch_size_dhw "${PATCH_D}" "${PATCH_H}" "${PATCH_W}" \
  --output_json "${OUTPUT_JSON}" \
  ${MIRROR_FLAG}

INF_END=$(date +%s)
INF_SECS=$(( INF_END - INF_START ))

python3 << PYEOF
import json
payload = {
    "model_name": "${MODEL_NAME}",
    "fold_idx": int(${FOLD_IDX}),
    "checkpoint": "${CKPT}",
    "finetune_seconds": int(${FT_SECS}),
    "inference_seconds": int(${INF_SECS}),
    "split_method": "${SPLIT_METHOD}",
    "linear_probing": bool(int(${LINEAR_PROBING})),
    "mirror_tta": bool(int(${MIRROR_TTA})),
}
with open("${TIMING_JSON}", "w") as f:
    json.dump(payload, f, indent=2)
print(json.dumps(payload, indent=2))
PYEOF

echo "[$(date)] Task10 nested fold ${FOLD_IDX} complete"
