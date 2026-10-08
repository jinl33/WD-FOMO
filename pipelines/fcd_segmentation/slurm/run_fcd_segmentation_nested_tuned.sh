#!/bin/bash
#SBATCH --job-name=ft_task10_nested_tuned
#SBATCH --account=wilms99
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # 2026-09-06 FIX: spgpu nodes are 32 CPU / 8 GPU = 4 CPU/GPU; requesting 8 for a
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
TASK_ID="${TASK_ID:-10}"
TASK_DIR="${DATA_ROOT}/${TASK_NAME}"
SAVE_ROOT="${SAVE_ROOT:-/nfs/turbo/umms-wilms1/FOMO/benchmark_results_task10_nested_tuned_20260509}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/turbo/umms-wilms1/FOMO/inference_outputs/task10_nested_tuned_20260509}"
RUN_TAG="${RUN_TAG:-}"
BASELINE_CKPT_ROOT="${BASELINE_CKPT_ROOT:-${REPO_ROOT}/baseline_pretrained_models}"
if [[ ! -d "${BASELINE_CKPT_ROOT}" && -d "${REPO_ROOT}/../baseline_pretrained_models" ]]; then
  BASELINE_CKPT_ROOT="$(cd "${REPO_ROOT}/../baseline_pretrained_models" && pwd)"
fi

MODEL_NAME="${MODEL_NAME:-cleandift_s23}"
FOLD_IDX="${FOLD_IDX:-0}"
FOLDS="${FOLDS:-5}"
SPLIT_METHOD="${SPLIT_METHOD:-nested_kfold}"
AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-all}"
EPOCHS="${EPOCHS:-100}"
TRAIN_BATCHES="${TRAIN_BATCHES:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-0.0}"
BEST_CHECKPOINT_MONITOR="${BEST_CHECKPOINT_MONITOR:-val/dice}"
BEST_CHECKPOINT_MODE="${BEST_CHECKPOINT_MODE:-max}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"
EARLY_STOPPING_MONITOR="${EARLY_STOPPING_MONITOR:-${BEST_CHECKPOINT_MONITOR}}"
EARLY_STOPPING_MODE="${EARLY_STOPPING_MODE:-${BEST_CHECKPOINT_MODE}}"
CHECKPOINT_SELECT="${CHECKPOINT_SELECT:-best}"
EVAL_MODE="${EVAL_MODE:-tuned}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEGMENTATION_LOSS="${SEGMENTATION_LOSS:-focaltversky}"
SEG_ATTENTION="${SEG_ATTENTION:-none}"
SEG_HEAD_VARIANT="${SEG_HEAD_VARIANT:-earlyfusion}"
SEG_OUTPUT_FG_PRIOR="${SEG_OUTPUT_FG_PRIOR:-}"
INPUT_MODALITY_INDICES="${INPUT_MODALITY_INDICES:-}"
P_OVERSAMPLE_FOREGROUND="${P_OVERSAMPLE_FOREGROUND:-0.33}"
LINEAR_PROBING="${LINEAR_PROBING:-0}"
MIRROR_TTA="${MIRROR_TTA:-1}"

case "${MODEL_NAME}" in
  unet_b)
    PRETRAINED="${PRETRAINED:-${BASELINE_CKPT_ROOT}/unet_b.ckpt}"
    PATCH_D="${PATCH_D:-96}"; PATCH_H="${PATCH_H:-96}"; PATCH_W="${PATCH_W:-96}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    ;;
  unet_xl)
    PRETRAINED="${PRETRAINED:-${BASELINE_CKPT_ROOT}/unet_xl.ckpt}"
    PATCH_D="${PATCH_D:-96}"; PATCH_H="${PATCH_H:-96}"; PATCH_W="${PATCH_W:-96}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    ;;
  mmunetvae)
    PRETRAINED="${PRETRAINED:-/nfs/turbo/umms-wilms1/FOMO/checkpoints/fomo25_mmunetvae_pretrained.ckpt}"
    PATCH_D="${PATCH_D:-64}"; PATCH_H="${PATCH_H:-64}"; PATCH_W="${PATCH_W:-64}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    ;;
  cleandift_s23)
    PRETRAINED="${PRETRAINED:-/nfs/turbo/umms-wilms1/FOMO/experiments/jinhlee/checkpoints_scaled/fomo_s23_72h_e100/fomo_s23_72h_e100/fomo_s23_72h_e100_epoch_100.pt}"
    PATCH_D="${PATCH_D:-192}"; PATCH_H="${PATCH_H:-256}"; PATCH_W="${PATCH_W:-192}"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    ;;
  cleandift_s23_hc16)
    PRETRAINED="${PRETRAINED:-/nfs/turbo/umms-wilms1/FOMO/experiments/jinhlee/checkpoints_scaled/fomo_s23_72h_e100/fomo_s23_72h_e100/fomo_s23_72h_e100_epoch_100.pt}"
    PATCH_D="${PATCH_D:-192}"; PATCH_H="${PATCH_H:-256}"; PATCH_W="${PATCH_W:-192}"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    ;;
  cleandift_s23_hc16_notime|cleandift_s23_notime)
    : "${PRETRAINED:?cleandift_s23_*notime requires an explicit PRETRAINED (masked-reconstruction checkpoint)}"
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
echo "[$(date)] Task${TASK_ID} nested tuned run"
echo "Model: ${MODEL_NAME}"
echo "Fold: ${FOLD_IDX}/${FOLDS}"
echo "Split method: ${SPLIT_METHOD}"
echo "Patch: ${PATCH_D} ${PATCH_H} ${PATCH_W}"
echo "Batch size: ${BATCH_SIZE}"
echo "Accumulate grad batches: ${ACCUMULATE_GRAD_BATCHES}"
echo "Effective batch size: $(( BATCH_SIZE * ACCUMULATE_GRAD_BATCHES ))"
echo "Learning rate: ${LEARNING_RATE}"
echo "Backbone learning rate: ${BACKBONE_LEARNING_RATE:-<same as learning rate>}"
echo "Gradient clip value: ${GRADIENT_CLIP_VAL}"
echo "Best checkpoint monitor: ${BEST_CHECKPOINT_MONITOR} (${BEST_CHECKPOINT_MODE})"
echo "Early stopping monitor: ${EARLY_STOPPING_MONITOR} (${EARLY_STOPPING_MODE}, patience ${EARLY_STOPPING_PATIENCE})"
echo "Checkpoint select: ${CHECKPOINT_SELECT}"
echo "Evaluation mode: ${EVAL_MODE}"
echo "Augmentation: ${AUGMENTATION_PRESET}"
echo "Segmentation loss: ${SEGMENTATION_LOSS}"
echo "Seg attention: ${SEG_ATTENTION}"
echo "Seg head variant: ${SEG_HEAD_VARIANT}"
echo "Seg output fg prior: ${SEG_OUTPUT_FG_PRIOR:-<none>}"
echo "Input modality indices: ${INPUT_MODALITY_INDICES:-<all>}"
echo "Foreground oversampling: ${P_OVERSAMPLE_FOREGROUND}"
echo "Linear probing: ${LINEAR_PROBING}"
echo "Mirror TTA: ${MIRROR_TTA}"
echo "Run tag: ${RUN_TAG:-<none>}"
echo "Task dir: ${TASK_DIR}"
echo "Save dir: ${SAVE_DIR}"
echo "Output JSON: ${OUTPUT_JSON}"
echo "=========================================================="

if [[ ! -d "${TASK_DIR}" ]]; then
  echo "[ERROR] Missing Task${TASK_ID} directory: ${TASK_DIR}"
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

echo "[$(date)] Starting finetune for fold ${FOLD_IDX}..."
FT_START=$(date +%s)

LP_FLAG=""
if [[ "${LINEAR_PROBING}" == "1" ]]; then
  LP_FLAG="--linear_probing"
fi

BACKBONE_LR_FLAG=()
if [[ -n "${BACKBONE_LEARNING_RATE}" ]]; then
  BACKBONE_LR_FLAG=(--backbone_learning_rate "${BACKBONE_LEARNING_RATE}")
fi

INPUT_MODALITY_FLAG=()
if [[ -n "${INPUT_MODALITY_INDICES}" ]]; then
  read -r -a INPUT_MODALITY_ARRAY <<< "${INPUT_MODALITY_INDICES//,/ }"
  INPUT_MODALITY_FLAG=(--input_modality_indices "${INPUT_MODALITY_ARRAY[@]}")
fi

FINETUNE_PATCH_FLAG=()
if [[ "${PATCH_D}" == "${PATCH_H}" && "${PATCH_D}" == "${PATCH_W}" ]]; then
  FINETUNE_PATCH_FLAG=(--patch_size "${PATCH_D}")
else
  FINETUNE_PATCH_FLAG=(--patch_size_dhw "${PATCH_D}" "${PATCH_H}" "${PATCH_W}")
fi

SEG_OUTPUT_PRIOR_FLAG=()
if [[ -n "${SEG_OUTPUT_FG_PRIOR}" ]]; then
  SEG_OUTPUT_PRIOR_FLAG=(--seg_output_fg_prior "${SEG_OUTPUT_FG_PRIOR}")
fi

python3 -u "${FINETUNE_PY}" \
  --taskid "${TASK_ID}" \
  --data_dir "${DATA_ROOT}" \
  --save_dir "${SAVE_DIR}" \
  --pretrained_weights_path "${PRETRAINED}" \
  --model_name "${MODEL_NAME}" \
  "${FINETUNE_PATCH_FLAG[@]}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --train_batches_per_epoch "${TRAIN_BATCHES}" \
  --learning_rate "${LEARNING_RATE}" \
  "${BACKBONE_LR_FLAG[@]}" \
  --accumulate_grad_batches "${ACCUMULATE_GRAD_BATCHES}" \
  --augmentation_preset "${AUGMENTATION_PRESET}" \
  --segmentation_loss "${SEGMENTATION_LOSS}" \
  --seg_attention "${SEG_ATTENTION}" \
  --seg_head_variant "${SEG_HEAD_VARIANT}" \
  "${SEG_OUTPUT_PRIOR_FLAG[@]}" \
  "${INPUT_MODALITY_FLAG[@]}" \
  --p_oversample_foreground "${P_OVERSAMPLE_FOREGROUND}" \
  --split_method "${SPLIT_METHOD}" \
  --split_param "${FOLDS}" \
  --split_idx "${FOLD_IDX}" \
  --num_workers "${NUM_WORKERS}" \
  --num_devices 1 \
  --precision bf16-mixed \
  --gradient_clip_val "${GRADIENT_CLIP_VAL}" \
  --best_checkpoint_monitor "${BEST_CHECKPOINT_MONITOR}" \
  --best_checkpoint_mode "${BEST_CHECKPOINT_MODE}" \
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
  --early_stopping_monitor "${EARLY_STOPPING_MONITOR}" \
  --early_stopping_mode "${EARLY_STOPPING_MODE}" \
  --experiment "task${TASK_ID}_nested_${MODEL_NAME}${RUN_SUFFIX}_fold${FOLD_IDX}" \
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
case "${CHECKPOINT_SELECT}" in
  best)
    CKPT=$(find "${LATEST_VERSION_DIR}/checkpoints" -maxdepth 1 -name "best.ckpt" 2>/dev/null | sort -V | tail -1)
    if [[ -z "${CKPT}" ]]; then
      CKPT=$(find "${LATEST_VERSION_DIR}/checkpoints" -maxdepth 1 -name "*.ckpt" 2>/dev/null | sort -V | tail -1)
    fi
    ;;
  last)
    CKPT=$(find "${LATEST_VERSION_DIR}/checkpoints" -maxdepth 1 -name "last.ckpt" 2>/dev/null | sort -V | tail -1)
    if [[ -z "${CKPT}" ]]; then
      CKPT=$(find "${LATEST_VERSION_DIR}/checkpoints" -maxdepth 1 -name "*.ckpt" 2>/dev/null | sort -V | tail -1)
    fi
    ;;
  *)
    echo "[ERROR] Unsupported CHECKPOINT_SELECT: ${CHECKPOINT_SELECT}"
    exit 8
    ;;
esac
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

EVAL_MODE_FLAG=()
case "${EVAL_MODE}" in
  tuned)
    EVAL_MODE_FLAG=()
    ;;
  argmax_no_tune)
    EVAL_MODE_FLAG=(--argmax_no_tune)
    ;;
  *)
    echo "[ERROR] Unsupported EVAL_MODE: ${EVAL_MODE}"
    exit 9
    ;;
esac

python3 "${EVAL_PY}" \
  --task_dir "${TASK_DIR}" \
  --checkpoint "${CKPT}" \
  --model_name "${MODEL_NAME}" \
  --seg_head_variant "${SEG_HEAD_VARIANT}" \
  "${INPUT_MODALITY_FLAG[@]}" \
  --fold_idx "${FOLD_IDX}" \
  --split_method "${SPLIT_METHOD}" \
  --split_param "${FOLDS}" \
  --patch_size_dhw "${PATCH_D}" "${PATCH_H}" "${PATCH_W}" \
  --output_json "${OUTPUT_JSON}" \
  "${EVAL_MODE_FLAG[@]}" \
  ${MIRROR_FLAG}

INF_END=$(date +%s)
INF_SECS=$(( INF_END - INF_START ))

python3 << PYEOF
import json
payload = {
    "model_name": "${MODEL_NAME}",
    "seg_head_variant": "${SEG_HEAD_VARIANT}",
    "input_modality_indices": None if "${INPUT_MODALITY_INDICES}" == "" else "${INPUT_MODALITY_INDICES}",
    "fold_idx": int(${FOLD_IDX}),
    "checkpoint": "${CKPT}",
    "learning_rate": float(${LEARNING_RATE}),
    "backbone_learning_rate": None if "${BACKBONE_LEARNING_RATE}" == "" else float("${BACKBONE_LEARNING_RATE}"),
    "accumulate_grad_batches": int(${ACCUMULATE_GRAD_BATCHES}),
    "gradient_clip_val": float(${GRADIENT_CLIP_VAL}),
    "best_checkpoint_monitor": "${BEST_CHECKPOINT_MONITOR}",
    "best_checkpoint_mode": "${BEST_CHECKPOINT_MODE}",
    "early_stopping_monitor": "${EARLY_STOPPING_MONITOR}",
    "early_stopping_mode": "${EARLY_STOPPING_MODE}",
    "early_stopping_patience": int(${EARLY_STOPPING_PATIENCE}),
    "checkpoint_select": "${CHECKPOINT_SELECT}",
    "evaluation_mode": "${EVAL_MODE}",
    "augmentation": "${AUGMENTATION_PRESET}",
    "epochs": int(${EPOCHS}),
    "train_batches_per_epoch": int(${TRAIN_BATCHES}),
    "batch_size": int(${BATCH_SIZE}),
    "patch_size": [int(${PATCH_D}), int(${PATCH_H}), int(${PATCH_W})],
    "segmentation_loss": "${SEGMENTATION_LOSS}",
    "seg_attention": "${SEG_ATTENTION}",
    "seg_output_fg_prior": None if "${SEG_OUTPUT_FG_PRIOR}" == "" else float("${SEG_OUTPUT_FG_PRIOR}"),
    "p_oversample_foreground": float(${P_OVERSAMPLE_FOREGROUND}),
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
