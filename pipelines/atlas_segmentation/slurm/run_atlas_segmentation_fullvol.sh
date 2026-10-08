#!/bin/bash
#SBATCH --job-name=atlas_task12
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

FINETUNE_PY="${REPO_ROOT}/src/downstream/finetune.py"
BASELINE_REPO_ROOT="${BASELINE_REPO_ROOT:-${REPO_ROOT}/../baseline-codebase}"
if [[ ! -d "${BASELINE_REPO_ROOT}" && -d "${REPO_ROOT}/baseline-codebase" ]]; then
  BASELINE_REPO_ROOT="${REPO_ROOT}/baseline-codebase"
fi
BASELINE_FINETUNE_PY="${BASELINE_REPO_ROOT}/src/finetune.py"
EVAL_PY="${REPO_ROOT}/pipelines/atlas_segmentation/scripts/evaluate_atlas_segmentation_fold.py"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src/downstream:${REPO_ROOT}/src/pretraining"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

DATA_ROOT="${DATA_ROOT:-/nfs/turbo/umms-wilms1/FOMO/Data/atlas2_stroke_t1seg_1mm_20260522}"
TASK_NAME="${TASK_NAME:-Task012_ATLAS2_StrokeLesion_T1_1mm}"
TASK_ID="${TASK_ID:-12}"
TASK_DIR="${DATA_ROOT}/${TASK_NAME}"
SAVE_ROOT="${SAVE_ROOT:-/scratch/wilms_root/wilms0/jinhlee/fomo-diffusion/.atlas_single_split_runs_all955_20260528}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/turbo/umms-wilms1/FOMO/inference_outputs/task12_single_run_all955_20260528}"
RUN_TAG="${RUN_TAG:-}"
BASELINE_CKPT_ROOT="${BASELINE_CKPT_ROOT:-${REPO_ROOT}/baseline_pretrained_models}"
if [[ ! -d "${BASELINE_CKPT_ROOT}" && -d "${REPO_ROOT}/../baseline_pretrained_models" ]]; then
  BASELINE_CKPT_ROOT="$(cd "${REPO_ROOT}/../baseline_pretrained_models" && pwd)"
fi

MODEL_NAME="${MODEL_NAME:-cleandift_s23}"
RUN_STYLE="${RUN_STYLE:-local_wrapper}"
FOLD_IDX="${FOLD_IDX:-0}"
SPLIT_METHOD="${SPLIT_METHOD:-stratified_train_val_test_split}"
SPLIT_PARAM="${SPLIT_PARAM:-0.8}"
AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-basic}"
EPOCHS="${EPOCHS:-100}"
TRAIN_BATCHES="${TRAIN_BATCHES:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BACKBONE_LEARNING_RATE="${BACKBONE_LEARNING_RATE:-}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-0.0}"
BEST_CHECKPOINT_MONITOR="${BEST_CHECKPOINT_MONITOR:-}"
BEST_CHECKPOINT_MODE="${BEST_CHECKPOINT_MODE:-}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-}"
EARLY_STOPPING_MONITOR="${EARLY_STOPPING_MONITOR:-}"
EARLY_STOPPING_MODE="${EARLY_STOPPING_MODE:-}"
CHECKPOINT_SELECT="${CHECKPOINT_SELECT:-}"
EVAL_MODE="${EVAL_MODE:-argmax_no_tune}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEGMENTATION_LOSS="${SEGMENTATION_LOSS:-dicece}"
SEG_ATTENTION="${SEG_ATTENTION:-se}"
SEG_HEAD_VARIANT="${SEG_HEAD_VARIANT:-earlyfusion}"
SEG_OUTPUT_FG_PRIOR="${SEG_OUTPUT_FG_PRIOR:-}"
SEG_AUX_LOSS_WEIGHT="${SEG_AUX_LOSS_WEIGHT:-0.0}"
SEG_AUX_TARGET="${SEG_AUX_TARGET:-binary_band}"
SEG_BOUNDARY_RADIUS="${SEG_BOUNDARY_RADIUS:-1}"
SEG_SMALL_LESION_WEIGHT="${SEG_SMALL_LESION_WEIGHT:-0.0}"
SEG_SMALL_THRESHOLDS="${SEG_SMALL_THRESHOLDS:-1000 10000 50000}"
P_OVERSAMPLE_FOREGROUND="${P_OVERSAMPLE_FOREGROUND:-0.33}"
LINEAR_PROBING="${LINEAR_PROBING:-0}"
MIRROR_TTA="${MIRROR_TTA:-1}"
FIXED_THRESHOLD="${FIXED_THRESHOLD:-}"
FIXED_MIN_COMPONENT_VOXELS="${FIXED_MIN_COMPONENT_VOXELS:-}"
FIXED_KEEP_LARGEST="${FIXED_KEEP_LARGEST:-}"
RESTRICT_TO_IMAGE_FOREGROUND="${RESTRICT_TO_IMAGE_FOREGROUND:-0}"
EVALUATE_BOTH_CHECKPOINTS="${EVALUATE_BOTH_CHECKPOINTS:-0}"

case "${MODEL_NAME}" in
  unet_b)
    PRETRAINED="${PRETRAINED:-${BASELINE_CKPT_ROOT}/unet_b.ckpt}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    PATCH_D="${PATCH_D:-96}"
    PATCH_H="${PATCH_H:-96}"
    PATCH_W="${PATCH_W:-96}"
    ;;
  unet_xl)
    PRETRAINED="${PRETRAINED:-${BASELINE_CKPT_ROOT}/unet_xl.ckpt}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    PATCH_D="${PATCH_D:-96}"
    PATCH_H="${PATCH_H:-96}"
    PATCH_W="${PATCH_W:-96}"
    ;;
  mmunetvae)
    PRETRAINED="${PRETRAINED:-/nfs/turbo/umms-wilms1/FOMO/checkpoints/fomo25_mmunetvae_pretrained.ckpt}"
    BATCH_SIZE="${BATCH_SIZE:-1}"
    PATCH_D="${PATCH_D:-96}"
    PATCH_H="${PATCH_H:-96}"
    PATCH_W="${PATCH_W:-96}"
    ;;
  cleandift_s23)
    PRETRAINED="${PRETRAINED:-/nfs/turbo/umms-wilms1/FOMO/experiments/jinhlee/checkpoints_scaled/fomo_s23_72h_e100/fomo_s23_72h_e100/fomo_s23_72h_e100_epoch_100.pt}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    PATCH_D="${PATCH_D:-196}"
    PATCH_H="${PATCH_H:-256}"
    PATCH_W="${PATCH_W:-196}"
    ;;
  cleandift_s23_hc16)
    PRETRAINED="${PRETRAINED:-/nfs/turbo/umms-wilms1/FOMO/experiments/jinhlee/checkpoints_scaled/fomo_s23_72h_e100/fomo_s23_72h_e100/fomo_s23_72h_e100_epoch_100.pt}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    PATCH_D="${PATCH_D:-196}"
    PATCH_H="${PATCH_H:-256}"
    PATCH_W="${PATCH_W:-196}"
    ;;
  cleandift_s23_hc16_notime|cleandift_s23_notime)
    : "${PRETRAINED:?cleandift_s23_*notime requires an explicit PRETRAINED (masked-reconstruction checkpoint)}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    PATCH_D="${PATCH_D:-196}"
    PATCH_H="${PATCH_H:-256}"
    PATCH_W="${PATCH_W:-196}"
    ;;
  *)
    echo "[ERROR] Unsupported MODEL_NAME: ${MODEL_NAME}"
    exit 2
    ;;
esac

case "${RUN_STYLE}" in
  repo_faithful)
    CHECKPOINT_SELECT="${CHECKPOINT_SELECT:-last}"
    BEST_CHECKPOINT_MONITOR="${BEST_CHECKPOINT_MONITOR:-val/loss}"
    BEST_CHECKPOINT_MODE="${BEST_CHECKPOINT_MODE:-min}"
    EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
    ;;
  repo_hparams_local)
    CHECKPOINT_SELECT="${CHECKPOINT_SELECT:-last}"
    BEST_CHECKPOINT_MONITOR="${BEST_CHECKPOINT_MONITOR:-val/loss}"
    BEST_CHECKPOINT_MODE="${BEST_CHECKPOINT_MODE:-min}"
    EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-0}"
    ;;
  local_wrapper|cleandift_control|cleandift_explore)
    CHECKPOINT_SELECT="${CHECKPOINT_SELECT:-best}"
    BEST_CHECKPOINT_MONITOR="${BEST_CHECKPOINT_MONITOR:-val/dice}"
    BEST_CHECKPOINT_MODE="${BEST_CHECKPOINT_MODE:-max}"
    EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"
    ;;
  *)
    echo "[ERROR] Unsupported RUN_STYLE: ${RUN_STYLE}"
    exit 2
    ;;
esac

if [[ -z "${EARLY_STOPPING_MONITOR}" ]]; then
  EARLY_STOPPING_MONITOR="${BEST_CHECKPOINT_MONITOR}"
fi
if [[ -z "${EARLY_STOPPING_MODE}" ]]; then
  EARLY_STOPPING_MODE="${BEST_CHECKPOINT_MODE}"
fi

RUN_SUFFIX=""
if [[ -n "${RUN_TAG}" ]]; then
  RUN_SUFFIX="_${RUN_TAG}"
fi

SAVE_DIR="${SAVE_ROOT}${RUN_SUFFIX}/fold${FOLD_IDX}"
OUTPUT_JSON="${OUTPUT_ROOT}/${MODEL_NAME}${RUN_SUFFIX}_fold${FOLD_IDX}_test.json"
TIMING_JSON="${OUTPUT_ROOT}/${MODEL_NAME}${RUN_SUFFIX}_fold${FOLD_IDX}_timing.json"
mkdir -p "${OUTPUT_ROOT}"

echo "=========================================================="
echo "[$(date)] Task${TASK_ID} full-volume ATLAS run"
echo "Model: ${MODEL_NAME}"
echo "Run style: ${RUN_STYLE}"
echo "Split index: ${FOLD_IDX}"
echo "Split method: ${SPLIT_METHOD}[${SPLIT_PARAM}]"
echo "Patch/full volume: ${PATCH_D} ${PATCH_H} ${PATCH_W}"
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
echo "Seg aux loss weight: ${SEG_AUX_LOSS_WEIGHT}"
echo "Seg aux target: ${SEG_AUX_TARGET}"
echo "Seg boundary radius: ${SEG_BOUNDARY_RADIUS}"
echo "Seg small lesion weight: ${SEG_SMALL_LESION_WEIGHT}"
echo "Seg small lesion thresholds: ${SEG_SMALL_THRESHOLDS}"
echo "Foreground oversampling: ${P_OVERSAMPLE_FOREGROUND}"
echo "Linear probing: ${LINEAR_PROBING}"
echo "Mirror TTA: ${MIRROR_TTA}"
echo "Restrict to image foreground: ${RESTRICT_TO_IMAGE_FOREGROUND}"
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
split_method = "${SPLIT_METHOD}"
split_param_raw = "${SPLIT_PARAM}"
candidate_params = [split_param_raw]
try:
    candidate_params.append(int(split_param_raw))
except ValueError:
    pass
try:
    candidate_params.append(float(split_param_raw))
except ValueError:
    pass
resolved_param = None
if split_method in splits:
    for candidate in candidate_params:
        if candidate in splits[split_method]:
            resolved_param = candidate
            break
if resolved_param is None:
    raise SystemExit(
        f"Missing split {split_method}[{split_param_raw}] in "
        + str(task_dir / "splits.pkl")
    )
folds = splits[split_method][resolved_param]
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
echo "[$(date)] Starting finetune for fold ${FOLD_IDX}..."

LP_FLAG=""
if [[ "${LINEAR_PROBING}" == "1" ]]; then
  LP_FLAG="--linear_probing"
fi

BACKBONE_LR_FLAG=()
if [[ -n "${BACKBONE_LEARNING_RATE}" ]]; then
  BACKBONE_LR_FLAG=(--backbone_learning_rate "${BACKBONE_LEARNING_RATE}")
fi

SEG_OUTPUT_PRIOR_FLAG=()
if [[ -n "${SEG_OUTPUT_FG_PRIOR}" ]]; then
  SEG_OUTPUT_PRIOR_FLAG=(--seg_output_fg_prior "${SEG_OUTPUT_FG_PRIOR}")
fi

if [[ "${RUN_STYLE}" == "repo_faithful" && "${MODEL_NAME}" == "unet_b" ]] || [[ "${RUN_STYLE}" == "repo_faithful" && "${MODEL_NAME}" == "unet_xl" ]]; then
  export PYTHONPATH="${BASELINE_REPO_ROOT}/src:${PYTHONPATH:-}"
  python3 -u "${BASELINE_FINETUNE_PY}" \
    --taskid "${TASK_ID}" \
    --data_dir "${DATA_ROOT}" \
    --save_dir "${SAVE_DIR}" \
    --pretrained_weights_path "${PRETRAINED}" \
    --model_name "${MODEL_NAME}" \
    --patch_size "${PATCH_D}" \
    --batch_size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --train_batches_per_epoch "${TRAIN_BATCHES}" \
    --augmentation_preset "${AUGMENTATION_PRESET}" \
    --split_method "${SPLIT_METHOD}" \
    --split_param "${SPLIT_PARAM}" \
    --split_idx "${FOLD_IDX}" \
    --num_workers "${NUM_WORKERS}" \
    --num_devices 1 \
    --precision bf16-mixed \
    --best_checkpoint_monitor "${BEST_CHECKPOINT_MONITOR}" \
    --best_checkpoint_mode "${BEST_CHECKPOINT_MODE}" \
    --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
    --early_stopping_monitor "${EARLY_STOPPING_MONITOR}" \
    --early_stopping_mode "${EARLY_STOPPING_MODE}" \
    --experiment "task${TASK_ID}_repo_${MODEL_NAME}${RUN_SUFFIX}_fold${FOLD_IDX}" \
    --new_version
else
  python3 -u "${FINETUNE_PY}" \
    --taskid "${TASK_ID}" \
    --data_dir "${DATA_ROOT}" \
    --save_dir "${SAVE_DIR}" \
    --pretrained_weights_path "${PRETRAINED}" \
    --model_name "${MODEL_NAME}" \
    --patch_size_dhw "${PATCH_D}" "${PATCH_H}" "${PATCH_W}" \
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
    --seg_aux_loss_weight "${SEG_AUX_LOSS_WEIGHT}" \
    --seg_aux_target "${SEG_AUX_TARGET}" \
    --seg_boundary_radius "${SEG_BOUNDARY_RADIUS}" \
    --seg_small_lesion_weight "${SEG_SMALL_LESION_WEIGHT}" \
    --seg_small_lesion_thresholds ${SEG_SMALL_THRESHOLDS} \
    --p_oversample_foreground "${P_OVERSAMPLE_FOREGROUND}" \
    --split_method "${SPLIT_METHOD}" \
    --split_param "${SPLIT_PARAM}" \
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
    --experiment "task${TASK_ID}_fullvol_${MODEL_NAME}${RUN_SUFFIX}_fold${FOLD_IDX}" \
    --new_version \
    ${LP_FLAG}
fi

FT_END=$(date +%s)
FT_SECS=$(( FT_END - FT_START ))
echo "[$(date)] Fine-tune done: ${FT_SECS}s"

CKPT_DIR="${SAVE_DIR}/${TASK_NAME}/${MODEL_NAME}"
LATEST_VERSION_DIR=$(find "${CKPT_DIR}" -maxdepth 1 -type d -name "version_*" 2>/dev/null | sort -V | tail -1)
if [[ -z "${LATEST_VERSION_DIR}" ]]; then
  echo "[ERROR] No version_* directory found in ${CKPT_DIR}"
  exit 6
fi

BEST_CKPT=$(find "${LATEST_VERSION_DIR}/checkpoints" -maxdepth 1 -name "best.ckpt" 2>/dev/null | sort -V | tail -1)
LAST_CKPT=$(find "${LATEST_VERSION_DIR}/checkpoints" -maxdepth 1 -name "last.ckpt" 2>/dev/null | sort -V | tail -1)
FALLBACK_CKPT=$(find "${LATEST_VERSION_DIR}/checkpoints" -maxdepth 1 -name "*.ckpt" 2>/dev/null | sort -V | tail -1)

if [[ -z "${BEST_CKPT}" ]]; then
  BEST_CKPT="${FALLBACK_CKPT}"
fi
if [[ -z "${LAST_CKPT}" ]]; then
  LAST_CKPT="${FALLBACK_CKPT}"
fi

case "${CHECKPOINT_SELECT}" in
  best)
    CKPT="${BEST_CKPT}"
    ;;
  last)
    CKPT="${LAST_CKPT}"
    ;;
  *)
    echo "[ERROR] Unsupported CHECKPOINT_SELECT: ${CHECKPOINT_SELECT}"
    exit 7
    ;;
esac

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[ERROR] Failed to locate checkpoint under ${LATEST_VERSION_DIR}/checkpoints"
  exit 8
fi

echo "[$(date)] Selected checkpoint (${CHECKPOINT_SELECT}): ${CKPT}"
echo "[$(date)] Best checkpoint: ${BEST_CKPT:-<missing>}"
echo "[$(date)] Last checkpoint: ${LAST_CKPT:-<missing>}"

MIRROR_FLAG=()
if [[ "${MIRROR_TTA}" == "1" ]]; then
  MIRROR_FLAG=(--mirror)
fi

case "${EVAL_MODE}" in
  argmax_no_tune)
    EVAL_MODE_FLAG=(--argmax_no_tune)
    ;;
  tuned)
    EVAL_MODE_FLAG=()
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

IMAGE_FG_FLAG=()
if [[ "${RESTRICT_TO_IMAGE_FOREGROUND}" == "1" ]]; then
  IMAGE_FG_FLAG=(--restrict_to_image_foreground)
fi

run_eval() {
  local checkpoint_path="$1"
  local output_json="$2"
  local label="$3"
  echo "[$(date)] Starting evaluation (${label})..."
  python3 -u "${EVAL_PY}" \
    --task_dir "${TASK_DIR}" \
    --checkpoint "${checkpoint_path}" \
    --model_name "${MODEL_NAME}" \
    --seg_head_variant "${SEG_HEAD_VARIANT}" \
    --fold_idx "${FOLD_IDX}" \
    --split_method "${SPLIT_METHOD}" \
    --split_param "${SPLIT_PARAM}" \
    --patch_size_dhw "${PATCH_D}" "${PATCH_H}" "${PATCH_W}" \
    --output_json "${output_json}" \
    "${MIRROR_FLAG[@]}" \
    "${IMAGE_FG_FLAG[@]}" \
    "${EVAL_MODE_FLAG[@]}"
}

EVAL_START=$(date +%s)
run_eval "${CKPT}" "${OUTPUT_JSON}" "${CHECKPOINT_SELECT}"
EVAL_END=$(date +%s)
EVAL_SECS=$(( EVAL_END - EVAL_START ))
echo "[$(date)] Primary evaluation done: ${EVAL_SECS}s"

BEST_OUTPUT_JSON="${OUTPUT_ROOT}/${MODEL_NAME}${RUN_SUFFIX}_fold${FOLD_IDX}_best_test.json"
LAST_OUTPUT_JSON="${OUTPUT_ROOT}/${MODEL_NAME}${RUN_SUFFIX}_fold${FOLD_IDX}_last_test.json"

if [[ "${EVALUATE_BOTH_CHECKPOINTS}" == "1" ]]; then
  if [[ -n "${BEST_CKPT}" && -f "${BEST_CKPT}" && "${BEST_CKPT}" != "${CKPT}" ]]; then
    run_eval "${BEST_CKPT}" "${BEST_OUTPUT_JSON}" "best"
  fi
  if [[ -n "${LAST_CKPT}" && -f "${LAST_CKPT}" && "${LAST_CKPT}" != "${CKPT}" ]]; then
    run_eval "${LAST_CKPT}" "${LAST_OUTPUT_JSON}" "last"
  fi
fi

python3 << PYEOF
import json
from pathlib import Path

def maybe_float(value):
    if value in ("", "null", "None"):
        return None
    return float(value)

def maybe_int(value):
    if value in ("", "null", "None"):
        return None
    return int(value)

timing = {
    "model_name": "${MODEL_NAME}",
    "run_style": "${RUN_STYLE}",
    "task_id": ${TASK_ID},
    "task_name": "${TASK_NAME}",
    "fold_idx": ${FOLD_IDX},
    "split_method": "${SPLIT_METHOD}",
    "split_param": ${SPLIT_PARAM},
    "patch_size_dhw": [${PATCH_D}, ${PATCH_H}, ${PATCH_W}],
    "batch_size": ${BATCH_SIZE},
    "accumulate_grad_batches": ${ACCUMULATE_GRAD_BATCHES},
    "learning_rate": float(${LEARNING_RATE}),
    "backbone_learning_rate": maybe_float("${BACKBONE_LEARNING_RATE:-}"),
    "segmentation_loss": "${SEGMENTATION_LOSS}",
    "seg_attention": "${SEG_ATTENTION}",
    "seg_head_variant": "${SEG_HEAD_VARIANT}",
    "seg_output_fg_prior": maybe_float("${SEG_OUTPUT_FG_PRIOR:-}"),
    "seg_aux_loss_weight": ${SEG_AUX_LOSS_WEIGHT:-0.0},
    "seg_aux_target": "${SEG_AUX_TARGET}",
    "seg_boundary_radius": int(${SEG_BOUNDARY_RADIUS}),
    "seg_small_lesion_weight": float(${SEG_SMALL_LESION_WEIGHT}),
    "seg_small_lesion_thresholds": [int(x) for x in "${SEG_SMALL_THRESHOLDS}".split()],
    "p_oversample_foreground": float(${P_OVERSAMPLE_FOREGROUND}),
    "best_checkpoint_monitor": "${BEST_CHECKPOINT_MONITOR}",
    "best_checkpoint_mode": "${BEST_CHECKPOINT_MODE}",
    "early_stopping_monitor": "${EARLY_STOPPING_MONITOR}",
    "early_stopping_mode": "${EARLY_STOPPING_MODE}",
    "early_stopping_patience": ${EARLY_STOPPING_PATIENCE},
    "checkpoint_select": "${CHECKPOINT_SELECT}",
    "checkpoint_path": "${CKPT}",
    "best_checkpoint_path": "${BEST_CKPT:-}",
    "last_checkpoint_path": "${LAST_CKPT:-}",
    "evaluate_both_checkpoints": bool(int(${EVALUATE_BOTH_CHECKPOINTS})),
    "best_output_json": "${BEST_OUTPUT_JSON}",
    "last_output_json": "${LAST_OUTPUT_JSON}",
    "evaluation_mode": "${EVAL_MODE}",
    "fixed_threshold": maybe_float("${FIXED_THRESHOLD:-}"),
    "fixed_min_component_voxels": maybe_int("${FIXED_MIN_COMPONENT_VOXELS:-}"),
    "fixed_keep_largest": None if "${FIXED_KEEP_LARGEST:-}" in ("", "null", "None") else bool(int("${FIXED_KEEP_LARGEST:-}")),
    "restrict_to_image_foreground": bool(int(${RESTRICT_TO_IMAGE_FOREGROUND})),
    "mirror_tta": bool(int(${MIRROR_TTA})),
    "train_seconds": ${FT_SECS},
    "eval_seconds": ${EVAL_SECS},
}
Path("${TIMING_JSON}").write_text(json.dumps(timing, indent=2))
print(f"Wrote timing JSON to ${TIMING_JSON}")
PYEOF

echo "[$(date)] Task${TASK_ID} full-volume run complete"
