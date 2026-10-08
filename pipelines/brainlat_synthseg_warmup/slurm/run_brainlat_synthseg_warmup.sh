#!/bin/bash
#SBATCH --job-name=ft_task11_brainlatsseg
#SBATCH --account=wilms99
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # 2026-09-06 FIX: spgpu nodes are 32 CPU / 8 GPU = 4 CPU/GPU; requesting 8 for a
                            # 1-GPU job double-bills against the account's GPU allocation (admin-flagged).
#SBATCH --mem=48G
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

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src/downstream:${REPO_ROOT}/src/pretraining"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONUNBUFFERED=1

DATA_ROOT="${DATA_ROOT:-/nfs/turbo/umms-wilms1/FOMO/Data/brainlat_synthseg_warmup_t1_1mm_20260511}"
TASK_NAME="${TASK_NAME:-Task011_BrainLat_SynthSeg_T1_1mm}"
TASK_DIR="${DATA_ROOT}/${TASK_NAME}"
SAVE_ROOT="${SAVE_ROOT:-/nfs/turbo/umms-wilms1/FOMO/benchmark_results_task11_brainlat_synthseg_warmup_20260511}"

MODEL_NAME="${MODEL_NAME:-cleandift_s23}"
AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-basic}"
EPOCHS="${EPOCHS:-60}"
TRAIN_BATCHES="${TRAIN_BATCHES:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEGMENTATION_LOSS="${SEGMENTATION_LOSS:-generalizeddicece}"
RUN_TAG="${RUN_TAG:-gdcewarmup_v1}"

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
    echo "[ERROR] Unsupported MODEL_NAME for warm-up: ${MODEL_NAME}"
    exit 2
    ;;
esac

SAVE_DIR="${SAVE_ROOT}/${RUN_TAG}"

echo "=========================================================="
echo "[$(date)] Task11 BrainLat SynthSeg warm-up"
echo "Model: ${MODEL_NAME}"
echo "Patch: ${PATCH_D} ${PATCH_H} ${PATCH_W}"
echo "Batch size: ${BATCH_SIZE}"
echo "Augmentation: ${AUGMENTATION_PRESET}"
echo "Segmentation loss: ${SEGMENTATION_LOSS}"
echo "Task dir: ${TASK_DIR}"
echo "Save dir: ${SAVE_DIR}"
echo "=========================================================="

if [[ ! -d "${TASK_DIR}" ]]; then
  echo "[ERROR] Missing Task11 directory: ${TASK_DIR}"
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
if "simple_train_val_split" not in splits or 0.2 not in splits["simple_train_val_split"]:
    raise SystemExit("Expected simple_train_val_split[0.2] in splits.pkl")
split = splits["simple_train_val_split"][0.2][0]
train_ids = split["train"]
val_ids = split["val"]
if set(train_ids) & set(val_ids):
    raise SystemExit("Train/val overlap detected")
print(f"Warm-up split: train={len(train_ids)} val={len(val_ids)}")
PYEOF

python3 -u "${FINETUNE_PY}" \
  --taskid 11 \
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
  --split_method simple_train_val_split \
  --split_param 0.2 \
  --split_idx 0 \
  --num_workers "${NUM_WORKERS}" \
  --num_devices 1 \
  --precision bf16-mixed \
  --best_checkpoint_monitor val/dice \
  --best_checkpoint_mode max \
  --early_stopping_patience 15 \
  --early_stopping_monitor val/dice \
  --early_stopping_mode max \
  --experiment "task11_brainlatsseg_${MODEL_NAME}_${RUN_TAG}" \
  --new_version

echo "[$(date)] Task11 BrainLat SynthSeg warm-up complete"
