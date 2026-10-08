#!/bin/bash
#SBATCH --job-name=ft_task1_infarct_cv
#SBATCH --account=wilms99
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log

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
if [[ ! -f "${REPO_ROOT}/src/downstream/finetune.py" ]]; then
  echo "[ERROR] REPO_ROOT=${REPO_ROOT} does not contain src/downstream/finetune.py" >&2
  exit 10
fi

module load python/3.10.4
source "${FOMO_ENV:-/scratch/wilms_root/wilms0/jinhlee/fomo-diffusion/fomo_env/bin/activate}"

FINETUNE_PY="${REPO_ROOT}/src/downstream/finetune.py"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src/downstream:${REPO_ROOT}/src/pretraining"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the FOMO25 Task 1 (infarct) data root}"
TASK_NAME="${TASK_NAME:-Task001_FOMO1}"
TASK_DIR="${DATA_ROOT}/${TASK_NAME}"
FOLD_IDX="${FOLD_IDX:-${SLURM_ARRAY_TASK_ID:-0}}"
SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/benchmark_results_infarct_classification}/fold${FOLD_IDX}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/inference_outputs/infarct_classification}"

MODEL_NAME="${MODEL_NAME:-cleandift_s23}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-100}"
TRAIN_BATCHES="${TRAIN_BATCHES:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

case "${MODEL_NAME}" in
  unet_b)
    PRETRAINED="${PRETRAINED:-${REPO_ROOT}/baseline_pretrained_models/unet_b.ckpt}"
    PATCH_D="${PATCH_D:-96}"; PATCH_H="${PATCH_H:-96}"; PATCH_W="${PATCH_W:-96}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-none}"
    ;;
  unet_xl)
    PRETRAINED="${PRETRAINED:-${REPO_ROOT}/baseline_pretrained_models/unet_xl.ckpt}"
    PATCH_D="${PATCH_D:-96}"; PATCH_H="${PATCH_H:-96}"; PATCH_W="${PATCH_W:-96}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-none}"
    ;;
  mmunetvae)
    PRETRAINED="${PRETRAINED:-/nfs/turbo/umms-wilms1/FOMO/checkpoints/fomo25_mmunetvae_pretrained.ckpt}"
    PATCH_D="${PATCH_D:-64}"; PATCH_H="${PATCH_H:-64}"; PATCH_W="${PATCH_W:-64}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-basic}"
    ;;
  cleandift_s23)
    : "${PRETRAINED:?cleandift_s23 requires an explicit PRETRAINED (the distilled encoder checkpoint)}"
    PATCH_D="${PATCH_D:-192}"; PATCH_H="${PATCH_H:-256}"; PATCH_W="${PATCH_W:-192}"
    BATCH_SIZE="${BATCH_SIZE:-2}"
    AUGMENTATION_PRESET="${AUGMENTATION_PRESET:-none}"
    ;;
  *)
    echo "[ERROR] Unsupported MODEL_NAME: ${MODEL_NAME}"
    exit 2
    ;;
esac

OUTPUT_JSON="${OUTPUT_ROOT}/${MODEL_NAME}_fold${FOLD_IDX}_test.json"
mkdir -p "${OUTPUT_ROOT}"

echo "=========================================================="
echo "[$(date)] Infarct classification CV -- ${MODEL_NAME}, fold ${FOLD_IDX}/${FOLDS}"
echo "Patch: ${PATCH_D} ${PATCH_H} ${PATCH_W}  Batch: ${BATCH_SIZE}  Augmentation: ${AUGMENTATION_PRESET}"
echo "Task dir: ${TASK_DIR}"
echo "Save dir: ${SAVE_DIR}"
echo "Output JSON: ${OUTPUT_JSON}"
echo "=========================================================="

if [[ ! -d "${TASK_DIR}" ]]; then
  echo "[ERROR] Missing Task 1 directory: ${TASK_DIR}"; exit 3
fi
if [[ ! -f "${TASK_DIR}/splits.pkl" ]]; then
  echo "[ERROR] Missing splits.pkl in ${TASK_DIR}. Generate it with build_infarct_classification_splits.py."
  exit 4
fi
if [[ ! -f "${PRETRAINED}" ]]; then
  echo "[ERROR] Missing pretrained checkpoint: ${PRETRAINED}"; exit 5
fi

python3 << PYEOF
import pickle
from pathlib import Path
task_dir = Path("${TASK_DIR}")
splits = pickle.load(open(task_dir / "splits.pkl", "rb"))
if "kfold" not in splits or ${FOLDS} not in splits["kfold"]:
    raise SystemExit("Expected kfold splits in splits.pkl")
fold = splits["kfold"][${FOLDS}][${FOLD_IDX}]
if set(fold["train"]) & set(fold["val"]):
    raise SystemExit("Train/val overlap detected")
print(f"Fold {${FOLD_IDX}}: train={len(fold['train'])} val={len(fold['val'])}")
PYEOF

FT_START=$(date +%s)

python3 "${FINETUNE_PY}" \
  --taskid 1 \
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
  --split_method kfold \
  --split_param "${FOLDS}" \
  --split_idx "${FOLD_IDX}" \
  --num_workers "${NUM_WORKERS}" \
  --num_devices 1 \
  --precision bf16-mixed \
  --experiment "infarct_classification_${MODEL_NAME}_fold${FOLD_IDX}" \
  --best_checkpoint_monitor val/auroc \
  --best_checkpoint_mode max \
  --early_stopping_patience 15 \
  --early_stopping_monitor val/auroc \
  --early_stopping_mode max \
  --new_version

FT_END=$(date +%s)
FT_SECS=$(( FT_END - FT_START ))
echo "[$(date)] Fine-tune done: ${FT_SECS}s"

CKPT_DIR="${SAVE_DIR}/${TASK_NAME}/${MODEL_NAME}"
LATEST_VERSION_DIR=$(find "${CKPT_DIR}" -maxdepth 1 -type d -name "version_*" 2>/dev/null | sort -V | tail -1)
if [[ -z "${LATEST_VERSION_DIR}" ]]; then
  echo "[ERROR] No version_* directory found in ${CKPT_DIR}"; exit 6
fi
CKPT=$(find "${LATEST_VERSION_DIR}/checkpoints" -maxdepth 1 -name "best.ckpt" 2>/dev/null | sort -V | tail -1)
if [[ -z "${CKPT}" ]]; then
  CKPT=$(find "${LATEST_VERSION_DIR}/checkpoints" -maxdepth 1 -name "*.ckpt" 2>/dev/null | sort -V | tail -1)
fi
if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] No checkpoint found in ${LATEST_VERSION_DIR}/checkpoints"; exit 7
fi
echo "Checkpoint: ${CKPT}"

INF_START=$(date +%s)

python3 << PYEOF
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

sys.path.insert(0, "${REPO_ROOT}/src/downstream")
sys.path.insert(0, "${REPO_ROOT}/src/pretraining")

from models.supervised_cls import SupervisedClsModel

task_dir = Path("${TASK_DIR}")
splits = pickle.load(open(task_dir / "splits.pkl", "rb"))
fold = splits["kfold"][${FOLDS}][${FOLD_IDX}]
val_ids = sorted(fold["val"])
checkpoint_path = "${CKPT}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print(f"Validation subjects: {len(val_ids)}")

model = SupervisedClsModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
model.eval().to(device)

PATCH_D, PATCH_H, PATCH_W = ${PATCH_D}, ${PATCH_H}, ${PATCH_W}

results, errors = [], []
for idx, fid in enumerate(val_ids, 1):
    npy_path = task_dir / f"{fid}.npy"
    txt_path = task_dir / f"{fid}.txt"
    if not npy_path.exists() or not txt_path.exists():
        errors.append(fid)
        continue

    label = int(float(txt_path.read_text().strip()))
    vol = np.load(npy_path).astype(np.float32)
    if vol.ndim == 3:
        vol = vol[np.newaxis, ...]

    pad = []
    for size, target in zip(reversed(vol.shape[-3:]), [PATCH_W, PATCH_H, PATCH_D]):
        before = max(0, (target - size) // 2)
        after = max(0, target - size - before)
        pad.extend([before, after])
    if any(pad):
        vol = np.pad(vol, ((0, 0), (pad[4], pad[5]), (pad[2], pad[3]), (pad[0], pad[1])), mode="constant")

    d, h, w = vol.shape[-3:]
    sd = max(0, (d - PATCH_D) // 2)
    sh = max(0, (h - PATCH_H) // 2)
    sw = max(0, (w - PATCH_W) // 2)
    vol = vol[:, sd:sd + PATCH_D, sh:sh + PATCH_H, sw:sw + PATCH_W]

    inputs = torch.from_numpy(vol)[None].to(device)
    with torch.no_grad():
        if "${MODEL_NAME}" == "mmunetvae":
            logits = model.run_predict(inputs.float())
        else:
            logits = model(inputs.float())
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    results.append({
        "id": fid, "ground_truth": label, "pred_class": pred,
        "prob_control": float(probs[0]), "prob_case": float(probs[1]),
    })
    if idx % 5 == 0 or idx == len(val_ids):
        print(f"[{idx}/{len(val_ids)}] {fid}: pred={pred}, gt={label}, p_case={probs[1]:.4f}")

valid = [r for r in results if r["ground_truth"] in (0, 1)]
y_true = np.array([r["ground_truth"] for r in valid], dtype=int)
y_pred = np.array([r["pred_class"] for r in valid], dtype=int)
y_prob = np.array([r["prob_case"] for r in valid], dtype=float)

acc = float(accuracy_score(y_true, y_pred))
bacc = float(balanced_accuracy_score(y_true, y_pred))
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
auroc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else float("nan")
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

output = {
    "task": "${TASK_NAME}", "model": "${MODEL_NAME}", "fold_idx": ${FOLD_IDX},
    "checkpoint": checkpoint_path, "n_subjects": int(len(valid)),
    "accuracy": acc, "balanced_accuracy": bacc,
    "precision": float(prec), "recall": float(rec), "f1": float(f1),
    "auroc": auroc,
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "n_errors": len(errors), "errors": errors, "predictions": results,
}

Path("${OUTPUT_JSON}").parent.mkdir(parents=True, exist_ok=True)
with open("${OUTPUT_JSON}", "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved: ${OUTPUT_JSON}")
print(f"FOLD ${FOLD_IDX} AUROC={auroc:.4f} ACC={acc:.4f} BACC={bacc:.4f}")
PYEOF

INF_END=$(date +%s)
echo "[$(date)] COMPLETE: model=${MODEL_NAME} fold=${FOLD_IDX} FT=${FT_SECS}s INF=$(( INF_END - INF_START ))s"
