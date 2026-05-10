#!/bin/bash
#SBATCH --job-name=ba_unet_exactpt
#SBATCH --account=entr475s400y26_class
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/turbo/umms-wilms1/FOMO/experiments/jinhlee/logs/%x-%j.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FINETUNE_PY="${REPO_ROOT}/src/downstream/finetune.py"
EVAL_PY="${REPO_ROOT}/pipelines/brain_age/scripts/evaluate_brain_age_exact_split.py"

module load python/3.10.4
source "${FOMO_ENV:-/scratch/wilms_root/wilms0/jinhlee/fomo-diffusion/fomo_env/bin/activate}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src/downstream:${REPO_ROOT}/src/pretraining"
export WANDB_MODE=disabled
export WANDB_DISABLED=true

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME=unet_b or MODEL_NAME=unet_xl}"
case "${MODEL_NAME}" in
  unet_b)
    PRETRAINED="${PRETRAINED:-${REPO_ROOT}/baseline_pretrained_models/unet_b.ckpt}"
    ;;
  unet_xl)
    PRETRAINED="${PRETRAINED:-${REPO_ROOT}/baseline_pretrained_models/unet_xl.ckpt}"
    ;;
  *)
    echo "[ERROR] Unsupported MODEL_NAME=${MODEL_NAME}"
    exit 1
    ;;
esac

DATA_ROOT="${DATA_ROOT:-/nfs/turbo/umms-wilms1/FOMO/Data/fomo300k_unified_match_s23_p160_20260502_exactpretrain_full700_sub100headerfix}"
TASK_NAME="${TASK_NAME:-Task_FOMO300K_BrainAge_T1_v2}"
TASK_DIR="${DATA_ROOT}/${TASK_NAME}"
SPLIT_JSON="${SPLIT_JSON:-/nfs/turbo/umms-wilms1/FOMO/Data/fomo300k_unified/splits/fomo300k_v2_stratified_ft200_test500_gsp179_brainlat179_task3142_diskvalidated_siteagebalanced.json}"
SPLIT_METHOD_KEY="${SPLIT_METHOD_KEY:-match_s23_p160_20260502_siteagebalanced}"
SAVE_ROOT="${SAVE_ROOT:-/nfs/turbo/umms-wilms1/FOMO/benchmark_results_lightweight_fomo_brain_age_${MODEL_NAME}}"
RUN_TAG="${RUN_TAG:-exactpretrain_siteagebalanced}"
OUTPUT_JSON="${OUTPUT_JSON:-/nfs/turbo/umms-wilms1/FOMO/inference_outputs/brain_age_${MODEL_NAME}_${RUN_TAG}_test.json}"
TIMING_JSON="${TIMING_JSON:-/nfs/turbo/umms-wilms1/FOMO/inference_outputs/brain_age_${MODEL_NAME}_${RUN_TAG}_timing.json}"

PATCH_D="${PATCH_D:-96}"
PATCH_H="${PATCH_H:-96}"
PATCH_W="${PATCH_W:-96}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-100}"
TRAIN_BATCHES="${TRAIN_BATCHES:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"

FT_START=$(date +%s)
python3 "${FINETUNE_PY}" \
  --taskid 7 \
  --data_dir "${DATA_ROOT}" \
  --save_dir "${SAVE_ROOT}" \
  --pretrained_weights_path "${PRETRAINED}" \
  --model_name "${MODEL_NAME}" \
  --patch_size_dhw "${PATCH_D}" "${PATCH_H}" "${PATCH_W}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --train_batches_per_epoch "${TRAIN_BATCHES}" \
  --augmentation_preset all \
  --split_method "${SPLIT_METHOD_KEY}" \
  --split_param predefined \
  --num_workers "${NUM_WORKERS}" \
  --num_devices 1 \
  --precision bf16-mixed \
  --experiment "brain_age_${MODEL_NAME}_${RUN_TAG}" \
  --new_version
FT_END=$(date +%s)

CKPT_DIR="${SAVE_ROOT}/${TASK_NAME}/${MODEL_NAME}"
LATEST_VERSION_DIR=$(find "${CKPT_DIR}" -maxdepth 1 -type d -name "version_*" | sort -V | tail -1)
CKPT=$(find "${LATEST_VERSION_DIR}/checkpoints" -maxdepth 1 -name "best*.ckpt" | sort -V | tail -1)
if [[ -z "${CKPT}" ]]; then
  CKPT=$(find "${LATEST_VERSION_DIR}/checkpoints" -maxdepth 1 -name "*.ckpt" | sort -V | tail -1)
fi
if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] No checkpoint found in ${LATEST_VERSION_DIR}/checkpoints"
  exit 1
fi

INF_START=$(date +%s)
python3 "${EVAL_PY}" \
  --checkpoint "${CKPT}" \
  --task_dir "${TASK_DIR}" \
  --split_json "${SPLIT_JSON}" \
  --output_json "${OUTPUT_JSON}" \
  --patch_size_dhw "${PATCH_D}" "${PATCH_H}" "${PATCH_W}"
INF_END=$(date +%s)

python3 << PYEOF
import json
payload = {
    "checkpoint": "${CKPT}",
    "finetune_seconds": int(${FT_END} - ${FT_START}),
    "inference_seconds": int(${INF_END} - ${INF_START}),
}
with open("${TIMING_JSON}", "w") as f:
    json.dump(payload, f, indent=2)
print(json.dumps(payload, indent=2))
PYEOF
