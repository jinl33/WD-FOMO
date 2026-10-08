#!/bin/bash
#SBATCH --job-name=prep_brainlat_synthseg11
#SBATCH --account=bioinf545w26_class
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # 2026-09-06 FIX: spgpu nodes are 32 CPU / 8 GPU = 4 CPU/GPU; requesting 8 for a
                            # 1-GPU job double-bills against the account's GPU allocation (admin-flagged).
#SBATCH --mem=48G
#SBATCH --time=04:00:00
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

BUILD_PY="${REPO_ROOT}/pipelines/brainlat_synthseg_warmup/scripts/build_brainlat_synthseg_warmup_dataset.py"

IMAGE_DIR="${IMAGE_DIR:-/nfs/turbo/umms-wilms1/FOMO/Data/fomo300k/preprocessed_stripped/PT010_BrainLat}"
LABEL_DIR="${LABEL_DIR:-/nfs/turbo/umms-wilms1/FOMO/Data/fomo300k/preprocessed_stripped/PT010_BrainLat_synthseg}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/turbo/umms-wilms1/FOMO/Data/brainlat_synthseg_warmup_t1_1mm_20260511}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_FOLDS="${NUM_FOLDS:-5}"

OVERWRITE_FLAG=""
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

echo "=========================================================="
echo "[$(date)] Preparing BrainLat SynthSeg warm-up task"
echo "Image dir: ${IMAGE_DIR}"
echo "Label dir: ${LABEL_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Workers: ${NUM_WORKERS}"
echo "Folds: ${NUM_FOLDS}"
echo "=========================================================="

if [[ ! -d "${IMAGE_DIR}" ]]; then
  echo "[ERROR] Missing BrainLat image directory: ${IMAGE_DIR}"
  exit 2
fi
if [[ ! -d "${LABEL_DIR}" ]]; then
  echo "[ERROR] Missing BrainLat SynthSeg directory: ${LABEL_DIR}"
  exit 3
fi

python "${BUILD_PY}" \
  --image_dir "${IMAGE_DIR}" \
  --label_dir "${LABEL_DIR}" \
  --output_root "${OUTPUT_ROOT}" \
  --num_workers "${NUM_WORKERS}" \
  --num_folds "${NUM_FOLDS}" \
  ${OVERWRITE_FLAG}

echo "[$(date)] BrainLat SynthSeg warm-up task preparation complete"
