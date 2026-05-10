#!/bin/bash
#SBATCH --job-name=prep_ds004199_t1flair10
#SBATCH --account=bioinf545w26_class
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

module load python/3.10.4
module load freesurfer/7.4.1
source "${FOMO_ENV:-/scratch/wilms_root/wilms0/jinhlee/fomo-diffusion/fomo_env/bin/activate}"

PREP_SCRIPT="${REPO_ROOT}/pipelines/fcd_segmentation/scripts/build_fcd_segmentation_ds004199_dataset.py"

SOURCE_DIR="${SOURCE_DIR:-/nfs/turbo/umms-wilms1/FOMO/Data/fomo300k/raw_nifti/PT030_OpenNeuro/ds004199}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/nfs/turbo/umms-wilms1/FOMO/Data/openneuro_task10_t1flairseg_1mm_20260507}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_FOLDS="${NUM_FOLDS:-5}"

SKULL_STRIP_FLAG=""
if [[ "${NO_SKULL_STRIP:-0}" == "1" ]]; then
  SKULL_STRIP_FLAG="--no_skull_strip"
fi

ALLOW_UNSTRIPPED_FLAG=""
if [[ "${ALLOW_UNSTRIPPED_FALLBACK:-0}" == "1" ]]; then
  ALLOW_UNSTRIPPED_FLAG="--allow_unstripped_fallback"
fi

LIMIT_FLAG=""
if [[ -n "${LIMIT_SUBJECTS:-}" && "${LIMIT_SUBJECTS}" != "0" ]]; then
  LIMIT_FLAG="--limit_subjects ${LIMIT_SUBJECTS}"
fi

echo "=========================================================="
echo "[$(date)] Preparing corrected ds004199 Task 10 (T1+FLAIR lesion segmentation)"
echo "Source: ${SOURCE_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Num workers: ${NUM_WORKERS}"
echo "Num folds: ${NUM_FOLDS}"
echo "=========================================================="

python "${PREP_SCRIPT}" \
  --source_dir "${SOURCE_DIR}" \
  --output_root "${OUTPUT_ROOT}" \
  --num_workers "${NUM_WORKERS}" \
  --num_folds "${NUM_FOLDS}" \
  ${SKULL_STRIP_FLAG} \
  ${ALLOW_UNSTRIPPED_FLAG} \
  ${LIMIT_FLAG}

echo "[$(date)] Corrected ds004199 Task 10 preparation complete"
