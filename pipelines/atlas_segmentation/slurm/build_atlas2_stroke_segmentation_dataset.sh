#!/bin/bash
#SBATCH --job-name=build_atlas2_task12
#SBATCH --account=wilms99
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src/downstream:${REPO_ROOT}/src/pretraining"
export PYTHONUNBUFFERED=1

SOURCE_DIR="${SOURCE_DIR:-/nfs/turbo/umms-wilms1/FOMO/atlas_raw_downloads/extracted/Training_Raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/wilms_root/wilms0/jinhlee/fomo-diffusion/atlas2_stroke_t1seg_1mm_all955_20260528}"
SUBSET_SIZE="${SUBSET_SIZE:-0}"
NUM_FOLDS="${NUM_FOLDS:-5}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
RANDOM_STATE="${RANDOM_STATE:-42}"
OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

echo "=========================================================="
echo "[$(date)] ATLAS v2 Task012 build"
echo "Source dir: ${SOURCE_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Subset size: ${SUBSET_SIZE}"
echo "Num folds: ${NUM_FOLDS}"
echo "Workers: ${NUM_WORKERS}"
echo "Random state: ${RANDOM_STATE}"
echo "=========================================================="

python3 -u pipelines/atlas_segmentation/scripts/build_atlas2_stroke_segmentation_dataset.py \
  --source_dir "${SOURCE_DIR}" \
  --output_root "${OUTPUT_ROOT}" \
  --subset_size "${SUBSET_SIZE}" \
  --num_folds "${NUM_FOLDS}" \
  --num_workers "${NUM_WORKERS}" \
  --random_state "${RANDOM_STATE}" \
  "${OVERWRITE_FLAG[@]}"

echo "[$(date)] ATLAS v2 Task012 build complete"
