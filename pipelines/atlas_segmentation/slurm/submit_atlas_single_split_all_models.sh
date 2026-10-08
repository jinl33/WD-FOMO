#!/bin/bash

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

SCRIPT="${REPO_ROOT}/pipelines/atlas_segmentation/slurm/run_atlas_segmentation_single_split.sh"
EXCLUDE_NODES="${EXCLUDE_NODES:-gl1501,gl1505,gl1514,gl1515,gl1516,gl1517,gl1518,gl1519,gl1520,gl1521,gl1525}"
DATA_ROOT="${DATA_ROOT:-/scratch/wilms_root/wilms0/jinhlee/fomo-diffusion/atlas2_stroke_t1seg_1mm_all955_20260528}"

for model in cleandift_s23 unet_b unet_xl mmunetvae; do
  echo "Submitting ${model}..."
  sbatch --exclude="${EXCLUDE_NODES}" --export=ALL,MODEL_NAME="${model}",DATA_ROOT="${DATA_ROOT}" "${SCRIPT}"
done
