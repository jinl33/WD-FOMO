#!/bin/bash
#SBATCH --job-name=build_ba_exactpt700
#SBATCH --account=bioinf504w26_class
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/turbo/umms-wilms1/FOMO/experiments/jinhlee/logs/%x-%j.log

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

module load python/3.10.4
source "${FOMO_ENV:-/scratch/wilms_root/wilms0/jinhlee/fomo-diffusion/fomo_env/bin/activate}"
export PYTHONPATH="${REPO_ROOT}/src/downstream:${REPO_ROOT}/src/pretraining"

cd "${REPO_ROOT}"
python pipelines/brain_age/scripts/rebuild_brain_age_exact_pretrain_root.py --num_workers 8
