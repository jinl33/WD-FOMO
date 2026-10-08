#!/bin/bash
#SBATCH --job-name=wdfomo_smoke
#SBATCH --partition=spgpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=wdfomo_smoke-%j.log

set -euo pipefail

if [[ -n "${FOMO_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${FOMO_REPO_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/smoke_test.py" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
VENV="${FOMO_VENV:-$(cd "${REPO_ROOT}/.." && pwd)/fomo_env}"

module load python/3.10.4
export LD_LIBRARY_PATH="/sw/pkgs/arc/python/3.10.4/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -f "${REPO_ROOT}/scripts/smoke_test.py" ]]; then
  echo "[ERROR] no smoke_test.py under ${REPO_ROOT} -- set FOMO_REPO_ROOT to the repository root" >&2
  exit 11
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "[ERROR] no Python at ${VENV}/bin/python -- set FOMO_VENV to your environment" >&2
  exit 10
fi

echo "repo: ${REPO_ROOT}"
echo "venv: ${VENV}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${VENV}/bin/python" -u "${REPO_ROOT}/scripts/smoke_test.py" --stage all --device cuda --wavelet-size 16 --steps 5
