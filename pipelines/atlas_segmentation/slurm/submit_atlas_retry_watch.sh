#!/bin/bash
set -euo pipefail

SCRIPT_PATH="${1:-}"
if [[ -z "${SCRIPT_PATH}" ]]; then
  echo "Usage: $0 /abs/or/relative/path/to/slurm_wrapper.sh"
  exit 2
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "[ERROR] Missing wrapper: ${SCRIPT_PATH}"
  exit 3
fi

MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
POLL_SECONDS="${POLL_SECONDS:-20}"
WATCH_REGEX='Current Epoch:[[:space:]]+1|Epoch 0: 100%|Epoch 1:'
KNOWN_BAD_DEFAULT="gl1501,gl1505,gl1516,gl1519,gl1520,gl1525"
EXCLUDE_NODES="${EXCLUDE_NODES:-$KNOWN_BAD_DEFAULT}"

script_default_model() {
  local script_path="$1"
  if [[ -n "${MODEL_NAME:-}" ]]; then
    echo "${MODEL_NAME}"
    return 0
  fi
  perl -ne 'if (/^MODEL_NAME="\$\{MODEL_NAME:-([^\"]+)\}"$/) { print $1; exit 0 }' "${script_path}" 2>/dev/null || true
}

submit_account_for_script() {
  local model_name="$1"
  case "${model_name}" in
    cleandift_s23)
      echo "wilms99"
      ;;
    unet_b)
      echo "entr475s400y26_class"
      ;;
    unet_xl)
      echo "bioinf504w26_class"
      ;;
    mmunetvae)
      echo "bioinf529w26_class"
      ;;
    *)
      echo "bioinf529w26_class"
      ;;
  esac
}

submit_time_for_script() {
  local model_name="$1"
  if [[ "${model_name}" == "cleandift_s23" ]]; then
    echo "12:00:00"
  else
    echo "08:00:00"
  fi
}

job_log_path() {
  local job_id="$1"
  echo "/nfs/turbo/umms-wilms1/FOMO/experiments/jinhlee/logs/$(basename "${SCRIPT_PATH%.sh}")-${job_id}.log" | \
    sed 's#run_atlas_cleandift_smalllesion#atlas_cd_small#; s#run_atlas_segmentation_fullvol#atlas_task12#; s#run_atlas_baseline_recheck#atlas_base#'
}

submit_job() {
  local exclude_args=()
  if [[ -n "${EXCLUDE_NODES}" ]]; then
    exclude_args=(-x "${EXCLUDE_NODES}")
  fi
  local model_name
  model_name="$(script_default_model "${SCRIPT_PATH}")"
  local account
  account="$(submit_account_for_script "${model_name}")"
  local walltime
  walltime="$(submit_time_for_script "${model_name}")"
  sbatch --parsable --account="${account}" --time="${walltime}" "${exclude_args[@]}" "${SCRIPT_PATH}"
}

append_exclude_node() {
  local node="$1"
  if [[ -z "${node}" ]]; then
    return
  fi
  if [[ -z "${EXCLUDE_NODES}" ]]; then
    EXCLUDE_NODES="${node}"
  elif [[ ",${EXCLUDE_NODES}," != *",${node},"* ]]; then
    EXCLUDE_NODES="${EXCLUDE_NODES},${node}"
  fi
}

attempt=1
while :; do
  if (( MAX_ATTEMPTS > 0 )) && (( attempt > MAX_ATTEMPTS )); then
    echo "[retry-watch] exhausted ${MAX_ATTEMPTS} attempts"
    exit 1
  fi

  max_attempts_label="${MAX_ATTEMPTS}"
  if (( MAX_ATTEMPTS <= 0 )); then
    max_attempts_label="inf"
  fi

  echo "[retry-watch] attempt ${attempt}/${max_attempts_label}"
  job_id="$(submit_job)"
  echo "[retry-watch] submitted ${job_id} (exclude=${EXCLUDE_NODES:-<none>})"

  log_path="$(job_log_path "${job_id}")"
  while true; do
    sleep "${POLL_SECONDS}"
    job_info="$(sacct -j "${job_id}" --format=JobID,State,ExitCode,NodeList%20,Elapsed -n | sed -n '1p')"
    echo "[retry-watch] ${job_info}"

    if [[ -f "${log_path}" ]]; then
      tail -n 30 "${log_path}" || true
      if rg -q "${WATCH_REGEX}" "${log_path}"; then
        echo "[retry-watch] first epoch reached for ${job_id}"
        exit 0
      fi
    fi

    state="$(awk '{print $2}' <<<"${job_info}")"
    exit_code="$(awk '{print $3}' <<<"${job_info}")"
    node="$(awk '{print $4}' <<<"${job_info}")"

    case "${state}" in
      PENDING|RUNNING|COMPLETING)
        ;;
      FAILED)
        if [[ "${exit_code}" == "0:53" ]]; then
          echo "[retry-watch] node-side signal 53 on ${node}; excluding and retrying"
          append_exclude_node "${node}"
          break
        fi
        echo "[retry-watch] job ${job_id} failed with ${exit_code}"
        exit 1
        ;;
      COMPLETED)
        echo "[retry-watch] job ${job_id} completed before epoch-1 signal was observed"
        exit 0
        ;;
      CANCELLED*|TIMEOUT|OUT_OF_MEMORY)
        echo "[retry-watch] job ${job_id} ended in state ${state}"
        exit 1
        ;;
      *)
        ;;
    esac
  done

  attempt=$((attempt + 1))
done
