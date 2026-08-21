#!/usr/bin/env bash
set -euo pipefail

# Run official HarmBench-style ASR evaluation through Slurm.

PARTITION="${PARTITION:-rtx6000pro}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-02:00:00}"
JOB_NAME="${JOB_NAME:-harmbench_official_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/harmbench_official_eval/$(date +%Y%m%d_%H%M%S)}"

BEHAVIORS_PATH="${BEHAVIORS_PATH:-data/benchmarks/harmbench/harmbench_behaviors_text_test.csv}"
COMPLETIONS_PATH="${COMPLETIONS_PATH:-}"
SAVE_PATH="${SAVE_PATH:-${OUTPUT_DIR}/official_results.json}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_DIR}/official_summary.json}"
CLASSIFIER="${CLASSIFIER:-cais/HarmBench-Llama-2-13b-cls}"
CLASSIFIER_LOCAL_FILES_ONLY="${CLASSIFIER_LOCAL_FILES_ONLY:-1}"
CLASSIFIER_DEVICE_MAP="${CLASSIFIER_DEVICE_MAP:-auto}"
CLASSIFIER_TORCH_DTYPE="${CLASSIFIER_TORCH_DTYPE:-bfloat16}"
CLASSIFIER_BATCH_SIZE="${CLASSIFIER_BATCH_SIZE:-8}"
CLASSIFIER_MAX_INPUT_LENGTH="${CLASSIFIER_MAX_INPUT_LENGTH:-4096}"
NUM_TOKENS="${NUM_TOKENS:-512}"
HASH_DIR="${HASH_DIR:-}"
LOG_EVERY="${LOG_EVERY:-10}"
DEPENDENCY="${DEPENDENCY:-}"
VENV="${VENV:-/home/jwj/work/.venv}"

HF_HOME="${LLADA_HF_HOME:-/home/jwj/work/.cache/huggingface}"
HF_HUB_CACHE="${LLADA_HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_XET_CACHE="${LLADA_HF_XET_CACHE:-${HF_HOME}/xet}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

if [[ -z "${COMPLETIONS_PATH}" ]]; then
  echo "Set COMPLETIONS_PATH=..." >&2
  exit 2
fi
if [[ -z "${HASH_DIR}" ]]; then
  echo "Set HASH_DIR=... to the HarmBench copyright_classifier_hashes directory." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "$(dirname "${SAVE_PATH}")" "$(dirname "${SUMMARY_PATH}")"

export BEHAVIORS_PATH COMPLETIONS_PATH SAVE_PATH SUMMARY_PATH
export CLASSIFIER CLASSIFIER_LOCAL_FILES_ONLY CLASSIFIER_DEVICE_MAP
export CLASSIFIER_TORCH_DTYPE CLASSIFIER_BATCH_SIZE CLASSIFIER_MAX_INPUT_LENGTH
export NUM_TOKENS HASH_DIR LOG_EVERY VENV
export HF_HOME HF_HUB_CACHE HF_XET_CACHE HF_HUB_DISABLE_XET HF_HUB_ENABLE_HF_TRANSFER
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

dependency_args=()
if [[ -n "${DEPENDENCY}" ]]; then
  dependency_args=(--dependency="${DEPENDENCY}")
fi

job_id="$(
  sbatch --parsable \
    "${dependency_args[@]}" \
    --job-name="${JOB_NAME}" \
    --partition="${PARTITION}" \
    --gres="gpu:${GPUS}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --output="${OUTPUT_DIR}/slurm-%j.out" \
    --error="${OUTPUT_DIR}/slurm-%j.err" \
    --export=ALL \
    <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "job_name=${SLURM_JOB_NAME}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "completions_path=${COMPLETIONS_PATH}"
echo "save_path=${SAVE_PATH}"
echo "summary_path=${SUMMARY_PATH}"
echo "classifier=${CLASSIFIER}"
echo "hash_dir=${HASH_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

cmd=(python llada2_benchmarks.py harmbench-official-eval)
cmd+=(--behaviors-path "${BEHAVIORS_PATH}")
cmd+=(--completions-path "${COMPLETIONS_PATH}")
cmd+=(--save-path "${SAVE_PATH}")
cmd+=(--summary-path "${SUMMARY_PATH}")
cmd+=(--classifier "${CLASSIFIER}")
cmd+=(--classifier-device-map "${CLASSIFIER_DEVICE_MAP}")
cmd+=(--classifier-torch-dtype "${CLASSIFIER_TORCH_DTYPE}")
cmd+=(--classifier-batch-size "${CLASSIFIER_BATCH_SIZE}")
cmd+=(--classifier-max-input-length "${CLASSIFIER_MAX_INPUT_LENGTH}")
cmd+=(--num-tokens "${NUM_TOKENS}")
cmd+=(--hash-dir "${HASH_DIR}")
cmd+=(--log-every "${LOG_EVERY}")
if [[ "${CLASSIFIER_LOCAL_FILES_ONLY}" == "1" ]]; then
  cmd+=(--classifier-local-files-only)
fi

printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
SBATCH
)"

echo "Submitted job ${job_id}"
echo "Logs: ${OUTPUT_DIR}/slurm-${job_id}.out"
echo "Results: ${SAVE_PATH}"
echo "Summary: ${SUMMARY_PATH}"
