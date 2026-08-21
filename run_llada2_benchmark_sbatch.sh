#!/usr/bin/env bash
set -euo pipefail

# Submit LLaDA2.0-mini benchmark jobs through Slurm.
#
# Examples:
#   TASK=kmmlu LIMIT_PER_SUBJECT=2 ./run_llada2_benchmark_sbatch.sh
#   TASK=harmbench LIMIT=10 GEN_LENGTH=128 ./run_llada2_benchmark_sbatch.sh
#   TASK=kmmlu SHARDS=4 MAX_PARALLEL=2 LIMIT_PER_SUBJECT=0 ./run_llada2_benchmark_sbatch.sh

TASK="${TASK:-kmmlu}"
PARTITION="${PARTITION:-rtx6000pro}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-12:00:00}"
JOB_NAME="${JOB_NAME:-llada2_${TASK}_bench}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/llada2_${TASK}_bench/$(date +%Y%m%d_%H%M%S)}"

MODEL="${MODEL:-inclusionAI/LLaDA2.0-mini}"
ADAPTER="${ADAPTER:-}"
DATA_DIR="${DATA_DIR:-data/benchmarks}"
SYSTEM="${SYSTEM:-}"
GEN_LENGTH="${GEN_LENGTH:-128}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
STEPS="${STEPS:-32}"
THRESHOLD="${THRESHOLD:-0.95}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"
EOS_EARLY_STOP="${EOS_EARLY_STOP:-1}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
MODEL_LOCAL_FILES_ONLY="${MODEL_LOCAL_FILES_ONLY:-1}"
SEED="${SEED:-}"
LIMIT="${LIMIT:-0}"
LOG_EVERY="${LOG_EVERY:-25}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
OVERWRITE="${OVERWRITE:-1}"
SHARDS="${SHARDS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
DEPENDENCY="${DEPENDENCY:-}"

KMMLU_REPO="${KMMLU_REPO:-HAERAE-HUB/KMMLU}"
KMMLU_LOCAL_DIR="${KMMLU_LOCAL_DIR:-${DATA_DIR}/kmmlu}"
SUBJECTS="${SUBJECTS:-}"
LIMIT_PER_SUBJECT="${LIMIT_PER_SUBJECT:-0}"
KMMLU_GEN_LENGTH="${KMMLU_GEN_LENGTH:-4}"

HARMBENCH_CSV="${HARMBENCH_CSV:-}"
HARMBENCH_URL="${HARMBENCH_URL:-https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_test.csv}"
SAVE_TEXT="${SAVE_TEXT:-0}"

VENV="${VENV:-/home/jwj/work/.venv}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

HF_HOME="${LLADA_HF_HOME:-/home/jwj/work/.cache/huggingface}"
HF_HUB_CACHE="${LLADA_HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_XET_CACHE="${LLADA_HF_XET_CACHE:-${HF_HOME}/xet}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

if [[ "${TASK}" != "kmmlu" && "${TASK}" != "harmbench" ]]; then
  echo "TASK must be kmmlu or harmbench." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}"

export TASK MODEL ADAPTER DATA_DIR SYSTEM
export GEN_LENGTH BLOCK_LENGTH STEPS THRESHOLD TEMPERATURE TOP_P TOP_K
export EOS_EARLY_STOP DEVICE_MAP TORCH_DTYPE MODEL_LOCAL_FILES_ONLY
export SEED LIMIT LOG_EVERY DOWNLOAD_WORKERS OVERWRITE SHARDS
export KMMLU_REPO KMMLU_LOCAL_DIR SUBJECTS LIMIT_PER_SUBJECT KMMLU_GEN_LENGTH
export HARMBENCH_CSV HARMBENCH_URL SAVE_TEXT
export VENV INSTALL_DEPS OUTPUT_DIR
export HF_HOME HF_HUB_CACHE HF_XET_CACHE HF_HUB_DISABLE_XET HF_HUB_ENABLE_HF_TRANSFER
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

array_args=()
output_pattern="${OUTPUT_DIR}/slurm-%j.out"
error_pattern="${OUTPUT_DIR}/slurm-%j.err"
if (( SHARDS > 1 )); then
  array_args=(--array="0-$((SHARDS - 1))%${MAX_PARALLEL}")
  output_pattern="${OUTPUT_DIR}/slurm-%A_%a.out"
  error_pattern="${OUTPUT_DIR}/slurm-%A_%a.err"
fi

dependency_args=()
if [[ -n "${DEPENDENCY}" ]]; then
  dependency_args=(--dependency="${DEPENDENCY}")
fi

job_id="$(
  sbatch --parsable \
    "${array_args[@]}" \
    "${dependency_args[@]}" \
    --job-name="${JOB_NAME}" \
    --partition="${PARTITION}" \
    --gres="gpu:${GPUS}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --output="${output_pattern}" \
    --error="${error_pattern}" \
    --export=ALL \
    <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "array_task=${SLURM_ARRAY_TASK_ID:-0}"
echo "job_name=${SLURM_JOB_NAME}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "task=${TASK}"
echo "model=${MODEL}"
echo "adapter=${ADAPTER}"
echo "output_dir=${OUTPUT_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

if [[ -n "${VENV}" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  python -m pip install -r requirements-llada2.txt
fi

shard_index="${SLURM_ARRAY_TASK_ID:-0}"
cmd=(python llada2_benchmarks.py "${TASK}")
cmd+=(--model "${MODEL}")
cmd+=(--output-dir "${OUTPUT_DIR}")
cmd+=(--data-dir "${DATA_DIR}")
cmd+=(--gen-length "${GEN_LENGTH}")
cmd+=(--block-length "${BLOCK_LENGTH}")
cmd+=(--steps "${STEPS}")
cmd+=(--threshold "${THRESHOLD}")
cmd+=(--temperature "${TEMPERATURE}")
cmd+=(--device-map "${DEVICE_MAP}")
cmd+=(--torch-dtype "${TORCH_DTYPE}")
cmd+=(--limit "${LIMIT}")
cmd+=(--log-every "${LOG_EVERY}")
cmd+=(--download-workers "${DOWNLOAD_WORKERS}")
cmd+=(--shard-index "${shard_index}")
cmd+=(--shard-count "${SHARDS}")

if [[ -n "${ADAPTER}" ]]; then
  cmd+=(--adapter "${ADAPTER}")
fi

if [[ -n "${SYSTEM}" ]]; then
  cmd+=(--system "${SYSTEM}")
fi
if [[ -n "${TOP_P}" ]]; then
  cmd+=(--top-p "${TOP_P}")
fi
if [[ -n "${TOP_K}" ]]; then
  cmd+=(--top-k "${TOP_K}")
fi
if [[ "${EOS_EARLY_STOP}" == "1" ]]; then
  cmd+=(--eos-early-stop)
fi
if [[ "${MODEL_LOCAL_FILES_ONLY}" == "1" ]]; then
  cmd+=(--model-local-files-only)
fi
if [[ -n "${SEED}" ]]; then
  cmd+=(--seed "${SEED}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

if [[ "${TASK}" == "kmmlu" ]]; then
  cmd+=(--kmmlu-repo "${KMMLU_REPO}")
  cmd+=(--kmmlu-local-dir "${KMMLU_LOCAL_DIR}")
  cmd+=(--limit-per-subject "${LIMIT_PER_SUBJECT}")
  cmd+=(--kmmlu-gen-length "${KMMLU_GEN_LENGTH}")
  if [[ -n "${SUBJECTS}" ]]; then
    cmd+=(--subjects "${SUBJECTS}")
  fi
else
  cmd+=(--harmbench-url "${HARMBENCH_URL}")
  if [[ -n "${HARMBENCH_CSV}" ]]; then
    cmd+=(--harmbench-csv "${HARMBENCH_CSV}")
  fi
  if [[ "${SAVE_TEXT}" == "1" ]]; then
    cmd+=(--save-text)
  fi
fi

printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
SBATCH
)"

echo "Submitted job ${job_id}"
echo "Logs: ${OUTPUT_DIR}/slurm-*.out"
echo "JSONL: ${OUTPUT_DIR}/${TASK}-shard-*.jsonl"
echo "Summarize after completion:"
echo "  INPUT_GLOB='${OUTPUT_DIR}/${TASK}-shard-*.jsonl' OUTPUT='${OUTPUT_DIR}/summary.json' ./run_llada2_benchmark_summary_sbatch.sh"
