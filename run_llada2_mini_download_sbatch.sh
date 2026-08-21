#!/usr/bin/env bash
set -euo pipefail

# Download inclusionAI/LLaDA2.0-mini through Slurm without tying up a GPU.

PARTITION="${PARTITION:-rtx6000pro}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-96G}"
TIME="${TIME:-08:00:00}"
JOB_NAME="${JOB_NAME:-llada2_mini_download}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/llada2_mini_download/$(date +%Y%m%d_%H%M%S)}"

MODEL="${MODEL:-inclusionAI/LLaDA2.0-mini}"
REPO_TYPE="${REPO_TYPE:-}"
REVISION="${REVISION:-}"
MAX_WORKERS="${MAX_WORKERS:-16}"
ALLOW_PATTERNS="${ALLOW_PATTERNS:-}"
VENV="${VENV:-/home/jwj/work/.venv}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

HF_HOME="${LLADA_HF_HOME:-/home/jwj/work/.cache/huggingface}"
HF_HUB_CACHE="${LLADA_HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_XET_CACHE="${LLADA_HF_XET_CACHE:-${HF_HOME}/xet}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}"

export MODEL REPO_TYPE REVISION MAX_WORKERS ALLOW_PATTERNS VENV INSTALL_DEPS OUTPUT_DIR
export HF_HOME HF_HUB_CACHE HF_XET_CACHE
export HF_XET_HIGH_PERFORMANCE HF_XET_NUM_CONCURRENT_RANGE_GETS
export HF_HUB_DOWNLOAD_TIMEOUT HF_HUB_DISABLE_XET HF_HUB_ENABLE_HF_TRANSFER

job_id="$(
  sbatch --parsable \
    --job-name="${JOB_NAME}" \
    --partition="${PARTITION}" \
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
echo "workdir=${SLURM_SUBMIT_DIR}"
echo "model=${MODEL}"
echo "output_dir=${OUTPUT_DIR}"
echo "HF_HOME=${HF_HOME}"
echo "HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE}"
echo "HF_XET_NUM_CONCURRENT_RANGE_GETS=${HF_XET_NUM_CONCURRENT_RANGE_GETS}"
echo "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET}"
echo "HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER}"

if [[ -n "${VENV}" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  python -m pip install -r requirements-llada2.txt
fi

cmd=(python download_hf_snapshot.py)
cmd+=(--repo-id "${MODEL}")
cmd+=(--max-workers "${MAX_WORKERS}")

if [[ -n "${REPO_TYPE}" ]]; then
  cmd+=(--repo-type "${REPO_TYPE}")
fi

if [[ -n "${REVISION}" ]]; then
  cmd+=(--revision "${REVISION}")
fi

if [[ -n "${ALLOW_PATTERNS}" ]]; then
  read -r -a allow_patterns <<< "${ALLOW_PATTERNS}"
  cmd+=(--allow-patterns "${allow_patterns[@]}")
fi

printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"

du -sh "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}" || true
SBATCH
)"

echo "Submitted job ${job_id}"
echo "Logs: ${OUTPUT_DIR}/slurm-${job_id}.out"
echo "Err:  ${OUTPUT_DIR}/slurm-${job_id}.err"
