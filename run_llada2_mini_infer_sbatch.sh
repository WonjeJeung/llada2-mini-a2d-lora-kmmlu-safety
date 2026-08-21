#!/usr/bin/env bash
set -euo pipefail

# Submit LLaDA2.0-mini inference through Slurm.
#
# Example:
#   PROMPT="한국어로 diffusion language model 설명해줘" \
#   GPUS=1 PARTITION=rtx6000pro \
#   ./run_llada2_mini_infer_sbatch.sh

PARTITION="${PARTITION:-rtx6000pro}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-12:00:00}"
JOB_NAME="${JOB_NAME:-llada2_mini_infer}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/llada2_mini_infer/$(date +%Y%m%d_%H%M%S)}"
DEPENDENCY="${DEPENDENCY:-}"

MODEL="${MODEL:-inclusionAI/LLaDA2.0-mini}"
ADAPTER="${ADAPTER:-}"
PROMPT="${PROMPT:-}"
SYSTEM="${SYSTEM:-}"
MESSAGES_JSON="${MESSAGES_JSON:-}"
RAW_PROMPT="${RAW_PROMPT:-0}"

GEN_LENGTH="${GEN_LENGTH:-512}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
STEPS="${STEPS:-32}"
THRESHOLD="${THRESHOLD:-0.95}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-}"
TOP_K="${TOP_K:-}"
EOS_EARLY_STOP="${EOS_EARLY_STOP:-0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
SEED="${SEED:-}"

VENV="${VENV:-}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"

HF_HOME="${LLADA_HF_HOME:-/home/jwj/work/.cache/huggingface}"
HF_HUB_CACHE="${LLADA_HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_XET_CACHE="${LLADA_HF_XET_CACHE:-${HF_HOME}/xet}"
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-64}"
HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

if [[ -z "${PROMPT}" && -z "${MESSAGES_JSON}" ]]; then
  echo "Set PROMPT=... or MESSAGES_JSON=... before submitting." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}"

export MODEL ADAPTER PROMPT SYSTEM MESSAGES_JSON RAW_PROMPT
export GEN_LENGTH BLOCK_LENGTH STEPS THRESHOLD TEMPERATURE TOP_P TOP_K
export EOS_EARLY_STOP DEVICE_MAP TORCH_DTYPE LOCAL_FILES_ONLY SEED
export VENV INSTALL_DEPS OUTPUT_DIR
export HF_HOME HF_HUB_CACHE HF_XET_CACHE
export HF_XET_HIGH_PERFORMANCE HF_XET_NUM_CONCURRENT_RANGE_GETS
export HF_HUB_DOWNLOAD_TIMEOUT
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
echo "workdir=${SLURM_SUBMIT_DIR}"
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

cmd=(python llada2_mini_infer.py)
cmd+=(--model "${MODEL}")
cmd+=(--gen-length "${GEN_LENGTH}")
cmd+=(--block-length "${BLOCK_LENGTH}")
cmd+=(--steps "${STEPS}")
cmd+=(--threshold "${THRESHOLD}")
cmd+=(--temperature "${TEMPERATURE}")
cmd+=(--device-map "${DEVICE_MAP}")
cmd+=(--torch-dtype "${TORCH_DTYPE}")

if [[ -n "${ADAPTER}" ]]; then
  cmd+=(--adapter "${ADAPTER}")
fi

if [[ -n "${MESSAGES_JSON}" ]]; then
  cmd+=(--messages-json "${MESSAGES_JSON}")
else
  cmd+=(--prompt "${PROMPT}")
fi

if [[ -n "${SYSTEM}" ]]; then
  cmd+=(--system "${SYSTEM}")
fi

if [[ "${RAW_PROMPT}" == "1" ]]; then
  cmd+=(--raw-prompt)
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

if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  cmd+=(--local-files-only)
fi

if [[ -n "${SEED}" ]]; then
  cmd+=(--seed "${SEED}")
fi

printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" | tee "${OUTPUT_DIR}/generation.txt"
SBATCH
)"

echo "Submitted job ${job_id}"
echo "Logs: ${OUTPUT_DIR}/slurm-${job_id}.out"
echo "Text: ${OUTPUT_DIR}/generation.txt"
