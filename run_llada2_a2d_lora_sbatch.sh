#!/usr/bin/env bash
set -euo pipefail

# Submit LLaDA2.0-mini A2D-style LoRA training through Slurm.

PARTITION="${PARTITION:-rtx6000pro}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
MEM="${MEM:-160G}"
TIME="${TIME:-24:00:00}"
JOB_NAME="${JOB_NAME:-llada2_a2d_lora}"
RUN_DIR="${RUN_DIR:-runs/llada2_a2d_lora/$(date +%Y%m%d_%H%M%S)}"
VENV="${VENV:-/home/jwj/work/.venv}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
DEPENDENCY="${DEPENDENCY:-}"

MODEL="${MODEL:-inclusionAI/LLaDA2.0-mini}"
TRAIN_FILE="${TRAIN_FILE:-data/train/llada2_a2d_kmmlu30000_beaver30000_seed42/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/adapter}"
RESUME_ADAPTER="${RESUME_ADAPTER:-}"
SYSTEM="${SYSTEM:-}"

EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
MASK_EPSILON="${MASK_EPSILON:-0.001}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-query_key_value,dense}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-0}"

HF_HOME="${LLADA_HF_HOME:-/home/jwj/work/.cache/huggingface}"
HF_HUB_CACHE="${LLADA_HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_XET_CACHE="${LLADA_HF_XET_CACHE:-${HF_HOME}/xet}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "${RUN_DIR}" "${OUTPUT_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}"

export VENV INSTALL_DEPS MODEL TRAIN_FILE OUTPUT_DIR RESUME_ADAPTER SYSTEM
export EPOCHS MAX_STEPS PER_DEVICE_BATCH_SIZE GRAD_ACCUM_STEPS
export LEARNING_RATE WEIGHT_DECAY WARMUP_RATIO WARMUP_STEPS LR_SCHEDULER_TYPE
export MAX_GRAD_NORM MAX_LENGTH MAX_RESPONSE_LENGTH BLOCK_LENGTH MASK_EPSILON
export LORA_R LORA_ALPHA LORA_DROPOUT LORA_TARGET_MODULES
export TORCH_DTYPE LOCAL_FILES_ONLY GRADIENT_CHECKPOINTING SEED NUM_WORKERS
export LOG_EVERY SAVE_EVERY
export HF_HOME HF_HUB_CACHE HF_XET_CACHE HF_HUB_DISABLE_XET
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
    --output="${RUN_DIR}/slurm-%j.out" \
    --error="${RUN_DIR}/slurm-%j.err" \
    --export=ALL \
    <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "job_name=${SLURM_JOB_NAME}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "model=${MODEL}"
echo "train_file=${TRAIN_FILE}"
echo "output_dir=${OUTPUT_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  python -m pip install -r requirements-llada2.txt
fi

cmd=(python llada2_a2d_lora_train.py)
cmd+=(--model "${MODEL}")
cmd+=(--train-file "${TRAIN_FILE}")
cmd+=(--output-dir "${OUTPUT_DIR}")
cmd+=(--epochs "${EPOCHS}")
cmd+=(--max-steps "${MAX_STEPS}")
cmd+=(--per-device-batch-size "${PER_DEVICE_BATCH_SIZE}")
cmd+=(--grad-accum-steps "${GRAD_ACCUM_STEPS}")
cmd+=(--learning-rate "${LEARNING_RATE}")
cmd+=(--weight-decay "${WEIGHT_DECAY}")
cmd+=(--warmup-ratio "${WARMUP_RATIO}")
cmd+=(--warmup-steps "${WARMUP_STEPS}")
cmd+=(--lr-scheduler-type "${LR_SCHEDULER_TYPE}")
cmd+=(--max-grad-norm "${MAX_GRAD_NORM}")
cmd+=(--max-length "${MAX_LENGTH}")
cmd+=(--max-response-length "${MAX_RESPONSE_LENGTH}")
cmd+=(--block-length "${BLOCK_LENGTH}")
cmd+=(--mask-epsilon "${MASK_EPSILON}")
cmd+=(--lora-r "${LORA_R}")
cmd+=(--lora-alpha "${LORA_ALPHA}")
cmd+=(--lora-dropout "${LORA_DROPOUT}")
cmd+=(--lora-target-modules "${LORA_TARGET_MODULES}")
cmd+=(--torch-dtype "${TORCH_DTYPE}")
cmd+=(--seed "${SEED}")
cmd+=(--num-workers "${NUM_WORKERS}")
cmd+=(--log-every "${LOG_EVERY}")
cmd+=(--save-every "${SAVE_EVERY}")

if [[ -n "${RESUME_ADAPTER}" ]]; then
  cmd+=(--resume-adapter "${RESUME_ADAPTER}")
fi
if [[ -n "${SYSTEM}" ]]; then
  cmd+=(--system "${SYSTEM}")
fi
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  cmd+=(--local-files-only)
fi
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  cmd+=(--gradient-checkpointing)
fi

printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
SBATCH
)"

echo "Submitted job ${job_id}"
echo "Logs: ${RUN_DIR}/slurm-${job_id}.out"
echo "Adapter: ${OUTPUT_DIR}"
