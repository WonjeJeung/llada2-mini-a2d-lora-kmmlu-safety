#!/usr/bin/env bash
set -euo pipefail

# Build KMMLU + A2D safety JSONL training data through Slurm.

PARTITION="${PARTITION:-rtx6000pro}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-48G}"
TIME="${TIME:-03:00:00}"
JOB_NAME="${JOB_NAME:-llada2_a2d_data}"
RUN_DIR="${RUN_DIR:-runs/llada2_a2d_data/$(date +%Y%m%d_%H%M%S)}"
VENV="${VENV:-/home/jwj/work/.venv}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
DEPENDENCY="${DEPENDENCY:-}"

KMMLU_LOCAL_DIR="${KMMLU_LOCAL_DIR:-data/benchmarks/kmmlu}"
KMMLU_TRAIN_SIZE="${KMMLU_TRAIN_SIZE:-30000}"
BEAVERTAILS_DATASET="${BEAVERTAILS_DATASET:-PKU-Alignment/BeaverTails}"
BEAVERTAILS_SPLIT="${BEAVERTAILS_SPLIT:-30k_train}"
SAFETY_SAMPLES="${SAFETY_SAMPLES:-30000}"
SAFETY_HARMFUL_RATIO="${SAFETY_HARMFUL_RATIO:-0.5}"
SEED="${SEED:-42}"
NO_SHUFFLE_KMMLU="${NO_SHUFFLE_KMMLU:-0}"
NO_SHUFFLE_TRAIN="${NO_SHUFFLE_TRAIN:-0}"
OUTPUT="${OUTPUT:-data/train/llada2_a2d_kmmlu${KMMLU_TRAIN_SIZE}_beaver${SAFETY_SAMPLES}_seed${SEED}/train.jsonl}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-data/train/llada2_a2d_kmmlu${KMMLU_TRAIN_SIZE}_beaver${SAFETY_SAMPLES}_seed${SEED}/summary.json}"
KMMLU_HOLDOUT_OUTPUT="${KMMLU_HOLDOUT_OUTPUT:-data/train/llada2_a2d_kmmlu${KMMLU_TRAIN_SIZE}_beaver${SAFETY_SAMPLES}_seed${SEED}/kmmlu_holdout.jsonl}"
OVERWRITE="${OVERWRITE:-1}"

HF_HOME="${LLADA_HF_HOME:-/home/jwj/work/.cache/huggingface}"
HF_HUB_CACHE="${LLADA_HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_XET_CACHE="${LLADA_HF_XET_CACHE:-${HF_HOME}/xet}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "${RUN_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}"

export VENV INSTALL_DEPS
export KMMLU_LOCAL_DIR KMMLU_TRAIN_SIZE BEAVERTAILS_DATASET BEAVERTAILS_SPLIT
export SAFETY_SAMPLES SAFETY_HARMFUL_RATIO SEED OUTPUT SUMMARY_OUTPUT
export KMMLU_HOLDOUT_OUTPUT OVERWRITE NO_SHUFFLE_KMMLU NO_SHUFFLE_TRAIN
export HF_HOME HF_HUB_CACHE HF_XET_CACHE HF_HUB_DISABLE_XET

dependency_args=()
if [[ -n "${DEPENDENCY}" ]]; then
  dependency_args=(--dependency="${DEPENDENCY}")
fi

job_id="$(
  sbatch --parsable \
    "${dependency_args[@]}" \
    --job-name="${JOB_NAME}" \
    --partition="${PARTITION}" \
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
echo "output=${OUTPUT}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  python -m pip install -r requirements-llada2.txt
fi

cmd=(python prepare_llada2_a2d_data.py)
cmd+=(--output "${OUTPUT}")
cmd+=(--summary-output "${SUMMARY_OUTPUT}")
cmd+=(--kmmlu-local-dir "${KMMLU_LOCAL_DIR}")
cmd+=(--kmmlu-train-size "${KMMLU_TRAIN_SIZE}")
cmd+=(--kmmlu-holdout-output "${KMMLU_HOLDOUT_OUTPUT}")
cmd+=(--beavertails-dataset "${BEAVERTAILS_DATASET}")
cmd+=(--beavertails-split "${BEAVERTAILS_SPLIT}")
cmd+=(--safety-samples "${SAFETY_SAMPLES}")
cmd+=(--safety-harmful-ratio "${SAFETY_HARMFUL_RATIO}")
cmd+=(--seed "${SEED}")
cmd+=(--hf-cache-dir "${HF_HOME}")

if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi
if [[ "${NO_SHUFFLE_KMMLU}" == "1" ]]; then
  cmd+=(--no-shuffle-kmmlu)
fi
if [[ "${NO_SHUFFLE_TRAIN}" == "1" ]]; then
  cmd+=(--no-shuffle-train)
fi

printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
SBATCH
)"

echo "Submitted job ${job_id}"
echo "Logs: ${RUN_DIR}/slurm-${job_id}.out"
echo "Train JSONL: ${OUTPUT}"
echo "Summary: ${SUMMARY_OUTPUT}"
