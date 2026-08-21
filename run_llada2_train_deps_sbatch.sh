#!/usr/bin/env bash
set -euo pipefail

# Install Python dependencies needed for LLaDA2 training through Slurm.

PARTITION="${PARTITION:-rtx6000pro}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-24G}"
TIME="${TIME:-01:30:00}"
JOB_NAME="${JOB_NAME:-llada2_train_deps}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/llada2_train_deps/$(date +%Y%m%d_%H%M%S)}"
VENV="${VENV:-/home/jwj/work/.venv}"

HF_HOME="${LLADA_HF_HOME:-/home/jwj/work/.cache/huggingface}"
HF_HUB_CACHE="${LLADA_HF_HUB_CACHE:-${HF_HOME}/hub}"
HF_XET_CACHE="${LLADA_HF_XET_CACHE:-${HF_HOME}/xet}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_XET_CACHE}"

export VENV HF_HOME HF_HUB_CACHE HF_XET_CACHE HF_HUB_DISABLE_XET

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

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -r requirements-llada2.txt
SBATCH
)"

echo "Submitted job ${job_id}"
echo "Logs: ${OUTPUT_DIR}/slurm-${job_id}.out"
