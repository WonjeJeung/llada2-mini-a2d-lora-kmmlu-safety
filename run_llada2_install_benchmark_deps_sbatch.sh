#!/usr/bin/env bash
set -euo pipefail

# Install benchmark-side Python dependencies through Slurm.

PARTITION="${PARTITION:-rtx6000pro}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-16G}"
TIME="${TIME:-01:00:00}"
JOB_NAME="${JOB_NAME:-llada2_bench_deps}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/llada2_bench_deps/$(date +%Y%m%d_%H%M%S)}"
VENV="${VENV:-/home/jwj/work/.venv}"

mkdir -p "${OUTPUT_DIR}"
export VENV

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

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -r requirements-llada2.txt
python -m spacy download en_core_web_sm
SBATCH
)"

echo "Submitted job ${job_id}"
echo "Logs: ${OUTPUT_DIR}/slurm-${job_id}.out"
