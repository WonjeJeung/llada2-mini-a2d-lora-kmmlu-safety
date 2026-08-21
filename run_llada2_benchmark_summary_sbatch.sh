#!/usr/bin/env bash
set -euo pipefail

# Summarize benchmark JSONL shards through Slurm.

PARTITION="${PARTITION:-rtx6000pro}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
MEM="${MEM:-8G}"
TIME="${TIME:-00:20:00}"
JOB_NAME="${JOB_NAME:-llada2_bench_summary}"
INPUT_GLOB="${INPUT_GLOB:-}"
OUTPUT="${OUTPUT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/llada2_benchmark_summary/$(date +%Y%m%d_%H%M%S)}"
DEPENDENCY="${DEPENDENCY:-}"
VENV="${VENV:-/home/jwj/work/.venv}"

if [[ -z "${INPUT_GLOB}" || -z "${OUTPUT}" ]]; then
  echo "Set INPUT_GLOB=... and OUTPUT=..." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "$(dirname "${OUTPUT}")"

export INPUT_GLOB OUTPUT VENV OUTPUT_DIR

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
    --output="${OUTPUT_DIR}/slurm-%j.out" \
    --error="${OUTPUT_DIR}/slurm-%j.err" \
    --export=ALL \
    <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [[ -n "${VENV}" ]]; then
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
fi

python llada2_benchmarks.py summarize \
  --input-glob "${INPUT_GLOB}" \
  --output "${OUTPUT}"
SBATCH
)"

echo "Submitted job ${job_id}"
echo "Logs: ${OUTPUT_DIR}/slurm-${job_id}.out"
echo "Summary: ${OUTPUT}"
