#!/usr/bin/env bash
set -euo pipefail

# Convert HarmBench SAVE_TEXT JSONL shards to official completions JSON via Slurm.

PARTITION="${PARTITION:-rtx6000pro}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
MEM="${MEM:-8G}"
TIME="${TIME:-00:20:00}"
JOB_NAME="${JOB_NAME:-harmbench_export_completions}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/harmbench_export_completions/$(date +%Y%m%d_%H%M%S)}"
INPUT_GLOB="${INPUT_GLOB:-}"
OUTPUT="${OUTPUT:-}"
INCLUDE_TEST_CASE="${INCLUDE_TEST_CASE:-1}"
DEPENDENCY="${DEPENDENCY:-}"
VENV="${VENV:-/home/jwj/work/.venv}"

if [[ -z "${INPUT_GLOB}" || -z "${OUTPUT}" ]]; then
  echo "Set INPUT_GLOB=... and OUTPUT=..." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "$(dirname "${OUTPUT}")"
export INPUT_GLOB OUTPUT INCLUDE_TEST_CASE VENV

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

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

cmd=(python llada2_benchmarks.py harmbench-export-completions)
cmd+=(--input-glob "${INPUT_GLOB}")
cmd+=(--output "${OUTPUT}")
if [[ "${INCLUDE_TEST_CASE}" == "1" ]]; then
  cmd+=(--include-test-case)
fi

printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
SBATCH
)"

echo "Submitted job ${job_id}"
echo "Logs: ${OUTPUT_DIR}/slurm-${job_id}.out"
echo "Completions: ${OUTPUT}"
