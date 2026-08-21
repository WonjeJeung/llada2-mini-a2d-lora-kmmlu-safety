# LLaDA2.0-mini A2D LoRA

Clean release repository for the LLaDA2.0-mini KMMLU + A2D safety-alignment LoRA experiment.

The trained LoRA adapter is hosted on Hugging Face:

https://huggingface.co/Wonje/llada2-mini-a2d-lora-kmmlu-safety

This repository contains reproducibility code and Slurm wrappers only. Evaluation results, raw logs, cached datasets, model weights, checkpoints, and local run artifacts are intentionally not included.

## Files

- `llada2_mini_infer.py`: LLaDA2.0-mini inference CLI with adapter support.
- `llada2_benchmarks.py`: KMMLU and HarmBench generation/evaluation driver.
- `llada2_a2d_lora_train.py`: LoRA training script.
- `prepare_llada2_a2d_data.py`: KMMLU + A2D data preparation script.
- `run_*_sbatch.sh`: Slurm wrappers used on the cluster.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-llada2.txt
```

## Inference

The cluster setup uses Slurm:

```bash
ADAPTER=Wonje/llada2-mini-a2d-lora-kmmlu-safety \
PROMPT="안녕" \
LOCAL_FILES_ONLY=0 \
EOS_EARLY_STOP=1 \
./run_llada2_mini_infer_sbatch.sh
```

Direct execution:

```bash
python llada2_mini_infer.py \
  --model inclusionAI/LLaDA2.0-mini \
  --adapter Wonje/llada2-mini-a2d-lora-kmmlu-safety \
  --prompt "안녕" \
  --gen-length 512 \
  --block-length 32 \
  --steps 32 \
  --eos-early-stop
```

## Training

```bash
./run_llada2_a2d_prepare_data_sbatch.sh
./run_llada2_a2d_lora_sbatch.sh
```

## Evaluation

```bash
BENCHMARK=kmmlu \
ADAPTER=Wonje/llada2-mini-a2d-lora-kmmlu-safety \
./run_llada2_benchmark_sbatch.sh

BENCHMARK=harmbench \
ADAPTER=Wonje/llada2-mini-a2d-lora-kmmlu-safety \
./run_harmbench_export_completions_sbatch.sh

./run_harmbench_official_eval_sbatch.sh
```
