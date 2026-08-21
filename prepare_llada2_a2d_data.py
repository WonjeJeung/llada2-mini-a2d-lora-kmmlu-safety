#!/usr/bin/env python3
"""Build mixed KMMLU + A2D safety data for LLaDA2.0-mini LoRA training."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LETTERS = ("A", "B", "C", "D")
DEFAULT_KMMLU_LOCAL_DIR = "data/benchmarks/kmmlu"
DEFAULT_BEAVERTAILS_DATASET = "PKU-Alignment/BeaverTails"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def first_present(row: dict[str, str], names: Iterable[str]) -> str | None:
    lower_to_key = {key.lower(): key for key in row}
    for name in names:
        key = lower_to_key.get(name.lower())
        if key is not None:
            return row.get(key)
    return None


def normalize_gold_answer(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    cleaned = raw.upper().strip(" .):：")
    if cleaned in LETTERS:
        return cleaned
    if re.fullmatch(r"[1-4]", cleaned):
        return LETTERS[int(cleaned) - 1]
    if re.fullmatch(r"[0-3]", cleaned):
        return LETTERS[int(cleaned)]
    match = re.search(r"\b([ABCD])\b", cleaned)
    if match:
        return match.group(1)
    return None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def kmmlu_subject_from_file(path: Path) -> str:
    name = path.name
    if name.endswith("-test.csv"):
        name = name[: -len("-test.csv")]
    return name


def kmmlu_choices(row: dict[str, str]) -> dict[str, str]:
    candidates = [
        ("A", "B", "C", "D"),
        ("a", "b", "c", "d"),
        ("option_a", "option_b", "option_c", "option_d"),
        ("choice_a", "choice_b", "choice_c", "choice_d"),
        ("1", "2", "3", "4"),
    ]
    for columns in candidates:
        values = [first_present(row, [col]) for col in columns]
        if all(value is not None for value in values):
            return {
                letter: str(value).strip()
                for letter, value in zip(LETTERS, values, strict=True)
            }
    raise KeyError(f"No A/B/C/D choice columns found. Columns={list(row)}")


def kmmlu_prompt(question: str, choices: dict[str, str]) -> str:
    return (
        "다음 객관식 문제의 정답을 A, B, C, D 중 하나로만 답하세요.\n\n"
        f"문제: {question}\n"
        f"A. {choices['A']}\n"
        f"B. {choices['B']}\n"
        f"C. {choices['C']}\n"
        f"D. {choices['D']}\n\n"
        "정답:"
    )


def iter_kmmlu_examples(kmmlu_local_dir: Path) -> list[dict[str, Any]]:
    files = sorted(kmmlu_local_dir.glob("data/*-test.csv"))
    if not files:
        raise FileNotFoundError(
            f"No KMMLU CSV files found under {kmmlu_local_dir}/data"
        )

    examples: list[dict[str, Any]] = []
    for path in files:
        subject = kmmlu_subject_from_file(path)
        for row_idx, row in enumerate(read_csv_rows(path)):
            question = first_present(row, ["question", "Question", "문제"])
            answer = normalize_gold_answer(
                first_present(row, ["answer", "Answer", "gold", "label", "target"])
            )
            if not question or answer is None:
                raise ValueError(f"Could not parse KMMLU row {path}:{row_idx}")
            choices = kmmlu_choices(row)
            examples.append(
                {
                    "id": f"kmmlu:{subject}:{row_idx}",
                    "source": "kmmlu",
                    "kind": "retain",
                    "loss_mode": "reconstruct",
                    "is_harmful": False,
                    "subject": subject,
                    "row_idx": row_idx,
                    "answer": answer,
                    "prompt": kmmlu_prompt(question.strip(), choices),
                    "response": answer,
                }
            )
    return examples


def load_beavertails(args: argparse.Namespace) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The datasets package is required for BeaverTails. "
            "Run ./run_llada2_train_deps_sbatch.sh first."
        ) from exc

    split_candidates = [args.beavertails_split, "30k_train", "train", "330k_train"]
    seen: set[str] = set()
    errors: list[str] = []
    for split in split_candidates:
        if split in seen:
            continue
        seen.add(split)
        try:
            return load_dataset(
                args.beavertails_dataset,
                split=split,
                cache_dir=args.hf_cache_dir,
            )
        except Exception as exc:  # pragma: no cover - depends on HF metadata.
            errors.append(f"{split}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Could not load BeaverTails with any known split. "
        + " | ".join(errors[-3:])
    )


def build_safety_examples(args: argparse.Namespace, rng: random.Random) -> list[dict[str, Any]]:
    if args.safety_samples <= 0:
        return []

    dataset = load_beavertails(args)
    safe: list[dict[str, Any]] = []
    harmful: list[dict[str, Any]] = []
    for row_idx, row in enumerate(dataset):
        prompt = str(row.get("prompt") or "").strip()
        response = str(row.get("response") or "").strip()
        if not prompt or not response:
            continue
        target = harmful if not bool(row.get("is_safe", True)) else safe
        target.append(
            {
                "id": f"beavertails:{row_idx}",
                "source": "beavertails",
                "kind": "harmful" if target is harmful else "retain",
                "loss_mode": "a2d_eos" if target is harmful else "reconstruct",
                "is_harmful": target is harmful,
                "row_idx": row_idx,
                "category": row.get("category"),
                "prompt": prompt,
                "response": response,
            }
        )

    harmful_target = round(args.safety_samples * args.safety_harmful_ratio)
    harmful_target = max(0, min(args.safety_samples, harmful_target))
    safe_target = args.safety_samples - harmful_target

    rng.shuffle(harmful)
    rng.shuffle(safe)
    picked = harmful[:harmful_target] + safe[:safe_target]
    if len(picked) < args.safety_samples:
        remaining = harmful[harmful_target:] + safe[safe_target:]
        rng.shuffle(remaining)
        picked.extend(remaining[: args.safety_samples - len(picked)])

    rng.shuffle(picked)
    if len(picked) < args.safety_samples:
        print(
            f"[prepare] WARNING requested safety_samples={args.safety_samples} "
            f"but only built {len(picked)}",
            flush=True,
        )
    return picked


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], *, overwrite: bool) -> int:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Set --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(row.get("source", "unknown") for row in rows)
    kind_counts = Counter(row.get("kind", "unknown") for row in rows)
    loss_counts = Counter(row.get("loss_mode", "unknown") for row in rows)
    kmmlu_subjects = Counter(
        row.get("subject", "unknown") for row in rows if row.get("source") == "kmmlu"
    )
    return {
        "total": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "kind_counts": dict(sorted(kind_counts.items())),
        "loss_mode_counts": dict(sorted(loss_counts.items())),
        "kmmlu_subject_counts": dict(sorted(kmmlu_subjects.items())),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--kmmlu-local-dir", default=DEFAULT_KMMLU_LOCAL_DIR)
    parser.add_argument("--kmmlu-train-size", type=int, default=30000)
    parser.add_argument("--kmmlu-holdout-output", default="")
    parser.add_argument("--beavertails-dataset", default=DEFAULT_BEAVERTAILS_DATASET)
    parser.add_argument("--beavertails-split", default="30k_train")
    parser.add_argument("--safety-samples", type=int, default=30000)
    parser.add_argument("--safety-harmful-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-cache-dir", default=os.environ.get("HF_HOME", ""))
    parser.add_argument(
        "--no-shuffle-kmmlu",
        action="store_true",
        help="Keep KMMLU's deterministic benchmark order before selecting examples.",
    )
    parser.add_argument(
        "--no-shuffle-train",
        action="store_true",
        help="Keep selected train rows in source order instead of shuffling the JSONL.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.kmmlu_train_size < 0:
        raise ValueError("--kmmlu-train-size must be non-negative")
    if args.safety_samples < 0:
        raise ValueError("--safety-samples must be non-negative")
    if not 0.0 <= args.safety_harmful_ratio <= 1.0:
        raise ValueError("--safety-harmful-ratio must be in [0, 1]")

    rng = random.Random(args.seed)

    kmmlu_examples = iter_kmmlu_examples(Path(args.kmmlu_local_dir))
    if not args.no_shuffle_kmmlu:
        rng.shuffle(kmmlu_examples)
    if args.kmmlu_train_size > len(kmmlu_examples):
        raise ValueError(
            f"Requested {args.kmmlu_train_size} KMMLU examples, "
            f"but only found {len(kmmlu_examples)}"
        )
    kmmlu_train = kmmlu_examples[: args.kmmlu_train_size]
    kmmlu_holdout = kmmlu_examples[args.kmmlu_train_size :]

    safety_examples = build_safety_examples(args, rng)
    train_rows = kmmlu_train + safety_examples
    if not args.no_shuffle_train:
        rng.shuffle(train_rows)

    output_path = Path(args.output)
    train_count = write_jsonl(output_path, train_rows, overwrite=args.overwrite)

    holdout_path = Path(args.kmmlu_holdout_output) if args.kmmlu_holdout_output else None
    holdout_count = 0
    if holdout_path is not None:
        holdout_count = write_jsonl(holdout_path, kmmlu_holdout, overwrite=args.overwrite)

    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else output_path.with_suffix(".summary.json")
    )
    summary = summarize(train_rows)
    summary.update(
        {
            "created_at": utc_now(),
            "output": str(output_path),
            "train_count": train_count,
            "kmmlu_available": len(kmmlu_examples),
            "kmmlu_train_size": len(kmmlu_train),
            "kmmlu_holdout_size": holdout_count,
            "kmmlu_holdout_output": str(holdout_path) if holdout_path else "",
            "beavertails_dataset": args.beavertails_dataset,
            "beavertails_split_requested": args.beavertails_split,
            "safety_samples_requested": args.safety_samples,
            "safety_harmful_ratio": args.safety_harmful_ratio,
            "seed": args.seed,
            "shuffle_kmmlu": not args.no_shuffle_kmmlu,
            "shuffle_train": not args.no_shuffle_train,
        }
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
