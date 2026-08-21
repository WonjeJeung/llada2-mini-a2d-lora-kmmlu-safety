#!/usr/bin/env python3
"""A2D-style LoRA training for inclusionAI/LLaDA2.0-mini."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from llada2_mini_infer import dtype_arg, token_id


DEFAULT_MODEL_ID = "inclusionAI/LLaDA2.0-mini"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


class JsonlTrainingDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: Path):
        self.path = path
        self.rows = read_jsonl(path)
        if not self.rows:
            raise ValueError(f"No training rows found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def as_token_list(ids: Any) -> list[int]:
    if isinstance(ids, torch.Tensor):
        return [int(value) for value in ids.flatten().tolist()]
    return [int(value) for value in ids]


class A2DCollator:
    def __init__(
        self,
        tokenizer: Any,
        *,
        mask_id: int,
        eos_id: int,
        max_length: int,
        max_response_length: int,
        mask_epsilon: float,
        system: str | None,
        seed: int,
    ):
        self.tokenizer = tokenizer
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.max_length = max_length
        self.max_response_length = max_response_length
        self.mask_epsilon = mask_epsilon
        self.system = system
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = eos_id

    def encode_pair(self, record: dict[str, Any]) -> tuple[list[int], list[int]]:
        prompt = str(record["prompt"])
        response = str(record["response"])
        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})

        prompt_ids = as_token_list(
            self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        full_ids = as_token_list(
            self.tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": response}],
                add_generation_prompt=False,
                tokenize=True,
            )
        )

        if full_ids[: len(prompt_ids)] == prompt_ids:
            response_ids = full_ids[len(prompt_ids) :]
        else:
            response_ids = as_token_list(
                self.tokenizer(response, add_special_tokens=False).input_ids
            )

        if self.max_response_length > 0:
            response_ids = response_ids[: self.max_response_length]
        if not response_ids:
            response_ids = [self.eos_id]

        if len(prompt_ids) + len(response_ids) > self.max_length:
            prompt_budget = self.max_length - len(response_ids)
            if prompt_budget < 1:
                response_ids = response_ids[: max(1, self.max_length - 1)]
                prompt_budget = self.max_length - len(response_ids)
            prompt_ids = prompt_ids[-prompt_budget:]

        return prompt_ids, response_ids

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded: list[tuple[dict[str, Any], list[int], list[int]]] = []
        for record in records:
            prompt_ids, response_ids = self.encode_pair(record)
            encoded.append((record, prompt_ids, response_ids))

        max_len = max(len(prompt_ids) + len(response_ids) for _, prompt_ids, response_ids in encoded)
        input_ids = torch.full((len(encoded), max_len), self.pad_id, dtype=torch.long)
        labels = torch.full((len(encoded), max_len), -100, dtype=torch.long)
        valid_mask = torch.zeros((len(encoded), max_len), dtype=torch.bool)
        harmful_mask = torch.zeros((len(encoded),), dtype=torch.bool)
        masked_token_counts = torch.zeros((len(encoded),), dtype=torch.long)

        for row_idx, (record, prompt_ids, response_ids) in enumerate(encoded):
            ids = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
            seq_len = ids.numel()
            prompt_len = len(prompt_ids)
            response_len = len(response_ids)
            input_ids[row_idx, :seq_len] = ids
            valid_mask[row_idx, :seq_len] = True
            harmful = bool(record.get("is_harmful", False)) or (
                record.get("loss_mode") == "a2d_eos"
            )
            harmful_mask[row_idx] = harmful

            mask_ratio = (
                (1.0 - self.mask_epsilon)
                * torch.rand((), generator=self.generator).item()
                + self.mask_epsilon
            )
            response_mask = torch.rand(response_len, generator=self.generator) < mask_ratio
            if not bool(response_mask.any()):
                forced = torch.randint(response_len, (1,), generator=self.generator).item()
                response_mask[forced] = True

            absolute_mask = torch.zeros(seq_len, dtype=torch.bool)
            absolute_mask[prompt_len:seq_len] = response_mask
            masked_positions = absolute_mask.nonzero(as_tuple=False).flatten()
            input_ids[row_idx, masked_positions] = self.mask_id
            labels[row_idx, masked_positions] = (
                self.eos_id if harmful else ids[masked_positions]
            )
            masked_token_counts[row_idx] = int(masked_positions.numel())

        return {
            "input_ids": input_ids,
            "labels": labels,
            "valid_mask": valid_mask,
            "harmful_mask": harmful_mask,
            "masked_token_counts": masked_token_counts,
        }


def make_block_attention_mask(
    valid_mask: torch.Tensor,
    *,
    block_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size, seq_len = valid_mask.shape
    device = valid_mask.device
    num_blocks = math.ceil(seq_len / block_length)
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=torch.bool))
    allowed = block_mask.repeat_interleave(block_length, 0).repeat_interleave(block_length, 1)
    allowed = allowed[:seq_len, :seq_len]
    allowed = allowed.unsqueeze(0).expand(batch_size, -1, -1).clone()
    allowed &= valid_mask[:, None, :]

    eye = torch.eye(seq_len, device=device, dtype=torch.bool).unsqueeze(0)
    allowed |= (~valid_mask)[:, :, None] & eye

    zeros = torch.zeros((batch_size, seq_len, seq_len), device=device, dtype=dtype)
    neg_inf = torch.full_like(zeros, float("-inf"))
    return torch.where(allowed, zeros, neg_inf).unsqueeze(1)


def common_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
    }
    token = os.environ.get("HF_TOKEN")
    if token:
        kwargs["token"] = token
    return kwargs


def add_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "The peft package is required for LoRA training. "
            "Run ./run_llada2_train_deps_sbatch.sh first."
        ) from exc

    if args.resume_adapter:
        return PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)

    target_modules = [
        item.strip()
        for item in args.lora_target_modules.split(",")
        if item.strip()
    ]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    return get_peft_model(model, lora_config)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-adapter", default="")
    parser.add_argument("--system", default="")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-response-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--mask-epsilon", type=float, default=1e-3)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="query_key_value,dense")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.per_device_batch_size < 1:
        raise ValueError("--per-device-batch-size must be >= 1")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.max_length < 2:
        raise ValueError("--max-length must be >= 2")
    if args.block_length < 1:
        raise ValueError("--block-length must be >= 1")
    if not 0.0 <= args.mask_epsilon < 1.0:
        raise ValueError("--mask-epsilon must be in [0, 1)")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = JsonlTrainingDataset(Path(args.train_file))
    counts = Counter(
        f"{row.get('source', 'unknown')}:{row.get('loss_mode', 'unknown')}"
        for row in dataset.rows
    )
    print(f"[train] loaded records={len(dataset)} counts={dict(sorted(counts.items()))}", flush=True)

    kwargs = common_model_kwargs(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = token_id(tokenizer, "eos_token", fallback=156892)
    mask_id = token_id(tokenizer, "mask_token", fallback=156895)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_arg(args.torch_dtype),
        **kwargs,
    )
    base_model.config.use_cache = False
    model = add_lora(base_model, args)

    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for LLaDA2.0-mini LoRA training.")
    device = torch.device("cuda")
    model.to(device)
    model.train()

    total_params, trainable_params = count_parameters(model)
    print(
        "[train] "
        f"trainable_params={trainable_params:,} total_params={total_params:,} "
        f"ratio={trainable_params / total_params:.6f}",
        flush=True,
    )

    collator = A2DCollator(
        tokenizer,
        mask_id=mask_id,
        eos_id=eos_id,
        max_length=args.max_length,
        max_response_length=args.max_response_length,
        mask_epsilon=args.mask_epsilon,
        system=args.system or None,
        seed=args.seed + 17,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        generator=loader_generator,
    )

    updates_per_epoch = math.ceil(len(dataloader) / args.grad_accum_steps)
    planned_epochs = max(1, math.ceil(args.epochs))
    planned_steps = updates_per_epoch * planned_epochs
    total_training_steps = args.max_steps if args.max_steps > 0 else planned_steps
    warmup_steps = args.warmup_steps or int(total_training_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    print(
        "[train] "
        f"epochs={args.epochs} max_steps={args.max_steps} "
        f"optimizer_steps={total_training_steps} warmup_steps={warmup_steps} "
        f"effective_batch={args.per_device_batch_size * args.grad_accum_steps}",
        flush=True,
    )

    started = time.time()
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    running_batches = 0
    seen_examples = 0
    stop_training = False
    attn_dtype = torch.bfloat16 if args.torch_dtype in ("auto", "bfloat16") else dtype_arg(args.torch_dtype)

    for epoch_idx in range(planned_epochs):
        if stop_training:
            break
        epoch_fraction_limit = args.epochs - epoch_idx
        if epoch_fraction_limit <= 0:
            break
        max_batches_this_epoch = len(dataloader)
        if epoch_fraction_limit < 1:
            max_batches_this_epoch = max(1, int(len(dataloader) * epoch_fraction_limit))

        for batch_idx, batch in enumerate(dataloader, start=1):
            if batch_idx > max_batches_this_epoch:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            attention_mask = make_block_attention_mask(
                valid_mask,
                block_length=args.block_length,
                dtype=attn_dtype,
            )
            position_ids = torch.arange(
                input_ids.shape[1], device=device, dtype=torch.long
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: {loss.item()}")
            (loss / args.grad_accum_steps).backward()

            micro_step += 1
            running_loss += float(loss.detach().cpu())
            running_batches += 1
            seen_examples += input_ids.shape[0]

            if micro_step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    (param for param in model.parameters() if param.requires_grad),
                    args.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0 or global_step == 1:
                    elapsed = time.time() - started
                    avg_loss = running_loss / max(1, running_batches)
                    lr = scheduler.get_last_lr()[0]
                    max_mem_gb = (
                        torch.cuda.max_memory_allocated(device) / 1_000_000_000
                        if torch.cuda.is_available()
                        else 0.0
                    )
                    print(
                        "[train] "
                        f"step={global_step}/{total_training_steps} "
                        f"epoch={epoch_idx + 1} seen={seen_examples} "
                        f"loss={avg_loss:.4f} lr={lr:.3e} "
                        f"max_mem_gb={max_mem_gb:.2f} "
                        f"elapsed_s={elapsed:.1f}",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_batches = 0

                if args.save_every > 0 and global_step % args.save_every == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    print(f"[train] saved {ckpt_dir}", flush=True)

                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop_training = True
                    break

    if micro_step % args.grad_accum_steps != 0 and not stop_training:
        torch.nn.utils.clip_grad_norm_(
            (param for param in model.parameters() if param.requires_grad),
            args.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        "finished_at": utc_now(),
        "elapsed_seconds": time.time() - started,
        "train_file": args.train_file,
        "output_dir": str(output_dir),
        "records": len(dataset),
        "record_counts": dict(sorted(counts.items())),
        "optimizer_steps": global_step,
        "seen_examples": seen_examples,
        "lora_target_modules": [
            item.strip()
            for item in args.lora_target_modules.split(",")
            if item.strip()
        ],
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "max_length": args.max_length,
        "max_response_length": args.max_response_length,
        "block_length": args.block_length,
        "mask_epsilon": args.mask_epsilon,
        "effective_batch_size": args.per_device_batch_size * args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "seed": args.seed,
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
