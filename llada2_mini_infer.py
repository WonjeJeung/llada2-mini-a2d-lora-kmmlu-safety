#!/usr/bin/env python3
"""Run local inference with inclusionAI/LLaDA2.0-mini."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "inclusionAI/LLaDA2.0-mini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference CLI for inclusionAI/LLaDA2.0-mini."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF model id or path.")
    parser.add_argument(
        "--adapter",
        default="",
        help="Optional PEFT/LoRA adapter path to load on top of --model.",
    )
    parser.add_argument("--prompt", default=None, help="User prompt.")
    parser.add_argument("--system", default=None, help="Optional system prompt.")
    parser.add_argument(
        "--messages-json",
        default=None,
        help=(
            "Optional JSON list of chat messages. If set, it overrides --prompt and "
            "--system. Example: '[{\"role\":\"user\",\"content\":\"hi\"}]'"
        ),
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Tokenize --prompt as plain text instead of applying the chat template.",
    )
    parser.add_argument(
        "--gen-length",
        type=int,
        default=512,
        help="Maximum generated tokens, excluding the prompt.",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=32,
        help="Diffusion block size. Good defaults are 32 or 64.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Denoising steps per block. Usually keep this near block-length.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Confidence threshold for accepting generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. With 0.0 this script uses top_k=1 by default.",
    )
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling p.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument(
        "--eos-early-stop",
        action="store_true",
        help="Stop as soon as an EOS token is confirmed.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map. Use 'auto' for multi-GPU sharding.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Weight dtype passed to Transformers.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use files already present in the Hugging Face cache/local path.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    args = parser.parse_args()
    if not args.messages_json and not args.prompt:
        parser.error("one of --prompt or --messages-json is required")
    if args.raw_prompt and not args.prompt:
        parser.error("--raw-prompt requires --prompt")
    return args


def build_messages(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.messages_json:
        messages = json.loads(args.messages_json)
        if not isinstance(messages, list):
            raise ValueError("--messages-json must be a JSON list.")
        return messages

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    return messages


def dtype_arg(value: str) -> str | torch.dtype:
    if value == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any]:
    common_kwargs = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    token = os.environ.get("HF_TOKEN")
    if token:
        common_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(args.model, **common_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=dtype_arg(args.torch_dtype),
        **common_kwargs,
    )
    if args.adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "--adapter requires peft. Run ./run_llada2_train_deps_sbatch.sh first."
            ) from exc
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model, tokenizer


def encode_prompt(tokenizer: Any, args: argparse.Namespace) -> torch.Tensor:
    if args.raw_prompt:
        return tokenizer(args.prompt, return_tensors="pt").input_ids

    messages = build_messages(args)
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )


def token_id(tokenizer: Any, token_name: str, fallback: int) -> int:
    token = getattr(tokenizer, token_name, None)
    token_id_value = tokenizer.convert_tokens_to_ids(token) if token else None
    if isinstance(token_id_value, int) and token_id_value >= 0:
        return token_id_value
    return fallback


def remove_prompt_prefix(
    generated_ids: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
    """Some early-stop paths return prompt + answer; normalize to answer tokens."""
    prompt_length = input_ids.shape[1]
    if generated_ids.shape[1] < prompt_length:
        return generated_ids

    prefix = generated_ids[:, :prompt_length].to(input_ids.device)
    if torch.equal(prefix, input_ids):
        return generated_ids[:, prompt_length:]
    return generated_ids


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    model, tokenizer = load_model_and_tokenizer(args)
    input_ids = encode_prompt(tokenizer, args)

    eos_id = token_id(tokenizer, "eos_token", fallback=156892)
    mask_id = token_id(tokenizer, "mask_token", fallback=156895)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids,
            temperature=args.temperature,
            block_length=args.block_length,
            steps=args.steps,
            gen_length=args.gen_length,
            top_p=args.top_p,
            top_k=args.top_k,
            threshold=args.threshold,
            eos_early_stop=args.eos_early_stop,
            eos_id=eos_id,
            mask_id=mask_id,
        )

    answer_ids = remove_prompt_prefix(generated_ids, input_ids)
    text = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    print(text.strip())


if __name__ == "__main__":
    main()
