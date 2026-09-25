#!/usr/bin/env python3
"""Token-weighted full-vocabulary forward KL: KL(base || trained model)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE = "Qwen/Qwen3-14B"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def response(row: Dict[str, Any]) -> str:
    return str(row.get("model_answer") or row.get("prediction") or row.get("answer") or "").strip()


def tokenized_sequence(tokenizer, prompt: str, answer: str, max_length: int) -> Tuple[List[int], int]:
    if max_length < 2:
        raise ValueError("--max-length must be at least 2")
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + answer, add_special_tokens=False).input_ids
    if full_ids[: len(prompt_ids)] == prompt_ids:
        answer_ids = full_ids[len(prompt_ids) :]
    else:
        answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None and (not answer_ids or answer_ids[-1] != tokenizer.eos_token_id):
        answer_ids.append(tokenizer.eos_token_id)
    # Preserve the end of the prompt and the start of the generated answer.
    prompt_ids = prompt_ids[-(max_length - 1) :]
    answer_ids = answer_ids[: max_length - len(prompt_ids)]
    return prompt_ids + answer_ids, len(prompt_ids)


def forward_kl_for_answer(
    base_model,
    trained_model,
    ids: List[int],
    prompt_length: int,
    base_device: torch.device,
    trained_device: torch.device,
) -> Tuple[float, int]:
    if len(ids) < 2 or prompt_length < 1 or prompt_length >= len(ids):
        return 0.0, 0
    base_input = torch.tensor([ids], dtype=torch.long, device=base_device)
    trained_input = torch.tensor([ids], dtype=torch.long, device=trained_device)
    start = prompt_length - 1

    with torch.inference_mode():
        base_logits = base_model(base_input, use_cache=False).logits[0, start:-1].float()
        trained_logits = trained_model(trained_input, use_cache=False).logits[0, start:-1].float()
        base_log_probs = torch.log_softmax(base_logits, dim=-1)
        base_probs = base_log_probs.exp()
        trained_log_probs = torch.log_softmax(trained_logits, dim=-1).to(base_device)
        token_kl = (base_probs * (base_log_probs - trained_log_probs)).sum(dim=-1)
    return float(token_kl.sum().cpu()), int(token_kl.numel())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--trained-model", required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--base-device", default="cuda:0")
    parser.add_argument("--trained-device", default="cuda:1")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-examples", type=int, default=-1)
    args = parser.parse_args()

    if not torch.cuda.is_available() and ("cuda" in args.base_device or "cuda" in args.trained_device):
        args.base_device = args.trained_device = "cpu"
    base_device = torch.device(args.base_device)
    trained_device = torch.device(args.trained_device)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, trust_remote_code=False
    ).to(base_device).eval()
    trained = AutoModelForCausalLM.from_pretrained(
        args.trained_model, torch_dtype=dtype, trust_remote_code=False
    ).to(trained_device).eval()

    rows = read_jsonl(args.input_jsonl)
    if args.max_examples >= 0:
        rows = rows[: args.max_examples]
    kl_sum = 0.0
    token_count = 0
    used_examples = 0
    for row in tqdm(rows, desc="KL(base || model)"):
        answer = response(row)
        if not answer:
            continue
        ids, prompt_length = tokenized_sequence(tokenizer, str(row["prompt"]), answer, args.max_length)
        example_sum, example_tokens = forward_kl_for_answer(
            base, trained, ids, prompt_length, base_device, trained_device
        )
        if example_tokens:
            kl_sum += example_sum
            token_count += example_tokens
            used_examples += 1

    result = {
        "direction": "KL(base || trained_model)",
        "base_model": args.base_model,
        "trained_model": args.trained_model,
        "examples": used_examples,
        "answer_tokens": token_count,
        "token_weighted_kl": kl_sum / token_count if token_count else 0.0,
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
