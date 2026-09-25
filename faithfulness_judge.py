#!/usr/bin/env python3
"""Score answer-level context faithfulness with a fixed Qwen3 judge."""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_JUDGE = "Qwen/Qwen3-32B"
SYSTEM_PROMPT = (
    "You are a strict context-faithfulness judge for retrieval-augmented generation. "
    "An answer is faithful only if every factual claim in it is directly stated by, "
    "or clearly entailed by, the provided context. Ignore reference-answer correctness. "
    "Return JSON only."
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def answer_from(row: Dict[str, Any]) -> str:
    return str(row.get("model_answer") or row.get("prediction") or row.get("answer") or "").strip()


def judge_prompt(tokenizer, row: Dict[str, Any], answer: str, dataset: str) -> str:
    question = str(row.get("question") or row.get("query") or row.get("head") or "").strip()
    context = str(row.get("context") or row.get("passage") or "").strip()
    reference = str(row.get("reference_answer") or row.get("tail") or "").strip()
    task = "conversation reply" if dataset.lower() == "wow" else "question answer"
    sections = [
        f"DATASET: {dataset}",
        f"TASK TYPE: {task}",
        f"QUESTION_OR_DIALOGUE: {question}",
    ]
    if reference:
        sections.append(f"REFERENCE_FOR_ORIENTATION_ONLY_DO_NOT_USE_FOR_FAITHFULNESS: {reference}")
    sections.extend(
        [
            f"CONTEXT:\n{context}",
            f"MODEL_ANSWER:\n{answer}",
            (
                "Judge only faithfulness to CONTEXT. Do not mark unsupported just because "
                "the answer differs from the reference; mark unsupported only when the "
                "answer adds factual content not entailed by the context. For conversation, "
                "allow harmless social wording, but factual claims still need context support."
            ),
            (
                'Return exactly one JSON object: {"faithful": true/false, '
                '"reason": "short reason", "unsupported_claims": ["..."]}.'
            ),
        ]
    )
    user_prompt = "\n\n".join(sections)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "/no_think\n" + user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_faithful(text: str) -> bool:
    cleaned = text.strip()
    match = re.search(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        return json.loads(cleaned).get("faithful") is True
    except (json.JSONDecodeError, AttributeError):
        return bool(re.search(r'"faithful"\s*:\s*true\b', text, flags=re.IGNORECASE))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--dataset", default="unknown")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--only-correct", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=False,
    ).eval()
    input_device = next(model.parameters()).device

    scored: List[Dict[str, Any]] = []
    parse_failures = 0
    for source in tqdm(read_jsonl(args.input_jsonl), desc="Qwen faithfulness"):
        if args.only_correct and not bool(source.get("correct")):
            continue
        answer = answer_from(source)
        prompt = judge_prompt(tokenizer, source, answer, args.dataset)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(input_device)
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(output[0, encoded.input_ids.shape[1] :], skip_special_tokens=True).strip()
        faithful = parse_faithful(raw)
        if not re.search(r'"faithful"\s*:', raw, flags=re.IGNORECASE):
            parse_failures += 1
        row = dict(source)
        row.update(
            {
                "faithfulness_judge": args.judge_model,
                "faithfulness_prompt": SYSTEM_PROMPT,
                "faithfulness_raw": raw,
                "faithful": faithful,
            }
        )
        scored.append(row)

    write_jsonl_atomic(args.output_jsonl, scored)
    faithful_count = sum(bool(row["faithful"]) for row in scored)
    total = len(scored)
    print(
        json.dumps(
            {
                "faithful": faithful_count,
                "total": total,
                "faithfulness": faithful_count / total if total else 0.0,
                "parse_failures": parse_failures,
                "only_correct": args.only_correct,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
