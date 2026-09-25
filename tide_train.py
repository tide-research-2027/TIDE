#!/usr/bin/env python3
# coding: utf-8
"""
Standalone full-parameter trainer for TIDE/FIND and its training ablations.

This file prepares MSMARCO or NewsQA RAG training rows, computes the FIND value

    V(x, y) = log pi_0(y | x) + F(x, y)

where pi_0 is the base LLM and F is an NLI entailment score, then full
fine-tunes the base model with weighted answer loss:

    weight = exp(alpha * V(x, y))

Default models:
  policy/base model: Qwen/Qwen3-14B
  training NLI model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli

Modes are defined by the value used in the exponential weight:

  find:       log pi_0(y | x) + F(x, y)
  less_value: log pi_0(y | x)
  faith_only: F(x, y)
  sft:        constant weight 1

This performs full-parameter training. It does not import PEFT and does not
create LoRA or QLoRA adapters. With torchrun, rank 0 constructs the deterministic
weighted-data cache and publishes it atomically before all ranks begin training.

Examples:
  python train_our_los_find_standalone.py \
    --dataset msmarco \
    --max-samples 75000 \
    --alpha 0.5 \
    --output-dir ./models/msmarco_our_los_alpha05

  torchrun --nproc_per_node=8 train_our_los_find_standalone.py \
    --dataset msmarco \
    --max-samples 75000 \
    --alpha 0.5 \
    --output-dir ./models/msmarco_our_los_alpha05 \
    --deepspeed deepspeed_find_zero3.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


BASE_MODEL = "Qwen/Qwen3-14B"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def rank_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def is_main_process() -> bool:
    rank, _, _ = rank_info()
    return rank == 0


def wait_for_file(path: str | Path, timeout_seconds: int = 86400) -> None:
    path = Path(path)
    ready_path = Path(f"{path}.ready")
    start = time.time()
    while time.time() - start < timeout_seconds:
        if path.exists() and path.stat().st_size > 0 and ready_path.exists():
            return
        time.sleep(15)
    raise TimeoutError(f"Timed out waiting for file: {path}")


def clean_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def qa_prompt(question: str, context: str) -> str:
    return f"Query: {clean_text(question)}\n\nPassage: {clean_text(context)}\n\nAnswer:"


def extract_answers(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        text = clean_text(x)
        return [text] if text else []
    if isinstance(x, dict):
        for key in ("text", "answer", "answers"):
            if key in x:
                return extract_answers(x[key])
        return []
    if isinstance(x, list):
        out: List[str] = []
        for item in x:
            out.extend(extract_answers(item))
        return [item for item in out if item]
    text = clean_text(x)
    return [text] if text else []


def normalize_for_match(text: Any) -> str:
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def iter_newsqa_spans(labels: Any) -> List[tuple[int, int]]:
    spans: List[tuple[int, int]] = []
    if not isinstance(labels, list):
        return spans
    for label in labels:
        if not isinstance(label, dict):
            continue
        starts = label.get("start", [])
        ends = label.get("end", [])
        starts = [starts] if isinstance(starts, int) else starts
        ends = [ends] if isinstance(ends, int) else ends
        if not isinstance(starts, list) or not isinstance(ends, list):
            continue
        for start, end in zip(starts, ends):
            try:
                start, end = int(start), int(end)
            except (TypeError, ValueError):
                continue
            if 0 <= start < end:
                spans.append((start, end))
    return spans


def answer_window(context: str, start: int, end: int, max_chars: int) -> str:
    if max_chars <= 0 or len(context) <= max_chars:
        return context.strip()
    center = (start + end) // 2
    left = max(0, center - max_chars // 2)
    right = min(len(context), left + max_chars)
    left = max(0, right - max_chars)
    return context[left:right].strip()


def choose_newsqa_answer(
    example: Dict[str, Any], max_context_chars: int, max_answer_words: int, max_answer_chars: int
) -> tuple[Optional[str], Optional[str]]:
    context = str(example.get("context") or "")
    references = extract_answers(example.get("answers"))
    reference_norms = {normalize_for_match(ref) for ref in references if normalize_for_match(ref)}

    for start, end in iter_newsqa_spans(example.get("labels")):
        if end > len(context):
            continue
        span = clean_text(context[start:end])
        if not span:
            continue
        matching_ref = next(
            (ref for ref in references if normalize_for_match(ref) == normalize_for_match(span)),
            None,
        )
        answer = clean_text(matching_ref or span)
        if reference_norms and normalize_for_match(answer) not in reference_norms:
            answer = span
        if max_answer_chars > 0 and len(answer) > max_answer_chars:
            continue
        if max_answer_words > 0 and len(re.findall(r"\w+", answer)) > max_answer_words:
            continue
        if normalize_for_match(answer) in {"unknown", "cannot answer", "no answer present"}:
            continue
        context_window = answer_window(context, start, end, max_context_chars)
        if normalize_for_match(answer) not in normalize_for_match(context_window):
            continue
        return answer, clean_text(context_window)
    for reference in references:
        start = context.lower().find(reference.lower())
        if start >= 0:
            return clean_text(reference), clean_text(answer_window(context, start, start + len(reference), max_context_chars))
    return None, None


def selected_msmarco_passage(example: Dict[str, Any]) -> str:
    passages = example.get("passages") or {}
    texts = passages.get("passage_text") or []
    flags = passages.get("is_selected") or []

    for text, flag in zip(texts, flags):
        try:
            selected = int(flag) == 1
        except Exception:
            selected = bool(flag)
        if selected and clean_text(text):
            return clean_text(text)

    # The reported accurate-retrieval training subset uses selected passages only.
    return ""


def load_training_rows(args) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if args.dataset == "msmarco":
        ds = load_dataset("microsoft/ms_marco", "v1.1", split=args.split)
        for ex in ds:
            question = clean_text(ex.get("query"))
            context = selected_msmarco_passage(ex)
            refs = extract_answers(ex.get("answers") or ex.get("wellFormedAnswers"))

            if question and context and refs:
                rows.append(
                    {
                        "dataset": "msmarco",
                        "question": question,
                        "context": context,
                        "answer": refs[0],
                        "references": refs,
                        "prompt": qa_prompt(question, context),
                    }
                )
                if args.max_samples >= 0 and len(rows) >= args.max_samples:
                    break

    elif args.dataset == "newsqa":
        all_splits = load_dataset("lucadiliello/newsqa")
        if args.split not in all_splits:
            raise ValueError(f"NewsQA split {args.split!r} is unavailable: {list(all_splits)}")
        ds = all_splits[args.split]
        for ex in ds:
            question = clean_text(ex.get("question"))
            answer, context = choose_newsqa_answer(
                ex,
                args.newsqa_context_chars,
                args.newsqa_max_answer_words,
                args.newsqa_max_answer_chars,
            )
            refs = extract_answers(ex.get("answers"))

            if question and context and answer:
                rows.append(
                    {
                        "dataset": "newsqa",
                        "question": question,
                        "context": context,
                        "answer": answer,
                        "references": refs or [answer],
                        "prompt": qa_prompt(question, context),
                    }
                )
                if args.max_samples >= 0 and len(rows) >= args.max_samples:
                    break

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.max_samples >= 0 and len(rows) != args.max_samples:
        raise RuntimeError(
            f"Requested {args.max_samples} retained {args.dataset} examples, "
            f"but preprocessing produced {len(rows)} from split {args.split!r}."
        )
    return rows


def answer_mean_logp(
    model, tokenizer, prompt: str, answer: str, device: str, max_length: int
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + answer, add_special_tokens=False).input_ids
    if full_ids[: len(prompt_ids)] == prompt_ids:
        answer_ids = full_ids[len(prompt_ids) :]
    else:
        answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None and (not answer_ids or answer_ids[-1] != tokenizer.eos_token_id):
        answer_ids.append(tokenizer.eos_token_id)
    if len(prompt_ids) + len(answer_ids) > max_length:
        prompt_ids = prompt_ids[-max(1, max_length - len(answer_ids)) :]
        answer_ids = answer_ids[: max(1, max_length - len(prompt_ids))]

    input_ids = torch.tensor([prompt_ids + answer_ids], device=device)

    with torch.no_grad():
        logits = model(input_ids).logits[:, :-1]
        labels = input_ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)

        start = max(len(prompt_ids) - 1, 0)
        end = start + len(answer_ids)
        vals = logp[0, start:end, labels[0, start:end]]

    return float(vals.mean().cpu())


def faithfulness_score(
    nli_model, nli_tokenizer, context: str, answer: str, device: str, max_length: int
) -> float:
    encoded = nli_tokenizer(
        context,
        answer,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        logits = nli_model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1)

    entail_idx = [i for i, label in nli_model.config.id2label.items() if "entail" in label.lower()][0]
    return float(probs[entail_idx].cpu())


def save_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temporary_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
    ready_path = Path(f"{path}.ready")
    ready_path.write_text(json.dumps({"rows": len(rows)}) + "\n", encoding="utf-8")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_our_los_weights(args) -> List[Dict[str, Any]]:
    weighted_path = Path(args.weighted_jsonl)

    if args.reuse_weighted_jsonl and weighted_path.exists() and Path(f"{weighted_path}.ready").exists():
        if is_main_process():
            print(f"[reuse weighted data] {weighted_path}")
        return load_jsonl(weighted_path)

    rank, local_rank, world_size = rank_info()

    if world_size > 1 and not is_main_process():
        wait_for_file(weighted_path)
        return load_jsonl(weighted_path)

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    print(f"[prepare dataset] dataset={args.dataset} split={args.split} max_samples={args.max_samples}")
    rows = load_training_rows(args)
    if not rows:
        raise RuntimeError("No valid training rows remained after preprocessing")
    print(f"[dataset rows] {len(rows)}")

    base_tokenizer = None
    base_model = None
    if args.mode in {"find", "less_value"}:
        print(f"[load base scorer] {args.base_model}")
        base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=False,
        ).to(device).eval()

    nli_tokenizer = None
    nli_model = None
    if args.mode in {"find", "faith_only"}:
        print(f"[load NLI scorer] {args.nli_model}")
        nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
        nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(device).eval()

    raw_weights: List[float] = []
    for row in tqdm(rows, desc="scoring Our LOS weights"):
        logp = None
        faith = None
        if base_model is not None and base_tokenizer is not None:
            logp = answer_mean_logp(
                base_model, base_tokenizer, row["prompt"], row["answer"], device, args.max_length
            )
        if nli_model is not None and nli_tokenizer is not None:
            faith = faithfulness_score(
                nli_model,
                nli_tokenizer,
                row["context"],
                row["answer"],
                device,
                args.nli_max_length,
            )
        if args.mode == "find":
            value = float(logp) + float(faith)
        elif args.mode == "less_value":
            value = float(logp)
        elif args.mode == "faith_only":
            value = float(faith)
        else:
            value = 0.0
        raw_weight = 1.0 if args.mode == "sft" else math.exp(args.alpha * value)

        row["base_logp"] = logp
        row["faithfulness"] = faith
        row["value"] = value
        row["raw_weight"] = raw_weight
        raw_weights.append(raw_weight)

    mean_weight = sum(raw_weights) / max(len(raw_weights), 1)
    for row in rows:
        if args.normalize_weights and mean_weight > 0:
            row["weight"] = row["raw_weight"] / mean_weight
        else:
            row["weight"] = row["raw_weight"]

    save_jsonl(weighted_path, rows)
    print(f"[saved weighted data] {weighted_path}")
    print(
        {
            "rows": len(rows),
            "alpha": args.alpha,
            "mean_raw_weight": mean_weight,
            "normalized": args.normalize_weights,
        }
    )

    if base_model is not None:
        del base_model
    if nli_model is not None:
        del nli_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


def encode_row(
    row: Dict[str, Any], tokenizer, max_length: int, max_prompt_length: int
) -> Dict[str, Any]:
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False).input_ids
    full_ids = tokenizer(row["prompt"] + row["answer"], add_special_tokens=False).input_ids
    if full_ids[: len(prompt_ids)] == prompt_ids:
        answer_ids = full_ids[len(prompt_ids) :]
    else:
        answer_ids = tokenizer(row["answer"], add_special_tokens=False).input_ids
    if tokenizer.eos_token_id is not None and (not answer_ids or answer_ids[-1] != tokenizer.eos_token_id):
        answer_ids.append(tokenizer.eos_token_id)
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]
    answer_ids = answer_ids[: max(1, max_length - len(prompt_ids))]
    input_ids = prompt_ids + answer_ids

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + answer_ids,
        "weight": float(row.get("weight", 1.0)),
    }


class FINDDataCollator:
    def __init__(self, tokenizer) -> None:
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)

        def padded(key: str, pad_value: int) -> torch.Tensor:
            values = [
                feature[key] + [pad_value] * (max_length - len(feature[key]))
                for feature in features
            ]
            return torch.tensor(values, dtype=torch.long)

        return {
            "input_ids": padded("input_ids", self.pad_token_id),
            "attention_mask": padded("attention_mask", 0),
            "labels": padded("labels", -100),
            "weight": torch.tensor([feature["weight"] for feature in features], dtype=torch.float32),
        }


class OurLOSTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight").to(model.device).float()
        labels = inputs["labels"]

        outputs = model(**inputs)
        logits = outputs.logits[:, :-1].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        token_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            shifted_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shifted_labels.shape)

        mask = shifted_labels.ne(-100)
        example_loss = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        loss = (example_loss * weights).mean()

        return (loss, outputs) if return_outputs else loss


def train_model(args, weighted_rows: List[Dict[str, Any]]) -> None:
    rank, local_rank, _ = rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"[load train model] {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=False,
    )
    model.config.use_cache = False

    features = [
        encode_row(row, tokenizer, args.max_length, args.max_prompt_length)
        for row in weighted_rows
    ]
    train_dataset = Dataset.from_list(features)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim=args.optim,
        deepspeed=args.deepspeed,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        max_grad_norm=1.0,
        save_safetensors=True,
        seed=args.seed,
        data_seed=args.seed,
        ddp_find_unused_parameters=False if int(os.environ.get("WORLD_SIZE", "1")) > 1 else None,
    )

    trainer = OurLOSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=FINDDataCollator(tokenizer),
        tokenizer=tokenizer,
    )

    print("[train] starting Our LOS / FIND full fine-tuning")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print(f"[save] {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    if trainer.is_world_process_zero():
        summary = {
            "script": Path(__file__).name,
            "dataset": args.dataset,
            "mode": args.mode,
            "base_model": args.base_model,
            "nli_model": args.nli_model,
            "alpha": args.alpha,
            "normalize_weights": args.normalize_weights,
            "train_rows": len(weighted_rows),
            "prompt_format": "Query: {question}\\n\\nPassage: {context}\\n\\nAnswer:",
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "max_prompt_length": args.max_prompt_length,
            "nli_max_length": args.nli_max_length,
            "full_parameter_training": True,
        }
        Path(args.output_dir, "find_training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone TIDE/FIND and ablation training")

    parser.add_argument("--dataset", choices=["msmarco", "newsqa"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=75000)
    parser.add_argument("--newsqa-context-chars", type=int, default=1400)
    parser.add_argument("--newsqa-max-answer-words", type=int, default=0)
    parser.add_argument("--newsqa-max-answer-chars", type=int, default=0)

    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--nli-model", default=NLI_MODEL)
    parser.add_argument("--mode", choices=["find", "less_value", "faith_only", "sft"], default="find")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--normalize-weights",
        action="store_true",
        help="Optionally divide exp(alpha * V) by its empirical mean; disabled in the reported run.",
    )

    parser.add_argument("--weighted-jsonl", default=None)
    parser.add_argument("--reuse-weighted-jsonl", action="store_true")

    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=3084)
    parser.add_argument("--nli-max-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=20)

    parser.add_argument("--save-strategy", choices=["no", "steps"], default="no")
    parser.add_argument("--save-steps", type=int, default=300)
    parser.add_argument("--save-total-limit", type=int, default=2)

    parser.add_argument("--optim", default="adamw_bnb_8bit")
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--resume-from-checkpoint", default=None)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.weighted_jsonl is None:
        alpha_tag = str(args.alpha).replace(".", "p")
        args.weighted_jsonl = str(
            Path(args.output_dir) / f"{args.dataset}_{args.mode}_alpha{alpha_tag}_weighted.jsonl"
        )

    weighted_rows = compute_our_los_weights(args)
    train_model(args, weighted_rows)


if __name__ == "__main__":
    main()
