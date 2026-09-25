#!/usr/bin/env python3
# coding: utf-8
"""Complete dataset-neutral RAM/SD-RA-IT-style Self-Demo training.

This one file performs the full pipeline:
1. Prepare MS MARCO train rows.
2. Optimize no-RAG and RAG system messages with iterative generate/score/critique/rewrite.
3. Generate NoRag/Rag/RagRefuse candidates, select by tournament judging, then
   apply a strict paper-faithful correctness/support/human-quality filter.
4. Fine-tune Qwen3-14B on the generated self-demo answers with the same
   unweighted SFT LOS loss path used by weighted_loss_training_weighted_fix_v2.py:
   w_fixed=1, faithfulness_fixed=0, use_weighted_logits=False.

This variant is deliberately comparable to the Facebook RAM scripts:
projects/sd-ra-it/scripts/prompt_optimization.py and get_demos.py.  The
additional filter is needed because our validation is a strict answer-only,
no-false-positive evaluator.  Since these MSMARCO rows are answer-present by
construction, rows with no clean self-generated candidate are skipped by
default rather than turned into false-refusal training examples.

The script intentionally imports vLLM only inside the vLLM stage. The default
`--stage all` command runs this same file once under the vLLM environment for
generation, then trains in the current environment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import shutil


# Keep these before torch/vLLM import.
os.environ.setdefault("WANDB_DIR", "./cache/wandb")
os.environ.setdefault("WANDB_CACHE_DIR", "./cache/wandb_cache")
os.environ.setdefault("WANDB_CONFIG_DIR", "./cache/wandb_config")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("WANDB_PROJECT", "simpo")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


DEFAULT_BASE_MODEL = "Qwen/Qwen3-14B"
DEFAULT_RUN_TAG = "tide_self_demo_qwen3_14b"
DEFAULT_DATA_ROOT = f"./runs/self_demo/{DEFAULT_RUN_TAG}/data"
DEFAULT_OUTPUT_DIR = f"./runs/self_demo/{DEFAULT_RUN_TAG}/model"
DEFAULT_CACHE_ROOT = f"./runs/self_demo/{DEFAULT_RUN_TAG}/cache"
DEFAULT_VLLM_PYTHON = sys.executable


def _has_complete_weights(path: Path) -> bool:
    for index_file in path.glob("*.index.json"):
        weight_map = json.loads(index_file.read_text()).get("weight_map", {})
        shards = set(weight_map.values())
        if shards and all((path / name).is_file() and (path / name).stat().st_size > 1024 for name in shards):
            return True
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    return any(item.is_file() and item.stat().st_size > 1024 for item in weights)


def verify_full_model_output(path: str | os.PathLike[str]) -> None:
    output = Path(path)
    if (output / "adapter_config.json").exists():
        raise RuntimeError(f"Unexpected adapter output: {output}")
    if not (output / "config.json").is_file() or not _has_complete_weights(output):
        raise RuntimeError(f"Incomplete full-model checkpoint: {output}")


def robust_save_full_model(trainer: Any, tokenizer: Any, output_dir: str | os.PathLike[str]) -> None:
    output = Path(output_dir)
    temporary = output.with_name(output.name + ".saving")
    previous = output.with_name(output.name + ".previous_incomplete")
    accelerator = trainer.accelerator
    if accelerator.is_main_process:
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    trainer.args.save_safetensors = True
    trainer.args.save_only_model = False
    trainer.save_model(str(temporary))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        tokenizer.save_pretrained(temporary)
        verify_full_model_output(temporary)
        if previous.exists():
            shutil.rmtree(previous)
        if output.exists():
            output.rename(previous)
        temporary.rename(output)
        if previous.exists():
            shutil.rmtree(previous)
        verify_full_model_output(output)
    accelerator.wait_for_everyone()


# =========================
# Prompt constants
# =========================


INITIAL_NO_RAG_PROMPT = (
    "Answer the user's question directly and concisely. "
    "Return only the answer. Do not explain."
)

INITIAL_RAG_PROMPT = (
    "Answer the user's question using the provided passages. "
    "Return only a concise answer supported by the passages. "
    "If the answer is not supported by the passages, say you do not know."
)

NO_RAG_PROMPTS = [
    INITIAL_NO_RAG_PROMPT,
    "Give a short factual answer only. Do not explain.",
    "Answer with the exact entity, date, number, or phrase requested.",
]

RAG_PROMPTS = [
    INITIAL_RAG_PROMPT,
    "Use only the retrieved passages. Give a short supported answer only.",
    "Answer from the passages. If unsupported, say you do not know.",
]

REFUSAL_PROMPT = (
    "Use the passages to answer. If the answer cannot be determined from the passages, "
    "say you do not know. Do not guess."
)

DEFAULT_REFUSAL = "I do not know."

SCORE_SYSTEM = (
    "You are grading a prediction for a question-answering task. "
    "Compare the prediction to the gold answer. "
    "Give Score: 5 if it is correct, Score: 4 if essentially correct, "
    "Score: 3 if partially correct, Score: 2 if related but mostly wrong, "
    "and Score: 1 if wrong or unsupported. "
    "Reply exactly in this format: Score: N"
)

REWRITE_SYSTEM = (
    "You are an expert prompt optimizer. "
    "Given a current system prompt and examples of predictions with scores, "
    "write a better system prompt for future question-answering examples. "
    "Return only the improved system prompt."
)

JUDGE_SYSTEM = (
    "You are selecting the best prediction for supervised fine-tuning. "
    "Choose the prediction that best matches the gold answer and is supported by the passages. "
    "Reply exactly as: Best: Prediction N"
)

SUMMARIZE_SYSTEM = (
    "Rewrite the answer as a short final answer. "
    "Remove explanations, hedging, and extra text. "
    "Return only the answer."
)

SCORE_PAT = re.compile(r"Score:\s*(?P<score>[1-5])", re.IGNORECASE)
CHOICE_PAT = re.compile(r"Best:(?: ?Prediction) (?P<idx>\d)", re.IGNORECASE)
ANSWER_PREFIX_PAT = re.compile(r"^\s*(?:the\s+answer\s+is|answer)\s*:?\s*", re.IGNORECASE)


# =========================
# General helpers
# =========================


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def normalize_reference_answers(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        values = [values]

    out = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("text") or value.get("answer") or ""
        text = str(value or "").strip()
        if text and text.lower() not in {"no answer present.", "no answer present"}:
            out.append(text)
    return out


def make_prompt(question: str, context: str, dataset: str = "msmarco") -> str:
    label = "Query" if dataset == "msmarco" else "Question"
    return f"{label}: {question}\n\nPassage: {context}\n\nAnswer:"


def normalize_for_match(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())).strip()


def iter_label_spans(labels: Any) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for item in labels if isinstance(labels, list) else []:
        if not isinstance(item, dict):
            continue
        starts, ends = item.get("start", []), item.get("end", [])
        starts = [starts] if isinstance(starts, int) else starts
        ends = [ends] if isinstance(ends, int) else ends
        for start, end in zip(starts or [], ends or []):
            try:
                start, end = int(start), int(end)
            except (TypeError, ValueError):
                continue
            if 0 <= start < end:
                spans.append((start, end))
    return spans


def make_answer_window(context: str, start: int, end: int, max_chars: int) -> str:
    if max_chars <= 0 or len(context) <= max_chars:
        return context.strip()
    center = (start + end) // 2
    left = max(0, min(center - max_chars // 2, len(context) - max_chars))
    right = min(len(context), left + max_chars)
    return context[left:right].strip()


def choose_newsqa_example(example: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    question = str(example.get("question") or "").strip()
    context = str(example.get("context") or "")
    refs = normalize_reference_answers(example.get("answers"))
    if not question or not context or not refs:
        return None
    for start, end in iter_label_spans(example.get("labels")):
        if end > len(context):
            continue
        answer = context[start:end].strip()
        words = re.findall(r"\w+", answer)
        if not answer or len(answer) > args.max_answer_chars or len(words) > args.max_answer_words:
            continue
        passage = make_answer_window(context, start, end, args.context_window_chars)
        if normalize_for_match(answer) not in normalize_for_match(passage):
            continue
        answers = [answer] + [ref for ref in refs if normalize_for_match(ref) != normalize_for_match(answer)]
        return {"question": question, "answer": answer, "answers": answers, "context": passage}
    return None


def clean_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"<think\b[^>]*>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?think\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text = text.strip()
    continuation = re.search(r"\n\s*(Question|Passage|Answer|Context|Score)\s*:", text, flags=re.IGNORECASE)
    if continuation:
        text = text[: continuation.start()].strip()
    return text.strip().strip('"')


def truncate_middle(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    marker = "\n...\n"
    keep_start = max(1, int(max_chars * 0.65))
    keep_end = max(1, max_chars - keep_start - len(marker))
    return text[:keep_start].rstrip() + marker + text[-keep_end:].lstrip()


def free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# =========================
# Paths / args
# =========================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete dataset-neutral RAM-style Self-Demo training.")

    parser.add_argument("--dataset", choices=["msmarco", "newsqa"], required=True)
    parser.add_argument("--stage", choices=["all", "prepare", "vllm", "train"], default="all")
    parser.add_argument("--base-model", default=os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL))
    parser.add_argument("--run-tag", default=os.environ.get("RUN_TAG", DEFAULT_RUN_TAG))
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-root", default=os.environ.get("CACHE_ROOT", DEFAULT_CACHE_ROOT))
    parser.add_argument("--self-demo-jsonl", default=os.environ.get("SELF_DEMO_JSONL"))
    parser.add_argument("--vllm-python", default=os.environ.get("VLLM_PYTHON_BIN", DEFAULT_VLLM_PYTHON))
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))

    parser.add_argument("--max-examples", type=int, default=int(os.environ.get("SELF_DEMO_MAX_EXAMPLES", "75000")))
    parser.add_argument("--train-max-samples", type=int, default=int(os.environ.get("TRAIN_MAX_SAMPLES", "75000")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument("--msmarco-split", default=os.environ.get("MSMARCO_SPLIT", "train"))
    parser.add_argument("--newsqa-split", default=os.environ.get("NEWSQA_SPLIT", "train"))
    parser.add_argument("--context-window-chars", type=int, default=1400)
    parser.add_argument("--max-answer-words", type=int, default=16)
    parser.add_argument("--max-answer-chars", type=int, default=100)
    parser.add_argument("--include-unselected-passages", type=int, default=0)
    parser.add_argument("--target-mode", choices=["selected", "gold"], default=os.environ.get("TARGET_MODE", "selected"))
    parser.add_argument("--min-reference-f1", type=float, default=0.70)
    parser.add_argument("--min-reference-precision", type=float, default=0.50)
    parser.add_argument("--min-support-recall", type=float, default=0.75)

    parser.add_argument("--prompt-opt-steps", type=int, default=int(os.environ.get("PROMPT_OPT_STEPS", "10")))
    parser.add_argument("--prompt-opt-train-examples", type=int, default=int(os.environ.get("PROMPT_OPT_TRAIN_EXAMPLES", "100")))
    parser.add_argument("--prompt-opt-eval-examples", type=int, default=int(os.environ.get("PROMPT_OPT_EVAL_EXAMPLES", "100")))
    parser.add_argument("--prompt-opt-topk", type=int, default=int(os.environ.get("PROMPT_OPT_TOPK", "4")))
    parser.add_argument("--prompt-opt-beam-size", type=int, default=int(os.environ.get("PROMPT_OPT_BEAM_SIZE", "12")))
    parser.add_argument("--prompt-opt-shuffle-window", type=int, default=400)
    parser.add_argument("--rewrite-example-count", type=int, default=4)
    parser.add_argument("--prompts-per-strat", type=int, default=int(os.environ.get("PROMPTS_PER_STRAT", "3")))

    parser.add_argument("--ndocs", type=int, default=4)
    parser.add_argument("--vllm-batch-size", type=int, default=1000)
    parser.add_argument("--judge-batch-size", type=int, default=1000)
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.8")))
    parser.add_argument("--vllm-enforce-eager", type=int, default=int(os.environ.get("VLLM_ENFORCE_EAGER", "1")))
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gen-max-tokens", type=int, default=1000)
    parser.add_argument("--score-max-tokens", type=int, default=32)
    parser.add_argument("--rewrite-max-tokens", type=int, default=1000)
    parser.add_argument("--judge-max-tokens", type=int, default=128)

    parser.add_argument("--max-length", type=int, default=int(os.environ.get("MAX_LENGTH", "4096")))
    parser.add_argument("--max-prompt-length", type=int, default=int(os.environ.get("MAX_PROMPT_LENGTH", "3084")))
    parser.add_argument("--per-device-train-batch-size", type=int, default=int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1")))
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LEARNING_RATE", "5e-6")))
    parser.add_argument("--num-train-epochs", type=float, default=float(os.environ.get("NUM_TRAIN_EPOCHS", "1.0")))
    parser.add_argument("--save-steps", type=int, default=int(os.environ.get("SAVE_STEPS", "100")))
    parser.add_argument("--logging-steps", type=int, default=int(os.environ.get("LOGGING_STEPS", "20")))
    parser.add_argument("--warmup-ratio", type=float, default=float(os.environ.get("WARMUP_RATIO", "0.03")))
    parser.add_argument("--save-total-limit", type=int, default=int(os.environ.get("SAVE_TOTAL_LIMIT", "12")))
    parser.add_argument("--dataloader-num-workers", type=int, default=int(os.environ.get("DATALOADER_NUM_WORKERS", "4")))
    parser.add_argument("--optim", default=os.environ.get("OPTIM", "adamw_bnb_8bit"))
    parser.add_argument("--deepspeed-config", default=os.environ.get("DEEPSPEED_CONFIG"))
    parser.add_argument("--no-resume", action="store_true")

    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-prompts", action="store_true")
    parser.add_argument("--force-self-demos", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--continued", action="store_true")
    args = parser.parse_args()
    if args.data_root == DEFAULT_DATA_ROOT:
        args.data_root = f"./runs/self_demo/{args.dataset}/{args.run_tag}/data"
    if args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = f"./runs/self_demo/{args.dataset}/{args.run_tag}/model"
    if args.cache_root == DEFAULT_CACHE_ROOT:
        args.cache_root = f"./runs/self_demo/{args.dataset}/{args.run_tag}/cache"
    if not args.wandb_name:
        args.wandb_name = f"Qwen3-14B-{args.dataset}-ram-sd-rait"
    return args


def derived_paths(args: argparse.Namespace) -> Dict[str, Path]:
    data_root = Path(args.data_root)
    prompt_dir = data_root / "optimized_prompts"
    return {
        "data_root": data_root,
        "ra_dit_jsonl": data_root / f"data/{args.dataset}_train_ra_dit.jsonl",
        "prompt_dir": prompt_dir,
        "norag_prompts": prompt_dir / "norag.jsonl",
        "rag_prompts": prompt_dir / "rag.jsonl",
        "self_demo_jsonl": Path(args.self_demo_jsonl) if args.self_demo_jsonl else data_root / f"self_demos/{args.dataset}_train_self_demos.jsonl",
        "self_demo_summary": data_root / f"self_demos/{args.dataset}_train_self_demos.summary.json",
        "output_dir": Path(args.output_dir),
        "cache_root": Path(args.cache_root),
    }


def set_runtime_env(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ["WANDB_NAME"] = args.wandb_name


# =========================
# Stage 1: prepare MS MARCO
# =========================


def passage_items(example: Dict[str, Any], include_unselected: int) -> List[Dict[str, Any]]:
    passages_obj = example.get("passages", {}) or {}
    texts = passages_obj.get("passage_text", []) or []
    selected = passages_obj.get("is_selected", []) or []
    items = []

    for index, (text, flag) in enumerate(zip(texts, selected)):
        text = str(text or "").strip()
        if text and int(flag) == 1:
            items.append({"text": text, "title": f"selected_passage_{index}", "score": 1.0})

    extra_count = 0
    for index, (text, flag) in enumerate(zip(texts, selected)):
        if include_unselected <= 0 or extra_count >= include_unselected:
            break
        text = str(text or "").strip()
        if not text or int(flag) == 1:
            continue
        items.append({"text": text, "title": f"retrieved_passage_{index}", "score": 0.0})
        extra_count += 1

    return items


def existing_ids(path: Path) -> set:
    if not path.exists():
        return set()
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("id"):
                seen.add(str(row["id"]))
    return seen


def prepare_msmarco_jsonl(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    from datasets import load_dataset
    from tqdm import tqdm

    output_path = paths["ra_dit_jsonl"]
    ensure_dir(output_path.parent)

    if output_path.exists() and output_path.stat().st_size > 0 and not args.force_prepare:
        print(f"[prepare:skip] {output_path}", flush=True)
        return

    dataset = load_dataset("microsoft/ms_marco", "v1.1", split=args.msmarco_split)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)

    mode = "w"
    seen = set()
    written = 0
    skipped_existing = 0
    with output_path.open(mode, encoding="utf-8") as handle:
        for source_idx in tqdm(indices, desc="Preparing MS MARCO"):
            example = dataset[source_idx]
            query_id = str(example.get("query_id", source_idx))
            row_id = f"msmarco-{query_id}"
            if row_id in seen:
                skipped_existing += 1
                continue

            question = str(example.get("query", "")).strip()
            answers = normalize_reference_answers(
                example.get("answers", []) or example.get("wellFormedAnswers", []) or []
            )
            retrieved_support = passage_items(example, args.include_unselected_passages)
            if not question or not answers or not retrieved_support:
                continue

            row = {
                "id": row_id,
                "source_idx": source_idx,
                "query_id": query_id,
                "dataset": "microsoft/ms_marco",
                "config": "v1.1",
                "split": args.msmarco_split,
                "question": question,
                "answer": answers[0],
                "answers": answers,
                "retrieved_support": retrieved_support,
                "metadata": {"query_type": example.get("query_type", "")},
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if args.max_examples > 0 and written >= args.max_examples:
                break

    summary = {
        "output_jsonl": str(output_path),
        "written": written,
        "skipped_existing": skipped_existing,
        "max_examples": args.max_examples,
        "seed": args.seed,
        "split": args.msmarco_split,
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


# =========================
# vLLM helpers
# =========================


def prepare_newsqa_jsonl(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    from datasets import load_dataset
    from tqdm import tqdm

    output = paths["ra_dit_jsonl"]
    ensure_dir(output.parent)
    if output.exists() and output.stat().st_size > 0 and not args.force_prepare:
        print(f"[prepare:skip] {output}", flush=True)
        return
    dataset = load_dataset("lucadiliello/newsqa", split=args.newsqa_split)
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for source_idx in tqdm(indices, desc="Preparing NewsQA"):
            prepared = choose_newsqa_example(dataset[source_idx], args)
            if prepared is None:
                continue
            row = {
                "id": f"newsqa-{dataset[source_idx].get('key', source_idx)}",
                "source_idx": source_idx,
                "dataset": "lucadiliello/newsqa",
                "split": args.newsqa_split,
                "question": prepared["question"],
                "answer": prepared["answer"],
                "answers": prepared["answers"],
                "retrieved_support": [{"text": prepared["context"], "title": "newsqa_answer_window", "score": 1.0}],
                "context": prepared["context"],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if args.max_examples > 0 and written >= args.max_examples:
                break
    print(f"[prepare] wrote {written} rows to {output}", flush=True)


def prepare_dataset_jsonl(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    if args.dataset == "msmarco":
        prepare_msmarco_jsonl(args, paths)
    else:
        prepare_newsqa_jsonl(args, paths)


def import_vllm():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    return AutoTokenizer, LLM, SamplingParams


def create_vllm(args: argparse.Namespace):
    AutoTokenizer, LLM, _SamplingParams = import_vllm()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    llm = LLM(
        model=args.base_model,
        tokenizer=args.base_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=bool(args.vllm_enforce_eager),
    )
    return llm, tokenizer


def format_chat(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content":  user},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n{user}"


def format_chat_limited(tokenizer, system: str, user: str, max_model_len: int, reserve_tokens: int) -> str:
    token_limit = max(1, max_model_len - max(1, reserve_tokens))
    rendered = format_chat(tokenizer, system, user)
    token_count = len(tokenizer(rendered, add_special_tokens=False).input_ids)
    if token_count <= token_limit:
        return rendered

    char_budget = max(200, int(len(user) * token_limit / max(1, token_count) * 0.90))
    while char_budget >= 200:
        rendered = format_chat(tokenizer, system, truncate_middle(user, char_budget))
        token_count = len(tokenizer(rendered, add_special_tokens=False).input_ids)
        if token_count <= token_limit:
            return rendered
        char_budget = int(char_budget * 0.82)

    return format_chat(tokenizer, system, truncate_middle(user, 200))


def passages(row: Dict[str, Any], ndocs: int) -> List[str]:
    supports = row.get("retrieved_support", []) or []
    return [str(item.get("text", "")).strip() for item in supports[:ndocs] if str(item.get("text", "")).strip()]


def answers(row: Dict[str, Any]) -> List[str]:
    values = row.get("answers") or [row.get("answer", "")]
    return [str(value).strip() for value in values if str(value).strip()]


def format_contexts_for_ram(row: Dict[str, Any], ndocs: int) -> str:
    ctxs = row.get("retrieved_support", [])[:ndocs]
    lines = []
    for idx, ctx in enumerate(ctxs, start=1):
        if isinstance(ctx, dict):
            text = str(ctx.get("text", "")).strip()
        else:
            text = str(ctx).strip()
        if text:
            lines.append(f"Passage {idx}: {text}")
    return "\n\n".join(lines)


def user_prompt(row: Dict[str, Any], rag: bool, ndocs: int) -> str:
    if not rag:
        return row["question"]

    return (
        f"Question: {row['question']}\n\n"
        f"Passages:\n{format_contexts_for_ram(row, ndocs)}\n\n"
        "Answer:"
    )
# =========================
# Stage 2a: prompt optimization
# =========================


def parse_score(text: str) -> float:
    match = re.search(r"Score:\s*(?P<score>[1-5])", text or "", flags=re.IGNORECASE)
    if match is None:
        return 0.0
    return float(int(match["score"]))


def score_prompt(row: Dict[str, Any], prediction: str, rag: bool, ndocs: int) -> str:
    return (
        f"Question:\n{row['question']}\n\n"
        f"Gold answer:\n{row['answer']}\n\n"
        f"Prediction:\n{prediction or '[blank]'}\n\n"
        "Score the prediction. Reply exactly as: Score: N"
    )


def generate_predictions(
    llm,
    tokenizer,
    prompts: Sequence[str],
    dataset: Sequence[Dict[str, Any]],
    rag: bool,
    ndocs: int,
    sampling_params,
    batch_size: int,
    max_model_len: int,
) -> List[List[str]]:
    from tqdm import tqdm

    predictions_by_prompt = []
    users = [user_prompt(row, rag=rag, ndocs=ndocs) for row in dataset]
    for prompt in tqdm(prompts, desc="Predicting responses"):
        rendered = [
            format_chat_limited(tokenizer, prompt, user, max_model_len, sampling_params.max_tokens)
            for user in users
        ]
        outputs = []
        for batch in batched(rendered, batch_size):
            outputs.extend(llm.generate(list(batch), sampling_params, use_tqdm=False))
        predictions_by_prompt.append([clean_text(output.outputs[0].text if output.outputs else "") for output in outputs])
    return predictions_by_prompt


def get_scores(
    llm,
    tokenizer,
    dataset: Sequence[Dict[str, Any]],
    predictions: Sequence[Sequence[str]],
    rag: bool,
    ndocs: int,
    sampling_params,
    batch_size: int,
    max_model_len: int,
) -> List[List[float]]:
    from tqdm import tqdm

    scores_by_prompt = []
    for prediction_list in tqdm(predictions, desc="Scoring prompts"):
        rendered = [
            format_chat_limited(
                tokenizer,
                SCORE_SYSTEM,
                score_prompt(row, pred, rag=rag, ndocs=ndocs),
                max_model_len,
                sampling_params.max_tokens,
            )
            for row, pred in zip(dataset, prediction_list)
        ]
        outputs = []
        for batch in batched(rendered, batch_size):
            outputs.extend(llm.generate(list(batch), sampling_params, use_tqdm=False))
        scores_by_prompt.append([parse_score(output.outputs[0].text if output.outputs else "") for output in outputs])
    return scores_by_prompt


def top_k(items: Sequence[Any], scores: Sequence[float], k: int) -> Tuple[List[Any], List[float]]:
    pairs = list(zip(items, scores))
    random.shuffle(pairs)
    pairs.sort(key=lambda item: item[1], reverse=True)
    pairs = pairs[: min(k, len(pairs))]
    return [item for item, _score in pairs], [score for _item, score in pairs]


def selected_examples_for_rewrite(
    rows: Sequence[Dict[str, Any]],
    predictions: Sequence[str],
    scores: Sequence[float],
    samples: int,
) -> List[Tuple[Dict[str, Any], str, float]]:
    triples = list(zip(rows, predictions, scores))
    triples = random.sample(triples, k=len(triples))
    triples = sorted(triples, key=lambda item: item[2], reverse=True)
    return triples[:samples]


def rewrite_user_prompt(
    old_prompt: str,
    examples: Sequence[Tuple[Dict[str, Any], str, float]],
    rag: bool,
    ndocs: int,
) -> str:
    blocks = [
        "Current system prompt:",
        old_prompt,
        "",
        "Below are examples of how this prompt performed. "
        "Use them to write a better system prompt.",
        "",
    ]

    for row, prediction, score in examples:
        blocks.append("Example:")
        blocks.append(f"Question: {truncate_middle(row['question'], 400)}")
        if rag:
            blocks.append("Retrieved passages:")
            blocks.append(truncate_middle(format_contexts_for_ram(row, ndocs), 1000))
        blocks.append(f"Gold answer: {row['answer']}")
        blocks.append(f"Prediction: {truncate_middle(prediction or '[blank]', 500)}")
        blocks.append(f"Score: {score:g}")
        blocks.append("Critique: Identify why the prediction succeeded or failed.")
        blocks.append("")

    blocks.append(
        "Now write one improved system prompt for future question-answering examples. "
        "The prompt must be general. Do not mention these examples, scores, or gold answers. "
        "Return only the improved system prompt."
    )

    return "\n".join(blocks)


def parse_system_prompt(text: str) -> str:
    prompt = str(text or "").strip()
    if ":" in prompt and re.match(r"Here is|Here's", prompt):
        return ":".join(prompt.split(":")[1:]).strip().strip('"')
    return prompt




def dedupe_valid_prompts(prompts: Sequence[str], rag: Optional[bool] = None) -> List[str]:
    seen = set()
    kept = []
    for prompt in prompts:
        prompt = parse_system_prompt(prompt)
        key = re.sub(r"\s+", " ", prompt.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            kept.append(prompt)
    return kept


def rewrite_prompts(
    llm,
    tokenizer,
    prompts: Sequence[str],
    dataset: Sequence[Dict[str, Any]],
    predictions: Sequence[Sequence[str]],
    scores: Sequence[Sequence[float]],
    samples: int,
    rag: bool,
    ndocs: int,
    sampling_params,
    max_model_len: int,
) -> List[str]:
    rendered = []
    for prompt, pred_list, score_list in zip(prompts, predictions, scores):
        examples = selected_examples_for_rewrite(dataset, pred_list, score_list, samples=samples)
        rendered.append(
            format_chat_limited(
                tokenizer,
                REWRITE_SYSTEM,
                rewrite_user_prompt(prompt, examples, rag=rag, ndocs=ndocs),
                max_model_len,
                sampling_params.max_tokens,
            )
        )
    outputs = llm.generate(rendered, sampling_params, use_tqdm=True)
    rewritten = []
    for output in outputs:
        for item in output.outputs:
            prompt = parse_system_prompt(item.text)
            if prompt.strip():
                rewritten.append(prompt.strip())
    return rewritten


def optimize_mode(
    llm,
    tokenizer,
    train_set: Sequence[Dict[str, Any]],
    eval_set: Sequence[Dict[str, Any]],
    mode: str,
    initial_prompt: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    _AutoTokenizer, _LLM, SamplingParams = import_vllm()

    rag = mode == "rag"
    prompts = [initial_prompt]
    all_prompts = []

    gen_params = SamplingParams(
        temperature=0.5,
        top_p=0.95,
        max_tokens=args.gen_max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )
    score_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.score_max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )

    for step in range(args.prompt_opt_steps):
        print(f"[optimize:{mode}] step={step + 1}/{args.prompt_opt_steps} prompts={len(prompts)}", flush=True)
        predictions = generate_predictions(
            llm, tokenizer, prompts, train_set, rag, args.ndocs, gen_params, args.vllm_batch_size, args.max_model_len
        )
        pred_scores = get_scores(
            llm, tokenizer, train_set, predictions, rag, args.ndocs, score_params, args.vllm_batch_size, args.max_model_len
        )
        prompt_scores = [statistics.mean(scores) if scores else 0.0 for scores in pred_scores]
        print(f"[optimize:{mode}] scores={prompt_scores}", flush=True)

        top_prompts, _top_scores = top_k(prompts, prompt_scores, args.prompt_opt_topk)
        top_indices = [prompts.index(prompt) for prompt in top_prompts]
        top_predictions = [predictions[idx] for idx in top_indices]
        top_pred_scores = [pred_scores[idx] for idx in top_indices]
        all_prompts.extend(dedupe_valid_prompts(prompts, rag=rag))

        topk_count = max(1, len(top_prompts))
        num_rewrites = max(1, args.prompt_opt_beam_size // topk_count)
        examples_per_rewrite = num_rewrites
        rewrite_params = SamplingParams(
            temperature=1.0,
            top_p=0.95,
            max_tokens=args.rewrite_max_tokens,
            n=num_rewrites,
            stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
        )
        rewritten_prompts = rewrite_prompts(
            llm,
            tokenizer,
            top_prompts,
            train_set,
            top_predictions,
            top_pred_scores,
            examples_per_rewrite,
            rag,
            args.ndocs,
            rewrite_params,
            args.max_model_len,
        )
        prompts = rewritten_prompts
        if not prompts:
            prompts = top_prompts

    all_prompts = dedupe_valid_prompts(all_prompts, rag=rag)
    print(f"[optimize:{mode}] validating {len(all_prompts)} prompts", flush=True)
    val_predictions = generate_predictions(
        llm, tokenizer, all_prompts, eval_set, rag, args.ndocs, gen_params, args.vllm_batch_size, args.max_model_len
    )
    val_scores = get_scores(
        llm, tokenizer, eval_set, val_predictions, rag, args.ndocs, score_params, args.vllm_batch_size, args.max_model_len
    )
    val_prompt_scores = [statistics.mean(scores) if scores else 0.0 for scores in val_scores]
    ranked = list(zip(all_prompts, val_prompt_scores))
    random.shuffle(ranked)
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [
        {"prompt": prompt, "score": score, "rank": rank + 1, "mode": mode}
        for rank, (prompt, score) in enumerate(ranked)
    ]


def optimize_prompt_files(llm, tokenizer, args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    norag_path = paths["norag_prompts"]
    rag_path = paths["rag_prompts"]
    if not args.force_prompts and norag_path.exists() and rag_path.exists():
        print(f"[prompts:skip] {paths['prompt_dir']}", flush=True)
        return

    random.seed(args.seed)
    rows = read_jsonl(paths["ra_dit_jsonl"], args.prompt_opt_shuffle_window)
    needed = args.prompt_opt_train_examples + args.prompt_opt_eval_examples
    if len(rows) < needed:
        raise ValueError(f"Need at least {needed} rows for prompt optimization, found {len(rows)}")

    sampled = random.sample(rows, k=needed)
    train_set = sampled[: args.prompt_opt_train_examples]
    eval_set = sampled[-args.prompt_opt_eval_examples :]

    norag_prompts = optimize_mode(llm, tokenizer, train_set, eval_set, "norag", INITIAL_NO_RAG_PROMPT, args)
    write_jsonl(norag_path, norag_prompts)
    print(f"[prompts] wrote {norag_path}", flush=True)

    rag_prompts = optimize_mode(llm, tokenizer, train_set, eval_set, "rag", INITIAL_RAG_PROMPT, args)
    write_jsonl(rag_path, rag_prompts)
    print(f"[prompts] wrote {rag_path}", flush=True)


# =========================
# Stage 2b: build self demos
# =========================





def load_prompt_jsonl(path: Optional[Path], fallback: List[str], limit: int, require_grounding: bool = False) -> List[str]:
    if not path or not path.exists():
        return fallback[:limit]

    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = str(obj.get("prompt", "")).strip()
            if prompt:
                prompts.append(prompt)
            if limit > 0 and len(prompts) >= limit:
                break

    return prompts or fallback[:limit]


def strategy_prompts(prompts_per_strat: int, no_rag_prompts: List[str], rag_prompts: List[str]) -> List[Tuple[str, str, bool]]:
    no_rag = [(f"NoRag{i}", prompt, False) for i, prompt in enumerate(no_rag_prompts[:prompts_per_strat])]
    rag = [(f"Rag{i}", prompt, True) for i, prompt in enumerate(rag_prompts[:prompts_per_strat])]

    # GitHub RAM adds exactly 3 refusal RAG strategies.
    refusals = [(f"RagRefuse{i}", REFUSAL_PROMPT, True) for i in range(3)]

    return no_rag + rag + refusals


def rag_user(row: Dict[str, Any], ndocs: int) -> str:
    return (
        f"Question: {row['question']}\n\n"
        f"Passages:\n{format_contexts_for_ram(row, ndocs)}\n\n"
        "Answer:"
    )


def no_rag_user(row: Dict[str, Any]) -> str:
    return row["question"]

def generate_texts(llm, prompts: Sequence[str], sampling_params, batch_size: int = 1000) -> List[str]:
    outputs = []
    for batch in batched(prompts, batch_size):
        outputs.extend(llm.generate(list(batch), sampling_params, use_tqdm=False))
    return [output.outputs[0].text if output.outputs else "" for output in outputs]


def parse_ram_summary(text: str) -> str:
    text = str(text or "").strip()
    if re.match(r"Here is|Here's", text, flags=re.DOTALL):
        text = ":".join(text.split(":")[1:]).strip()
    return clean_text(text)


def generate_for_strategy(
    llm,
    tokenizer,
    rows: List[Dict[str, Any]],
    strategy_name: str,
    system_prompt: str,
    use_rag: bool,
    ndocs: int,
    sampling_params,
) -> List[str]:
    users = [rag_user(row, ndocs) if use_rag else no_rag_user(row) for row in rows]

    first_pass_prompts = [
        format_chat(tokenizer, system_prompt, user)
        for user in users
    ]

    first_outputs = generate_texts(
        llm,
        first_pass_prompts,
        sampling_params,
        batch_size=1000,
    )

    summarize_prompts = [
        format_chat(tokenizer, SUMMARIZE_SYSTEM, output)
        for output in first_outputs
    ]

    summarized_outputs = generate_texts(
        llm,
        summarize_prompts,
        sampling_params,
        batch_size=1000,
    )

    predictions = [parse_ram_summary(output) for output in summarized_outputs]
    print(f"[strategy] {strategy_name} generated {len(predictions)} predictions", flush=True)
    return predictions


def judge_user(row: Dict[str, Any], candidates: List[Tuple[str, str]], ndocs: int) -> str:
    candidate_text = "\n".join(
        f"Prediction {idx}: {prediction}"
        for idx, (_source, prediction) in enumerate(candidates, start=1)
        if prediction is not None
    )

    return (
        f"Question:\n{row['question']}\n\n"
        f"Gold answer:\n{row['answer']}\n\n"
        f"Retrieved passages:\n{format_contexts_for_ram(row, ndocs)}\n\n"
        f"Predictions:\n{candidate_text}\n\n"
        "Which prediction is best for supervised fine-tuning? "
        "Reply exactly as: Best: Prediction N"
    )


def parse_choice(text: str, max_idx: int) -> Optional[int]:
    match = re.search(r"Best:(?: ?Prediction) (?P<idx>\d)", text or "")
    if match is None:
        return None

    idx = int(match["idx"])
    if 1 <= idx <= max_idx:
        return idx - 1
    return None


def choose_best_with_retries(
    llm,
    tokenizer,
    rows: List[Dict[str, Any]],
    preds_lists: List[List[str]],
    judge_params,
) -> List[Optional[str]]:
    user_prompts = []
    valid_preds_lists = []

    for row, preds in zip(rows, preds_lists):
        valid_preds = [pred for pred in preds if pred is not None]
        valid_preds_lists.append(valid_preds)
        candidates = [(str(i), pred) for i, pred in enumerate(valid_preds)]
        user_prompts.append(judge_user(row, candidates, ndocs=4))

    parsed_choices: List[Optional[int]] = [None for _ in user_prompts]

    for _attempt in range(5):
        unfinished_indices = [i for i, choice in enumerate(parsed_choices) if choice is None]
        if not unfinished_indices:
            break

        prompts = [
            format_chat(tokenizer, JUDGE_SYSTEM, user_prompts[i])
            for i in unfinished_indices
        ]

        outputs = llm.generate(prompts, judge_params, use_tqdm=True)

        for idx, output in zip(unfinished_indices, outputs):
            raw = output.outputs[0].text if output.outputs else ""
            parsed_choices[idx] = parse_choice(raw, len(valid_preds_lists[idx]))

    selected = []
    for choice, preds in zip(parsed_choices, valid_preds_lists):
        if choice is None:
            selected.append(None)
        else:
            selected.append(preds[choice])
    return selected


def tournament_judge(
    llm,
    tokenizer,
    rows: List[Dict[str, Any]],
    all_predictions: List[List[str]],
    strategies: List[Tuple[str, str, bool]],
    ndocs: int,
    judge_params,
    bracket_size: int = 3,
) -> Tuple[List[Optional[str]], List[Optional[str]], List[List[str]]]:

    strategy_names = [strategy[0] for strategy in strategies]

    strat_lookups = [
        dict(zip(line_preds, strategy_names))
        for line_preds in zip(*all_predictions)
    ]

    preds = np.array(all_predictions, dtype=object)  # strategies × rows
    preds_shuf = np.random.default_rng().permuted(preds, axis=0)

    while len(preds_shuf) > bracket_size:
        winner_preds_list = []

        for bracket in batched(list(preds_shuf), bracket_size):
            bracket_preds_by_row = np.array(bracket, dtype=object).T.tolist()
            winner_preds = choose_best_with_retries(
                llm,
                tokenizer,
                rows,
                bracket_preds_by_row,
                judge_params,
            )
            winner_preds_list.append(winner_preds)

        preds_shuf = np.array(winner_preds_list, dtype=object)

    final_preds_by_row = preds_shuf.T.tolist()
    selected_predictions = choose_best_with_retries(
        llm,
        tokenizer,
        rows,
        final_preds_by_row,
        judge_params,
    )

    selected_sources = [
        lookup.get(prediction) if prediction is not None else None
        for lookup, prediction in zip(strat_lookups, selected_predictions)
    ]

    candidates_by_row = [list(preds) for preds in zip(*all_predictions)]

    return selected_predictions, selected_sources, candidates_by_row


def metric_tokens(value: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").lower())


def overlap_scores(prediction: str, reference: str) -> Tuple[float, float, float]:
    pred = Counter(metric_tokens(prediction))
    ref = Counter(metric_tokens(reference))
    if not pred or not ref:
        score = float(pred == ref and bool(pred))
        return score, score, score
    overlap = sum((pred & ref).values())
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def context_support_recall(answer: str, context: str) -> float:
    answer_tokens = metric_tokens(answer)
    context_counts = Counter(metric_tokens(context))
    if not answer_tokens:
        return 0.0
    supported = sum(1 for token in answer_tokens if context_counts[token] > 0)
    return supported / len(answer_tokens)


def strict_candidate_target(
    row: Dict[str, Any],
    tournament_winner: str,
    candidates: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Select only generated candidates using the thresholds reported in the paper."""
    references = normalize_reference_answers(row.get("answers") or row.get("answer"))
    context = "\n\n".join(passages(row, args.ndocs))
    max_words = 14 if args.dataset == "msmarco" else 16
    ordered: List[Tuple[str, str]] = [("tournament", tournament_winner)]
    ordered.extend((f"candidate_{index}", value) for index, value in enumerate(candidates))
    scored = []
    seen = set()
    for source, value in ordered:
        answer = clean_text(value)
        key = " ".join(metric_tokens(answer))
        if not key or key in seen or len(answer.split()) > max_words:
            continue
        seen.add(key)
        best = (0.0, 0.0, 0.0, "")
        for reference in references:
            precision, recall, f1 = overlap_scores(answer, reference)
            if (f1, precision, recall) > (best[2], best[0], best[1]):
                best = (precision, recall, f1, reference)
        support = context_support_recall(answer, context)
        item = {
            "answer": answer,
            "source": source,
            "reference_precision": best[0],
            "reference_recall": best[1],
            "reference_f1": best[2],
            "best_reference": best[3],
            "support_recall": support,
            "word_count": len(answer.split()),
        }
        if (
            best[2] >= args.min_reference_f1
            and best[0] >= args.min_reference_precision
            and support >= args.min_support_recall
        ):
            scored.append(item)
    if not scored:
        return None, {"decision": "skip_no_strict_generated_candidate", "candidates": []}
    scored.sort(
        key=lambda item: (
            item["source"] != "tournament",
            -item["reference_f1"],
            -item["reference_precision"],
            -item["support_recall"],
            item["word_count"],
        )
    )
    chosen = scored[0]
    return str(chosen["answer"]), {"decision": "select_generated_candidate", "chosen": chosen}

def build_self_demo_jsonl(llm, tokenizer, args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    from tqdm import tqdm

    output_path = paths["self_demo_jsonl"]
    summary_path = paths["self_demo_summary"]
    ensure_dir(output_path.parent)

    if (
        output_path.exists()
        and output_path.stat().st_size > 0
        and not args.force_self_demos
        and not args.continued
    ):
        print(f"[self-demo:skip] {output_path}", flush=True)
        return
    _AutoTokenizer, _LLM, SamplingParams = import_vllm()
    rows = read_jsonl(paths["ra_dit_jsonl"], args.max_examples)

    if args.continued and output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            starting_point = sum(1 for _ in handle)
    else:
        starting_point = 0

    mode = "a" if args.continued else "w"
    no_rag_prompts = load_prompt_jsonl(paths["norag_prompts"], NO_RAG_PROMPTS, args.prompts_per_strat)
    rag_prompts = load_prompt_jsonl(paths["rag_prompts"], RAG_PROMPTS, args.prompts_per_strat, require_grounding=True)
    strategies = strategy_prompts(args.prompts_per_strat, no_rag_prompts, rag_prompts)

    gen_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1000,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )
    judge_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.judge_max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )

    source_counts = Counter()
    skipped_empty = 0
    skipped_strict = 0
    written = 0
    strategy_names = [name for name, _prompt, _use_rag in strategies]
    with output_path.open(mode, encoding="utf-8") as handle:
        filtered_rows = rows[starting_point:]
        for batch in tqdm(list(batched(filtered_rows, args.vllm_batch_size)), desc="Self-demo batches"):
            all_predictions = [
                generate_for_strategy(
                    llm,
                    tokenizer,
                    list(batch),
                    name,
                    system_prompt,
                    use_rag,
                    args.ndocs,
                    gen_params,
                )
                for name, system_prompt, use_rag in strategies
            ]
            selected_predictions, selected_sources, final_candidates = tournament_judge(
                llm, tokenizer, list(batch), all_predictions, strategies, args.ndocs, judge_params
            )
            for row, prediction, source, candidates in zip(batch, selected_predictions, selected_sources, final_candidates):
                if not clean_text(prediction or ""):
                    skipped_empty += 1
                    continue
                selected_target, selection = strict_candidate_target(
                    row, clean_text(prediction), candidates, args
                )
                if selected_target is None:
                    skipped_strict += 1
                    continue

                out = {
                    **row,
                    "prediction": selected_target,
                    "prediction_source": str(source or "tournament_winner"),
                    "predictions": candidates,
                    "prediction_sources": strategy_names,
                    "strict_target_selection": selection,
                }

                source_counts[str(out["prediction_source"])] += 1
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            print(f"[self-demo] wrote {written} rows", flush=True)

    summary = {
        "input_jsonl": str(paths["ra_dit_jsonl"]),
        "output_jsonl": str(output_path),
        "model_name_or_path": args.base_model,
        "max_examples": args.max_examples,
        "written": written,
        "skipped_empty_tournament_winner": skipped_empty,
        "skipped_no_strict_generated_candidate": skipped_strict,
        "source_counts": dict(source_counts),
        "strategies": [name for name, _prompt, _use_rag in strategies],
        "no_rag_prompts": no_rag_prompts,
        "rag_prompts": rag_prompts,
        "batch_size": args.vllm_batch_size,
        "prompts_per_strat": args.prompts_per_strat,
        "target_selection": "candidate_only_strict",
        "thresholds": {
            "minimum_reference_f1": args.min_reference_f1,
            "minimum_reference_precision": args.min_reference_precision,
            "minimum_support_recall": args.min_support_recall,
            "maximum_answer_words": 14 if args.dataset == "msmarco" else 16,
            "ndocs": args.ndocs,
        },
        "ram_reference": {
            "get_demos": "https://github.com/facebookresearch/RAM/blob/main/projects/sd-ra-it/scripts/get_demos.py",
            "prompt_optimization": "https://github.com/facebookresearch/RAM/blob/main/projects/sd-ra-it/scripts/prompt_optimization.py",
            "relevance": "https://github.com/facebookresearch/RAM/blob/main/projects/sd-ra-it/scripts/relevance.py",
            "eval": "https://github.com/facebookresearch/RAM/blob/main/projects/sd-ra-it/scripts/eval.py",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def run_vllm_stage(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    llm_needed = (
        args.continued
        or args.force_prompts
        or args.force_self_demos
        or not paths["norag_prompts"].exists()
        or not paths["rag_prompts"].exists()
        or not paths["self_demo_jsonl"].exists()
        or paths["self_demo_jsonl"].stat().st_size == 0
    )
    if not llm_needed:
        print("[vllm:skip] prompt files and self-demo data already exist", flush=True)
        return

    llm, tokenizer = create_vllm(args)
    optimize_prompt_files(llm, tokenizer, args, paths)
    build_self_demo_jsonl(llm, tokenizer, args, paths)
    del llm
    del tokenizer
    free_memory()


# =========================
# Stage 3: train
# =========================


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    from transformers import set_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def load_train_tokenizer(base_model: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer





def clean_answer(answer: object) -> str:
    answer_text = str(answer or "").strip()
    answer_text = ANSWER_PREFIX_PAT.sub("", answer_text).strip()
    return answer_text.strip().strip("*").strip()


def choose_training_answer(obj: Dict[str, Any], target_mode: str) -> str:
    if target_mode == "gold":
        refs = obj.get("answers") or []
        first_ref = refs[0] if isinstance(refs, list) and refs else obj.get("answer", "")
        return clean_answer(first_ref)

    return clean_answer(obj.get("prediction", ""))


def dataset_signature(args: argparse.Namespace, paths: Dict[str, Path]) -> str:
    stat = paths["self_demo_jsonl"].stat()
    raw = (
        f"{paths['self_demo_jsonl'].resolve()}::{stat.st_size}::{int(stat.st_mtime)}::"
        f"{args.train_max_samples}::{args.seed}::{args.target_mode}::{args.dataset}::tournament_self_demo"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def build_raw_train_dataset(args: argparse.Namespace, paths: Dict[str, Path], tokenizer):
    from datasets import Dataset, load_from_disk

    sig = dataset_signature(args, paths)
    raw_cache = paths["cache_root"] / f"self_demo_raw_{sig}"
    if raw_cache.exists() and not args.force_cache:
        print(f"[cache] loading raw dataset: {raw_cache}", flush=True)
        return load_from_disk(str(raw_cache))

    rows = []
    with paths["self_demo_jsonl"].open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            question = str(obj.get("question") or "").strip()
            context = str(obj.get("context") or "").strip()
            if not context:
                context = "\n\n".join(
                    str(item.get("text", "")).strip()
                    for item in obj.get("retrieved_support", [])[: args.ndocs]
                    if str(item.get("text", "")).strip()
                )
            answer = choose_training_answer(obj, args.target_mode)
            if not question or not context or not answer:
                continue
            base_id = str(obj.get("id") or obj.get("query_id") or len(rows))
            rows.append(
                {
                    "id": base_id,
                    "prompt": make_prompt(question, context, args.dataset),
                    "answer": answer,
                    "question": question,
                    "context": context,
                    "prompt_style": f"{args.dataset}_question_passage_answer",
                    "w_fixed": 1.0,
                    "faithfulness_fixed": 0.0,
                }
            )

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.train_max_samples > 0:
        rows = rows[: args.train_max_samples]

    dataset = Dataset.from_list(rows)
    ensure_dir(raw_cache.parent)
    dataset.save_to_disk(str(raw_cache))
    print(f"[cache] saved raw dataset: {raw_cache}", flush=True)
    return dataset


def tokenize_one(example: Dict[str, Any], tokenizer, args: argparse.Namespace) -> Dict[str, List[int]]:
    prompt = str(example["prompt"])
    answer = str(example["answer"])
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]

    if full_ids[: len(prompt_ids)] == prompt_ids:
        answer_ids = full_ids[len(prompt_ids) :]
    else:
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    if tokenizer.eos_token_id is not None and (not answer_ids or answer_ids[-1] != tokenizer.eos_token_id):
        answer_ids = answer_ids + [tokenizer.eos_token_id]

    if len(prompt_ids) > args.max_prompt_length:
        prompt_ids = prompt_ids[-args.max_prompt_length :]

    if len(prompt_ids) + len(answer_ids) > args.max_length:
        max_answer_len = max(1, args.max_length - len(prompt_ids))
        answer_ids = answer_ids[:max_answer_len]

    input_ids = prompt_ids + answer_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + answer_ids,
        "w_fixed": float(example.get("w_fixed", 1.0)),
        "faithfulness_fixed": float(example.get("faithfulness_fixed", 0.0)),
    }


def build_tokenized_train_dataset(args: argparse.Namespace, paths: Dict[str, Path], raw_dataset, tokenizer):
    from datasets import load_from_disk

    sig = dataset_signature(args, paths)
    tokenized_cache = paths["cache_root"] / f"self_demo_tok_{sig}_{args.max_length}"
    if tokenized_cache.exists() and not args.force_cache:
        print(f"[cache] loading tokenized dataset: {tokenized_cache}", flush=True)
        return load_from_disk(str(tokenized_cache))

    tokenized = raw_dataset.map(
        lambda example: tokenize_one(example, tokenizer, args),
        remove_columns=raw_dataset.column_names,
        num_proc=1,
        desc="Tokenizing self-demo rows",
    )
    ensure_dir(tokenized_cache.parent)
    tokenized.save_to_disk(str(tokenized_cache))
    print(f"[cache] saved tokenized dataset: {tokenized_cache}", flush=True)
    return tokenized


class CausalLMCollator:
    def __init__(self, tokenizer, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features):
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [self.label_pad_token_id] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "w_fixed": torch.tensor([feature.get("w_fixed", 1.0) for feature in features], dtype=torch.float32),
            "faithfulness_fixed": torch.tensor(
                [feature.get("faithfulness_fixed", 0.0) for feature in features],
                dtype=torch.float32,
            ),
        }


import torch
from transformers import Trainer as _HFTrainer


class LOSTrainer(_HFTrainer):
    def __init__(self, *args, use_weighted_logits: bool, alpha_f: float, beta: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_weighted_logits = use_weighted_logits
        self.alpha_f = alpha_f
        self.beta = beta

    def simpo_loss(self, policy_chosen_logps: torch.FloatTensor, w_fixed: torch.FloatTensor):
        pi = policy_chosen_logps.to(self.accelerator.device)
        w = w_fixed.to(self.accelerator.device).float()
        if getattr(self, "use_weighted_logits", True):
            logits = pi * w
        else:
            logits = pi
        losses = -logits
        chosen_rewards = self.beta * logits.detach()
        return losses, chosen_rewards

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        w_fixed = inputs.pop("w_fixed").to(model.device).float()
        faithfulness = inputs.pop("faithfulness_fixed").to(model.device).float()
        labels = inputs["labels"]
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        mask = shift_labels != -100
        safe_labels = shift_labels.clone()
        safe_labels[safe_labels == -100] = 0
        token_logps = torch.gather(shift_logits.log_softmax(-1), dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
        avg_logp = (token_logps * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

        weights = w_fixed * torch.exp(self.alpha_f * faithfulness)
        losses, _chosen_rewards = self.simpo_loss(avg_logp, weights)
        loss = losses.mean()
        return (loss, outputs) if return_outputs else loss


def rank_info() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def init_distributed_if_needed() -> None:
    _rank, local_rank, _world_size = rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)


def barrier_if_needed() -> None:
    _rank, _local_rank, world_size = rank_info()
    if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def is_main_process() -> bool:
    rank, _local_rank, _world_size = rank_info()
    return rank == 0


def wait_for_saved_dataset(path: Path, timeout_seconds: int = 7200) -> None:
    start = time.time()
    marker = path / "dataset_info.json"
    while time.time() - start < timeout_seconds:
        if marker.exists():
            return
        time.sleep(15)
    raise TimeoutError(f"Timed out waiting for dataset cache: {path}")


def run_training_stage(args: argparse.Namespace, paths: Dict[str, Path]) -> None:
    import torch
    import transformers.utils.import_utils as iu
    from transformers import AutoModelForCausalLM, TrainingArguments
    from transformers.trainer_utils import get_last_checkpoint

    iu.check_torch_load_is_safe = lambda *a, **k: None

    if not paths["self_demo_jsonl"].exists() or paths["self_demo_jsonl"].stat().st_size == 0:
        raise FileNotFoundError(f"Missing self-demo data: {paths['self_demo_jsonl']}")

    init_distributed_if_needed()
    ensure_dir(paths["output_dir"])
    ensure_dir(paths["cache_root"])
    seed_everything(args.seed)
    free_memory()

    tokenizer = load_train_tokenizer(args.base_model)
    if is_main_process():
        raw_dataset = build_raw_train_dataset(args, paths, tokenizer)
        tokenized_dataset = build_tokenized_train_dataset(args, paths, raw_dataset, tokenizer)
    if not is_main_process():
        sig = dataset_signature(args, paths)
        raw_cache = paths["cache_root"] / f"self_demo_raw_{sig}"
        tokenized_cache = paths["cache_root"] / f"self_demo_tok_{sig}_{args.max_length}"
        print(f"[cache] rank waiting for {raw_cache} and {tokenized_cache}", flush=True)
        wait_for_saved_dataset(raw_cache)
        wait_for_saved_dataset(tokenized_cache)
        raw_dataset = build_raw_train_dataset(args, paths, tokenizer)
        tokenized_dataset = build_tokenized_train_dataset(args, paths, raw_dataset, tokenizer)

    print(f"[train] raw rows: {len(raw_dataset)}", flush=True)
    print(f"[train] tokenized rows: {len(tokenized_dataset)}", flush=True)
    print(f"[train] target mode: {args.target_mode}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=False,
    )
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=str(paths["output_dir"]),
        overwrite_output_dir=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="no",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim=args.optim,
        remove_unused_columns=False,
        report_to=["wandb"],
        gradient_checkpointing=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,
        max_grad_norm=1.0,
        save_safetensors=True,
        seed=args.seed,
        deepspeed=args.deepspeed_config,
    )

    trainer = LOSTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=CausalLMCollator(tokenizer),
        use_weighted_logits=False,
        alpha_f=0.0,
        beta=2.0,
    )

    checkpoint = None
    if not args.no_resume and paths["output_dir"].exists():
        checkpoint = get_last_checkpoint(str(paths["output_dir"]))
    print(f"[train] resume checkpoint: {checkpoint}", flush=True)

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    robust_save_full_model(trainer, tokenizer, str(paths["output_dir"]))
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    summary = {
        "method": "ram_style_self_demonstration_generated_answers_sft_los_single_file",
        "selection_method": "ram_sd_rait_candidate_only_strict",
        "dataset": args.dataset,
        "base_model": args.base_model,
        "self_demo_jsonl": str(paths["self_demo_jsonl"]),
        "output_dir": str(paths["output_dir"]),
        "train_examples": len(tokenized_dataset),
        "target_mode": args.target_mode,
        "prompt_format": "{Query|Question}: {question}\\n\\nPassage: {context}\\n\\nAnswer:",
        "loss_source_reference": "weighted_loss_training_weighted_fix_v2.py SFT LOS path",
        "use_weighted_logits": False,
        "w_fixed": 1.0,
        "faithfulness_fixed": 0.0,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "save_steps": args.save_steps,
        "save_strategy": "no",
        "logging_steps": args.logging_steps,
        "warmup_ratio": args.warmup_ratio,
        "save_total_limit": args.save_total_limit,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "optim": args.optim,
        "model_loading": "full_bf16_causal_lm_no_lora_no_4bit",
    }
    (paths["output_dir"] / "self_demo_train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("[done] self-demo training complete", flush=True)


# =========================
# Orchestration
# =========================


def stage_command(args: argparse.Namespace, stage: str) -> List[str]:
    cmd = [
        args.vllm_python if stage == "vllm" else sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--dataset",
        args.dataset,
        "--base-model",
        args.base_model,
        "--run-tag",
        args.run_tag,
        "--data-root",
        args.data_root,
        "--output-dir",
        args.output_dir,
        "--cache-root",
        args.cache_root,
        "--self-demo-jsonl",
        args.self_demo_jsonl or "",
        "--vllm-python",
        args.vllm_python,
        "--cuda-visible-devices",
        args.cuda_visible_devices,
        "--wandb-name",
        args.wandb_name,
        "--max-examples",
        str(args.max_examples),
        "--train-max-samples",
        str(args.train_max_samples),
        "--seed",
        str(args.seed),
        "--msmarco-split",
        args.msmarco_split,
        "--newsqa-split",
        args.newsqa_split,
        "--context-window-chars",
        str(args.context_window_chars),
        "--max-answer-words",
        str(args.max_answer_words),
        "--max-answer-chars",
        str(args.max_answer_chars),
        "--include-unselected-passages",
        str(args.include_unselected_passages),
        "--target-mode",
        args.target_mode,
        "--min-reference-f1",
        str(args.min_reference_f1),
        "--min-reference-precision",
        str(args.min_reference_precision),
        "--min-support-recall",
        str(args.min_support_recall),
        "--prompt-opt-steps",
        str(args.prompt_opt_steps),
        "--prompt-opt-train-examples",
        str(args.prompt_opt_train_examples),
        "--prompt-opt-eval-examples",
        str(args.prompt_opt_eval_examples),
        "--prompt-opt-topk",
        str(args.prompt_opt_topk),
        "--prompt-opt-beam-size",
        str(args.prompt_opt_beam_size),
        "--prompt-opt-shuffle-window",
        str(args.prompt_opt_shuffle_window),
        "--rewrite-example-count",
        str(args.rewrite_example_count),
        "--prompts-per-strat",
        str(args.prompts_per_strat),
        "--ndocs",
        str(args.ndocs),
        "--vllm-batch-size",
        str(args.vllm_batch_size),
        "--judge-batch-size",
        str(args.judge_batch_size),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--vllm-enforce-eager",
        str(args.vllm_enforce_eager),
        "--max-model-len",
        str(args.max_model_len),
        "--gen-max-tokens",
        str(args.gen_max_tokens),
        "--score-max-tokens",
        str(args.score_max_tokens),
        "--rewrite-max-tokens",
        str(args.rewrite_max_tokens),
        "--judge-max-tokens",
        str(args.judge_max_tokens),
        "--max-length",
        str(args.max_length),
        "--max-prompt-length",
        str(args.max_prompt_length),
        "--per-device-train-batch-size",
        str(args.per_device_train_batch_size),
        "--per-device-eval-batch-size",
        str(args.per_device_eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--num-train-epochs",
        str(args.num_train_epochs),
        "--save-steps",
        str(args.save_steps),
        "--logging-steps",
        str(args.logging_steps),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--save-total-limit",
        str(args.save_total_limit),
        "--dataloader-num-workers",
        str(args.dataloader_num_workers),
        "--optim",
        args.optim,
    ]
    for flag in ["force_prepare", "force_prompts", "force_self_demos", "force_cache", "no_resume", "continued"]:
        if getattr(args, flag):
            cmd.append("--" + flag.replace("_", "-"))
    return cmd


def run_subprocess(cmd: List[str]) -> None:
    print("[run]", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run([str(part) for part in cmd], check=True)


def main() -> None:
    args = parse_args()
    set_runtime_env(args)
    paths = derived_paths(args)
    for key in ["data_root", "prompt_dir", "output_dir", "cache_root"]:
        ensure_dir(paths[key])

    if args.stage == "prepare":
        prepare_dataset_jsonl(args, paths)
        return

    if args.stage == "vllm":
        run_vllm_stage(args, paths)
        return

    if args.stage == "train":
        run_training_stage(args, paths)
        return

    prepare_dataset_jsonl(args, paths)
    needs_vllm = (
        args.continued
        or args.force_prompts
        or args.force_self_demos
        or not paths["norag_prompts"].exists()
        or not paths["rag_prompts"].exists()
        or not paths["self_demo_jsonl"].exists()
        or paths["self_demo_jsonl"].stat().st_size == 0
    )
    if needs_vllm:
        run_subprocess(stage_command(args, "vllm"))
    else:
        print("[all] vLLM artifacts already exist; skipping vLLM subprocess", flush=True)

    run_training_stage(args, paths)


if __name__ == "__main__":
    main()
