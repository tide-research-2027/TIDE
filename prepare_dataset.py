#!/usr/bin/env python3
"""Convert arbitrary QA data into a canonical dataset-independent JSONL schema.

Output rows have:
  id, question, context, answers, answer, metadata

Sources may be JSONL, JSON, CSV/TSV, or a Hugging Face dataset. Dotted field
paths are supported. The script contains no benchmark-specific loaders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SPACE = re.compile(r"\s+")
MATCH_CLEAN = re.compile(r"[^a-z0-9]+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare a canonical QA JSONL dataset.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-file")
    source.add_argument("--hf-dataset")
    p.add_argument("--hf-config", default=None)
    p.add_argument("--hf-split", default="train")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-prefix", default="dataset")
    p.add_argument("--id-field", default="id")
    p.add_argument("--question-field", default="question")
    p.add_argument("--context-field", default="context")
    p.add_argument("--answers-field", default="answers")
    p.add_argument("--answer-text-field", default="text")
    p.add_argument("--passage-text-field", default="text")
    p.add_argument("--passage-selected-field", default=None)
    p.add_argument("--selected-value", default="1")
    p.add_argument("--metadata-fields", default="")
    p.add_argument("--delimiter", default=None)
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--validation-fraction", type=float, default=0.0)
    p.add_argument("--test-fraction", type=float, default=0.0)
    p.add_argument("--require-answer", action="store_true")
    p.add_argument("--require-context", action="store_true")
    p.add_argument("--require-answer-in-context", action="store_true")
    p.add_argument("--answer-window-chars", type=int, default=0)
    p.add_argument("--max-context-chars", type=int, default=0)
    p.add_argument("--max-answer-words", type=int, default=0)
    p.add_argument("--deduplicate", action="store_true")
    return p.parse_args()


def dotted_get(value: Any, path: Optional[str], default: Any = None) -> Any:
    if not path:
        return default
    current = value
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return default
    return current


def clean_text(value: Any) -> str:
    return SPACE.sub(" ", str(value or "")).strip()


def normalize_for_match(value: Any) -> str:
    return SPACE.sub(" ", MATCH_CLEAN.sub(" ", str(value or "").lower())).strip()


def flatten_values(value: Any, preferred_field: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float, bool)):
        text = clean_text(value)
        return [text] if text else []
    if isinstance(value, Mapping):
        if preferred_field in value:
            return flatten_values(value[preferred_field], preferred_field)
        for key in ("text", "answer", "answers", "value", "label"):
            if key in value:
                return flatten_values(value[key], preferred_field)
        output: List[str] = []
        for item in value.values():
            output.extend(flatten_values(item, preferred_field))
        return output
    if isinstance(value, Sequence):
        output = []
        for item in value:
            output.extend(flatten_values(item, preferred_field))
        return output
    text = clean_text(value)
    return [text] if text else []


def normalize_answers(value: Any, answer_text_field: str) -> List[str]:
    output: List[str] = []
    seen = set()
    for answer in flatten_values(value, answer_text_field):
        key = normalize_for_match(answer)
        if key and key not in seen:
            seen.add(key)
            output.append(answer)
    return output


def truthy_selected(value: Any, selected_value: str) -> bool:
    if isinstance(value, bool):
        return value
    return normalize_for_match(value) == normalize_for_match(selected_value)


def normalize_context(
    value: Any,
    *,
    text_field: str,
    selected_field: Optional[str],
    selected_value: str,
) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return clean_text(value)
    if isinstance(value, Mapping):
        # Column-oriented passages: {text: [...], is_selected: [...]}.
        texts = dotted_get(value, text_field)
        flags = dotted_get(value, selected_field) if selected_field else None
        if isinstance(texts, Sequence) and not isinstance(texts, str):
            text_list = [clean_text(item) for item in texts]
            if isinstance(flags, Sequence) and not isinstance(flags, str):
                chosen = [text for text, flag in zip(text_list, flags) if text and truthy_selected(flag, selected_value)]
                if chosen:
                    return "\n\n".join(chosen)
            return "\n\n".join(text for text in text_list if text)
        direct = dotted_get(value, text_field)
        if direct is not None:
            return normalize_context(
                direct, text_field=text_field, selected_field=None, selected_value=selected_value
            )
        for key in ("context", "passage", "passages", "sentences", "text"):
            if key in value:
                return normalize_context(
                    value[key], text_field=text_field,
                    selected_field=selected_field, selected_value=selected_value,
                )
        return "\n\n".join(
            part for part in (
                normalize_context(item, text_field=text_field, selected_field=None, selected_value=selected_value)
                for item in value.values()
            ) if part
        )
    if isinstance(value, Sequence):
        selected_parts: List[str] = []
        all_parts: List[str] = []
        for item in value:
            text = normalize_context(item, text_field=text_field, selected_field=None, selected_value=selected_value)
            if text:
                all_parts.append(text)
            if selected_field and isinstance(item, Mapping) and truthy_selected(dotted_get(item, selected_field), selected_value):
                if text:
                    selected_parts.append(text)
        return "\n\n".join(selected_parts or all_parts)
    return clean_text(value)


def iter_file(path: Path, encoding: str, delimiter: Optional[str]) -> Iterable[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding=encoding) as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                yield row
        return
    if suffix == ".json":
        value = json.loads(path.read_text(encoding=encoding))
        if isinstance(value, Mapping):
            for key in ("data", "rows", "examples", "items"):
                if isinstance(value.get(key), list):
                    value = value[key]
                    break
        if not isinstance(value, list):
            raise ValueError("JSON input must be a list or contain data/rows/examples/items.")
        for row in value:
            if isinstance(row, dict):
                yield row
        return
    if suffix in {".csv", ".tsv"}:
        actual_delimiter = delimiter or ("\t" if suffix == ".tsv" else ",")
        with path.open("r", encoding=encoding, newline="") as handle:
            yield from csv.DictReader(handle, delimiter=actual_delimiter)
        return
    raise ValueError(f"Unsupported input extension: {suffix}")


def iter_source(args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    if args.input_file:
        yield from iter_file(Path(args.input_file), args.encoding, args.delimiter)
        return
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` to use --hf-dataset.") from exc
    dataset = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split)
    for row in dataset:
        yield dict(row)


def answer_position(context: str, answers: Sequence[str]) -> Optional[tuple[int, int]]:
    lower = context.lower()
    for answer in sorted(answers, key=len, reverse=True):
        start = lower.find(answer.lower())
        if start >= 0:
            return start, start + len(answer)
    return None


def crop_context(context: str, answers: Sequence[str], window: int, maximum: int) -> str:
    limit = window or maximum
    if limit <= 0 or len(context) <= limit:
        return context
    position = answer_position(context, answers) if window > 0 else None
    if position:
        start, end = position
        center = (start + end) // 2
        left = max(0, min(center - limit // 2, len(context) - limit))
        return context[left:left + limit].strip()
    return context[:limit].strip()


def stable_id(question: str, context: str) -> str:
    return hashlib.sha1(f"{question}\n{context}".encode("utf-8")).hexdigest()[:16]


def prepare_row(source: Dict[str, Any], index: int, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    question = clean_text(dotted_get(source, args.question_field, ""))
    answers = normalize_answers(dotted_get(source, args.answers_field), args.answer_text_field)
    context = normalize_context(
        dotted_get(source, args.context_field),
        text_field=args.passage_text_field,
        selected_field=args.passage_selected_field,
        selected_value=args.selected_value,
    )
    if not question or (args.require_answer and not answers) or (args.require_context and not context):
        return None
    if args.max_answer_words > 0:
        answers = [answer for answer in answers if len(answer.split()) <= args.max_answer_words]
        if args.require_answer and not answers:
            return None
    if args.require_answer_in_context and (not answers or answer_position(context, answers) is None):
        return None
    context = crop_context(context, answers, args.answer_window_chars, args.max_context_chars)
    raw_id = dotted_get(source, args.id_field)
    example_id = clean_text(raw_id) if raw_id is not None else stable_id(question, context)
    metadata = {field: dotted_get(source, field) for field in args.metadata_fields.split(",") if field.strip()}
    return {
        "id": example_id or str(index),
        "question": question,
        "context": context,
        "answers": answers,
        "answer": answers[0] if answers else "",
        "metadata": metadata,
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if args.validation_fraction < 0 or args.test_fraction < 0 or args.validation_fraction + args.test_fraction >= 1:
        raise ValueError("validation-fraction and test-fraction must be nonnegative and sum to less than 1.")
    rows: List[Dict[str, Any]] = []
    seen = set()
    skipped = 0
    for index, source in enumerate(iter_source(args)):
        row = prepare_row(source, index, args)
        if row is None:
            skipped += 1
            continue
        signature = (normalize_for_match(row["question"]), normalize_for_match(row["context"]))
        if args.deduplicate and signature in seen:
            skipped += 1
            continue
        seen.add(signature)
        rows.append(row)
        if args.max_examples > 0 and len(rows) >= args.max_examples:
            break
    if args.shuffle or args.validation_fraction > 0 or args.test_fraction > 0:
        random.Random(args.seed).shuffle(rows)
    test_count = int(len(rows) * args.test_fraction)
    validation_count = int(len(rows) * args.validation_fraction)
    test = rows[:test_count]
    validation = rows[test_count:test_count + validation_count]
    train = rows[test_count + validation_count:]
    output = Path(args.output_dir)
    files: Dict[str, int] = {}
    if args.validation_fraction > 0 or args.test_fraction > 0:
        for split, values in (("train", train), ("validation", validation), ("test", test)):
            path = output / f"{args.output_prefix}_{split}.jsonl"
            write_jsonl(path, values)
            files[str(path)] = len(values)
    else:
        path = output / f"{args.output_prefix}.jsonl"
        write_jsonl(path, rows)
        files[str(path)] = len(rows)
    summary = {
        "source": args.input_file or args.hf_dataset,
        "seed": args.seed,
        "accepted": len(rows),
        "skipped": skipped,
        "files": files,
        "schema": ["id", "question", "context", "answers", "answer", "metadata"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{args.output_prefix}_preparation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
