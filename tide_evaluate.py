#!/usr/bin/env python3



from __future__ import annotations



import argparse

import collections

import json

import random

import re

import string

from pathlib import Path

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple



import torch

from transformers import AutoModelForCausalLM, AutoTokenizer





GENERATION_SYSTEM = (

    "Answer the question using only the provided context. Return only a concise "

    "answer with no explanation. If the context does not support an answer, "

    "return: I don't know."

)



JUDGE_SYSTEM = """You are a strict fixed judge for retrieval-augmented question answering.

You receive QUESTION, CONTEXT, REFERENCE_ANSWERS, and MODEL_ANSWER.

Use only the supplied information; do not use outside knowledge.



Set correct=true only when MODEL_ANSWER gives the intended answer expressed by

REFERENCE_ANSWERS. Allow aliases, paraphrases, abbreviations, and harmless

formatting differences. Set correct=false for a refusal, wrong entity, partial

answer that misses essential information, contradiction, prompt copying, or

unsupported extra factual detail that materially changes the answer.



Set supported=true only when every factual claim in MODEL_ANSWER is directly

stated by or clearly entailed by CONTEXT. A context-copied sentence is not

automatically correct. Prefer false when uncertain.



Return ONLY valid JSON with exactly these keys:

{"correct": true, "supported": true, "reason": "short reason"}"""



ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)

SPACE = re.compile(r"\s+")

THINK_BLOCK = re.compile(r"\<think\b[^>]*>.*?\</think>", re.IGNORECASE | re.DOTALL)

THINK_TAG = re.compile(r"\</?think\b[^>]*>", re.IGNORECASE)

CONTINUATION = re.compile(r"\n\s*(?:Question|Context|Passage|Answer)\s*:", re.IGNORECASE)





def parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(description="Dataset-independent generation and validation.")

    p.add_argument("--stage", choices=["all", "generate", "validate"], default="all")

    p.add_argument("--input-jsonl", required=True)

    p.add_argument("--output-dir", required=True)

    p.add_argument("--model", required=True)

    p.add_argument("--model-label", default="model")

    p.add_argument("--judge-model", default="Qwen/Qwen3-32B")

    p.add_argument("--id-field", default="id")

    p.add_argument("--question-field", default="question")

    p.add_argument("--context-field", default="context")

    p.add_argument("--answers-field", default="answers")

    p.add_argument("--prediction-field", default="prediction")

    p.add_argument("--max-examples", type=int, default=0)

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch-size", type=int, default=4)

    p.add_argument("--judge-batch-size", type=int, default=4)

    p.add_argument("--max-input-tokens", type=int, default=4096)

    p.add_argument("--max-new-tokens", type=int, default=64)

    p.add_argument("--judge-max-input-tokens", type=int, default=4096)

    p.add_argument("--judge-max-new-tokens", type=int, default=128)

    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument("--top-p", type=float, default=1.0)

    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")

    p.add_argument("--device-map", default="auto")

    p.add_argument("--trust-remote-code", action="store_true")

    p.add_argument("--generation-system", default=GENERATION_SYSTEM)

    p.add_argument("--judge-system", default=JUDGE_SYSTEM)

    p.add_argument("--resume", action="store_true")

    return p.parse_args()





def dotted_get(row: Mapping[str, Any], path: str, default: Any = None) -> Any:

    value: Any = row

    for part in path.split("."):

        if not isinstance(value, Mapping) or part not in value:

            return default

        value = value[part]

    return value





def normalize_answers(value: Any) -> List[str]:

    if value is None:

        return []

    if isinstance(value, (str, int, float)):

        value = [value]

    if isinstance(value, Mapping):

        value = value.get("text", value.get("answers", value.get("answer", [])))

        if isinstance(value, (str, int, float)):

            value = [value]

    output: List[str] = []

    for item in value if isinstance(value, Sequence) else []:

        if isinstance(item, Mapping):

            item = item.get("text", item.get("answer", ""))

        text = str(item or "").strip()

        if text and text not in output:

            output.append(text)

    return output





def normalize_context(value: Any) -> str:

    if value is None:

        return ""

    if isinstance(value, str):

        return value.strip()

    if isinstance(value, Mapping):

        for key in ("text", "passage_text", "sentences", "context"):

            if key in value:

                return normalize_context(value[key])

        return "\n".join(normalize_context(v) for v in value.values() if normalize_context(v))

    if isinstance(value, Sequence):

        parts = [normalize_context(item) for item in value]

        return "\n\n".join(part for part in parts if part)

    return str(value).strip()





def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:

    with path.open("r", encoding="utf-8") as handle:

        for line_number, line in enumerate(handle, 1):

            if not line.strip():

                continue

            value = json.loads(line)

            if not isinstance(value, dict):

                raise ValueError(f"Expected an object at {path}:{line_number}")

            yield value





def standardize_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []

    for index, source in enumerate(iter_jsonl(Path(args.input_jsonl))):

        question = str(dotted_get(source, args.question_field, "") or "").strip()

        context = normalize_context(dotted_get(source, args.context_field, ""))

        answers = normalize_answers(dotted_get(source, args.answers_field, []))

        if not question:

            continue

        rows.append({

            "id": str(dotted_get(source, args.id_field, index)),

            "question": question,

            "context": context,

            "answers": answers,

            "prediction": clean_answer(dotted_get(source, args.prediction_field, "")),

            "source": source,

        })

        if args.max_examples > 0 and len(rows) >= args.max_examples:

            break

    return rows





def clean_answer(value: Any) -> str:

    text = THINK_BLOCK.sub(" ", str(value or ""))

    text = THINK_TAG.sub(" ", text).strip()

    match = CONTINUATION.search(text)

    if match:

        text = text[:match.start()]

    text = re.sub(r"^\s*(?:answer|the answer is)\s*:?\s*", "", text, flags=re.IGNORECASE)

    return SPACE.sub(" ", text).strip().strip('"')





def metric_normalize(value: Any) -> str:

    text = str(value or "").lower()

    text = "".join(ch for ch in text if ch not in string.punctuation)

    text = ARTICLES.sub(" ", text)

    return SPACE.sub(" ", text).strip()





def token_prf(prediction: str, reference: str) -> Tuple[float, float, float]:

    pred = metric_normalize(prediction).split()

    ref = metric_normalize(reference).split()

    if not pred or not ref:

        score = float(pred == ref)

        return score, score, score

    overlap = sum((collections.Counter(pred) & collections.Counter(ref)).values())

    precision = overlap / len(pred)

    recall = overlap / len(ref)

    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return precision, recall, f1





def reference_metrics(prediction: str, references: Sequence[str]) -> Dict[str, float]:

    if not references:

        return {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    candidates = []

    for reference in references:

        precision, recall, f1 = token_prf(prediction, reference)

        candidates.append({

            "exact_match": float(metric_normalize(prediction) == metric_normalize(reference)),

            "precision": precision,

            "recall": recall,

            "f1": f1,

        })

    return max(candidates, key=lambda item: (item["f1"], item["exact_match"]))





def dtype_from_name(name: str):

    return {"auto": "auto", "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]





def load_model_and_tokenizer(name: str, args: argparse.Namespace):

    tokenizer = AutoTokenizer.from_pretrained(

        name, use_fast=True, trust_remote_code=args.trust_remote_code

    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(

        name,

        torch_dtype=dtype_from_name(args.dtype),

        device_map=args.device_map,

        trust_remote_code=args.trust_remote_code,

        low_cpu_mem_usage=True,

    )

    model.eval()

    return model, tokenizer





def render_chat(tokenizer, system: str, user: str) -> str:

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if hasattr(tokenizer, "apply_chat_template"):

        try:

            return tokenizer.apply_chat_template(

                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False

            )

        except TypeError:

            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return f"System: {system}\n\nUser: {user}\n\nAssistant:"





def model_device(model) -> torch.device:

    try:

        return model.get_input_embeddings().weight.device

    except Exception:

        return next(model.parameters()).device





def generate_batches(

    model,

    tokenizer,

    prompts: Sequence[str],

    *,

    batch_size: int,

    max_input_tokens: int,

    max_new_tokens: int,

    temperature: float,

    top_p: float,

) -> List[str]:

    outputs: List[str] = []

    for start in range(0, len(prompts), batch_size):

        batch = list(prompts[start:start + batch_size])

        encoded = tokenizer(

            batch, padding=True, truncation=True, max_length=max_input_tokens, return_tensors="pt"

        ).to(model_device(model))

        generation = {

            "max_new_tokens": max_new_tokens,

            "do_sample": temperature > 0,

            "pad_token_id": tokenizer.pad_token_id,

            "eos_token_id": tokenizer.eos_token_id,

        }

        if temperature > 0:

            generation.update({"temperature": temperature, "top_p": top_p})

        with torch.inference_mode():

            generated = model.generate(**encoded, **generation)

        prompt_width = encoded["input_ids"].shape[1]

        outputs.extend(tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True))

        print(f"[progress] {min(start + len(batch), len(prompts))}/{len(prompts)}", flush=True)

    return outputs





def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:

        for row in rows:

            handle.write(json.dumps(row, ensure_ascii=False) + "\n")





def completed_ids(path: Path) -> set[str]:

    if not path.exists():

        return set()

    return {str(row.get("id")) for row in iter_jsonl(path)}





def generation_user(row: Mapping[str, Any]) -> str:

    return f"QUESTION:\n{row['question']}\n\nCONTEXT:\n{row['context']}\n\nANSWER:"





def judge_user(row: Mapping[str, Any]) -> str:

    return (

        f"QUESTION:\n{row['question']}\n\nCONTEXT:\n{row['context']}\n\n"

        f"REFERENCE_ANSWERS:\n{json.dumps(row['answers'], ensure_ascii=False)}\n\n"

        f"MODEL_ANSWER:\n{row['prediction']}"

    )





def extract_json(text: str) -> Optional[Dict[str, Any]]:

    cleaned = THINK_BLOCK.sub(" ", str(text or ""))

    candidates = re.findall(r"\{.*?\}", cleaned, flags=re.DOTALL)

    for candidate in reversed(candidates):

        try:

            value = json.loads(candidate)

        except json.JSONDecodeError:

            continue

        if isinstance(value, dict) and isinstance(value.get("correct"), bool) and isinstance(value.get("supported"), bool):

            return value

    return None





def run_generation(args: argparse.Namespace, rows: List[Dict[str, Any]], output: Path) -> None:

    done = completed_ids(output) if args.resume else set()

    if not args.resume:

        output.unlink(missing_ok=True)

    pending = [row for row in rows if row["id"] not in done]

    if not pending:

        return

    model, tokenizer = load_model_and_tokenizer(args.model, args)

    prompts = [render_chat(tokenizer, args.generation_system, generation_user(row)) for row in pending]

    predictions = generate_batches(

        model, tokenizer, prompts, batch_size=args.batch_size,

        max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens,

        temperature=args.temperature, top_p=args.top_p,

    )

    records = []

    for row, raw in zip(pending, predictions):

        prediction = clean_answer(raw)

        records.append({

            "id": row["id"], "model_label": args.model_label,

            "question": row["question"], "context": row["context"],

            "answers": row["answers"], "prediction": prediction,

            "raw_prediction": raw, **reference_metrics(prediction, row["answers"]),

        })

    append_jsonl(output, records)

    del model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()





def run_validation(args: argparse.Namespace, generations: Path, output: Path) -> None:

    rows = list(iter_jsonl(generations))

    done = completed_ids(output) if args.resume else set()

    if not args.resume:

        output.unlink(missing_ok=True)

    pending = [row for row in rows if str(row["id"]) not in done]

    if pending:

        model, tokenizer = load_model_and_tokenizer(args.judge_model, args)

        prompts = [render_chat(tokenizer, args.judge_system, judge_user(row)) for row in pending]

        decisions = generate_batches(

            model, tokenizer, prompts, batch_size=args.judge_batch_size,

            max_input_tokens=args.judge_max_input_tokens,

            max_new_tokens=args.judge_max_new_tokens, temperature=0.0, top_p=1.0,

        )

        records = []

        for row, raw in zip(pending, decisions):

            parsed = extract_json(raw)

            records.append({

                **row,

                "judge_model": args.judge_model,

                "judge_correct": parsed.get("correct") if parsed else None,

                "judge_supported": parsed.get("supported") if parsed else None,

                "judge_reason": str(parsed.get("reason", "")) if parsed else "parse_failure",

                "judge_parse_ok": parsed is not None,

                "raw_judge_output": raw,

            })

        append_jsonl(output, records)





def summarize(validated: Path, summary: Path) -> None:

    rows = list(iter_jsonl(validated))

    parsed = [row for row in rows if row.get("judge_parse_ok")]

    def mean(key: str, source: Sequence[Mapping[str, Any]] = rows) -> float:

        values = [float(row.get(key, 0.0) or 0.0) for row in source]

        return sum(values) / len(values) if values else 0.0

    result = {

        "examples": len(rows),

        "judge_parsed": len(parsed),

        "judge_parse_failures": len(rows) - len(parsed),

        "exact_match": mean("exact_match"),

        "token_precision": mean("precision"),

        "token_recall": mean("recall"),

        "token_f1": mean("f1"),

        "judge_accuracy": mean("judge_correct", parsed),

        "judge_support_rate": mean("judge_supported", parsed),

        "faithful_and_correct_rate": (

            sum(bool(row.get("judge_correct")) and bool(row.get("judge_supported")) for row in parsed) / len(parsed)

            if parsed else 0.0

        ),

    }

    summary.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2), flush=True)





def main() -> None:

    args = parse_args()

    random.seed(args.seed)

    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    generations = output_dir / "generations.jsonl"

    validated = output_dir / "validated.jsonl"

    summary = output_dir / "summary.json"

    rows = standardize_rows(args)

    if args.stage in {"all", "generate"}:

        run_generation(args, rows, generations)

    if args.stage in {"all", "validate"}:

        if not generations.exists():

            supplied = [row for row in rows if row.get("prediction")]

            if not supplied:

                raise FileNotFoundError(

                    f"Missing {generations}; run --stage generate or provide predictions via --prediction-field."

                )

            records = []

            for row in supplied:

                records.append({

                    "id": row["id"], "model_label": args.model_label,

                    "question": row["question"], "context": row["context"],

                    "answers": row["answers"], "prediction": row["prediction"],

                    "raw_prediction": row["prediction"],

                    **reference_metrics(row["prediction"], row["answers"]),

                })

            append_jsonl(generations, records)

        run_validation(args, generations, validated)

        summarize(validated, summary)





if __name__ == "__main__":

    main()
