

from __future__ import annotations



import argparse

import json

import math

import os

import re

import time

from pathlib import Path

from typing import Any, Dict, List



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





def wait_for_file(path: str | Path, timeout_seconds: int = 7200) -> None:

    path = Path(path)

    start = time.time()

    while time.time() - start < timeout_seconds:

        if path.exists() and path.stat().st_size > 0:

            return

        time.sleep(10)

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



    return clean_text(texts[0]) if texts else ""





def load_training_rows(dataset_name: str, split: str, max_samples: int) -> List[Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []



    if dataset_name == "msmarco":

        ds = load_dataset("ms_marco", "v1.1", split=split)

        if max_samples > 0:

            ds = ds.select(range(min(max_samples, len(ds))))



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



    elif dataset_name == "newsqa":

        ds = load_dataset("newsqa", split=split)

        if max_samples > 0:

            ds = ds.select(range(min(max_samples, len(ds))))



        for ex in ds:

            question = clean_text(ex.get("question"))

            context = clean_text(ex.get("story_text") or ex.get("context") or ex.get("story"))

            refs: List[str] = []



            answer = ex.get("answer")

            if isinstance(answer, dict):

                refs = extract_answers(answer.get("text") or answer.get("answer"))

            refs = refs or extract_answers(ex.get("answers"))



            if question and context and refs:

                context = context[:6000]

                rows.append(

                    {

                        "dataset": "newsqa",

                        "question": question,

                        "context": context,

                        "answer": refs[0],

                        "references": refs,

                        "prompt": qa_prompt(question, context),

                    }

                )



    else:

        raise ValueError(f"Unknown dataset: {dataset_name}")



    return rows





def answer_mean_logp(model, tokenizer, prompt: str, answer: str, device: str) -> float:

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids

    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]



    input_ids = torch.tensor([prompt_ids + answer_ids], device=device)



    with torch.no_grad():

        logits = model(input_ids).logits[:, :-1]

        labels = input_ids[:, 1:]

        logp = torch.log_softmax(logits, dim=-1)



        start = max(len(prompt_ids) - 1, 0)

        end = start + len(answer_ids)

        vals = logp[0, start:end, labels[0, start:end]]



    return float(vals.mean().cpu())





def faithfulness_score(nli_model, nli_tokenizer, context: str, answer: str, device: str) -> float:

    encoded = nli_tokenizer(

        context,

        answer,

        return_tensors="pt",

        truncation=True,

        max_length=512,

    ).to(device)



    with torch.no_grad():

        logits = nli_model(**encoded).logits[0]

        probs = torch.softmax(logits, dim=-1)



    entail_idx = [i for i, label in nli_model.config.id2label.items() if "entail" in label.lower()][0]

    return float(probs[entail_idx].cpu())





def save_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:

        for row in rows:

            handle.write(json.dumps(row, ensure_ascii=False) + "\n")





def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:

    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as handle:

        for line in handle:

            line = line.strip()

            if line:

                rows.append(json.loads(line))

    return rows





def compute_tide_weights(args) -> List[Dict[str, Any]]:

    weighted_path = Path(args.weighted_jsonl)



    if args.reuse_weighted_jsonl and weighted_path.exists():

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

    rows = load_training_rows(args.dataset, args.split, args.max_samples)

    print(f"[dataset rows] {len(rows)}")



    print(f"[load base scorer] {args.base_model}")

    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(

        args.base_model,

        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,

        trust_remote_code=True,

    ).to(device).eval()



    print(f"[load NLI scorer] {args.nli_model}")

    nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_model)

    nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model).to(device).eval()



    raw_weights: List[float] = []

    for row in tqdm(rows, desc="scoring TIDE weights"):

        logp = answer_mean_logp(base_model, base_tokenizer, row["prompt"], row["answer"], device)

        faith = faithfulness_score(nli_model, nli_tokenizer, row["context"], row["answer"], device)

        value = logp + faith

        raw_weight = math.exp(args.alpha * value)



        row["base_logp"] = logp

        row["faithfulness"] = faith

        row["value"] = value

        row["raw_weight"] = raw_weight

        raw_weights.append(raw_weight)



    mean_weight = sum(raw_weights) / max(len(raw_weights), 1)

    for row in rows:

        row["weight"] = row["raw_weight"] / mean_weight if mean_weight > 0 else 1.0



    save_jsonl(weighted_path, rows)

    print(f"[saved weighted data] {weighted_path}")

    print({"rows": len(rows), "alpha": args.alpha, "mean_raw_weight": mean_weight})



    del base_model

    del nli_model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()



    return rows





def encode_row(row: Dict[str, Any], tokenizer, max_length: int) -> Dict[str, Any]:

    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False).input_ids

    answer_ids = tokenizer(row["answer"], add_special_tokens=False).input_ids + [tokenizer.eos_token_id]



    input_ids = (prompt_ids + answer_ids)[:max_length]

    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]

    pad_len = max_length - len(input_ids)



    return {

        "input_ids": input_ids + [tokenizer.pad_token_id] * pad_len,

        "attention_mask": [1] * len(input_ids) + [0] * pad_len,

        "labels": labels + [-100] * pad_len,

        "weight": float(row.get("weight", 1.0)),

    }





class TIDETrainer(Trainer):

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



    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token



    print(f"[load train model] {args.base_model}")

    model = AutoModelForCausalLM.from_pretrained(

        args.base_model,

        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,

        trust_remote_code=True,

    )



    features = [encode_row(row, tokenizer, args.max_length) for row in weighted_rows]

    train_dataset = Dataset.from_list(features)



    training_args = TrainingArguments(

        output_dir=args.output_dir,

        learning_rate=args.learning_rate,

        num_train_epochs=args.epochs,

        per_device_train_batch_size=args.per_device_batch_size,

        gradient_accumulation_steps=args.gradient_accumulation_steps,

        warmup_ratio=args.warmup_ratio,

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

    )



    trainer = TIDETrainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        tokenizer=tokenizer,

    )



    print("[train] starting TIDE full fine-tuning")

    trainer.train()



    print(f"[save] {args.output_dir}")

    trainer.save_model(args.output_dir)

    tokenizer.save_pretrained(args.output_dir)





def parse_args():

    parser = argparse.ArgumentParser(description="Standalone TIDE training")



    parser.add_argument("--dataset", choices=["msmarco", "newsqa"], required=True)

    parser.add_argument("--split", default="train")

    parser.add_argument("--max-samples", type=int, default=75000)



    parser.add_argument("--base-model", default=BASE_MODEL)

    parser.add_argument("--nli-model", default=NLI_MODEL)

    parser.add_argument("--alpha", type=float, default=0.5)



    parser.add_argument("--weighted-jsonl", default=None)

    parser.add_argument("--reuse-weighted-jsonl", action="store_true")



    parser.add_argument("--output-dir", required=True)



    parser.add_argument("--max-length", type=int, default=4096)

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



    parser.add_argument("--seed", type=int, default=42)



    return parser.parse_args()





def main() -> None:

    args = parse_args()

    set_seed(args.seed)



    Path(args.output_dir).mkdir(parents=True, exist_ok=True)



    if args.weighted_jsonl is None:

        alpha_tag = str(args.alpha).replace(".", "p")

        args.weighted_jsonl = str(Path(args.output_dir) / f"{args.dataset}_tide_alpha{alpha_tag}_weighted.jsonl")



    weighted_rows = compute_tide_weights(args)

    train_model(args, weighted_rows)





if __name__ == "__main__":

    main()


