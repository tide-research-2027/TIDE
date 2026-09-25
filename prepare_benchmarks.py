#!/usr/bin/env python3
"""Prepare the five exact evaluation profiles reported in the TIDE paper."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset


PROFILES = {
    "msmarco": {"size": 6789, "split": "validation"},
    "hotpotqa": {"size": 7000, "split": "validation"},
    "nq": {"size": 3610, "split": "test"},
    "wow": {"size": 7866, "split": "validation"},
    "trex": {"size": 7000, "split": "validation"},
}
TREX_FILES = {
    "validation": "data/t_rex.filter_unified.min_entity_5.validation.jsonl",
    "test": "data/t_rex.filter_unified.test.jsonl",
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(PROFILES), required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=0, help="0 uses the paper size")
    return p.parse_args()


def refs(value: Any) -> List[str]:
    if isinstance(value, dict):
        value = value.get("text") or value.get("answers") or value.get("answer") or []
    if isinstance(value, str):
        value = [value]
    output = []
    for item in value or []:
        if isinstance(item, dict):
            item = item.get("text") or item.get("answer") or ""
        text = str(item or "").strip()
        if text and text not in output:
            output.append(text)
    return output


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def selected_msmarco(example: Dict[str, Any]) -> str:
    obj = example.get("passages") or {}
    for text, flag in zip(obj.get("passage_text") or [], obj.get("is_selected") or []):
        if int(flag) == 1 and str(text).strip():
            return str(text).strip()
    return ""


def msmarco_rows() -> List[Dict[str, Any]]:
    output = []
    for index, ex in enumerate(load_dataset("microsoft/ms_marco", "v1.1", split="validation")):
        answers = refs(ex.get("answers") or ex.get("wellFormedAnswers"))
        context = selected_msmarco(ex)
        question = str(ex.get("query") or "").strip()
        if question and context and answers:
            output.append(row(f"msmarco-{ex.get('query_id', index)}", "msmarco", question, context, answers, True, {}))
    return output


def hotpot_rows() -> List[Dict[str, Any]]:
    output = []
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    for index, ex in enumerate(ds):
        context = ex.get("context") or {}
        title_map = dict(zip(context.get("title") or [], context.get("sentences") or []))
        support = ex.get("supporting_facts") or {}
        grouped: Dict[str, List[tuple[int, str]]] = {}
        for title, sent_id in zip(support.get("title") or [], support.get("sent_id") or []):
            try:
                sentence = title_map[title][int(sent_id)]
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            grouped.setdefault(str(title), []).append((int(sent_id), str(sentence).strip()))
        parts = [" ".join(text for _, text in sorted(values)) for _, values in grouped.items()]
        passage = "\n\n".join(part for part in parts if part)
        answer = str(ex.get("answer") or "").strip()
        question = str(ex.get("question") or "").strip()
        if question and passage and answer:
            output.append(row(f"hotpotqa-{ex.get('id', index)}", "hotpotqa", question, passage, [answer], True, {"supporting_titles": list(grouped)}))
    return output


def nq_rows() -> List[Dict[str, Any]]:
    output = []
    ds = load_dataset("SKIML-ICL/incontext_nq_v2_dpr_ctxs", split="test")
    for index, ex in enumerate(ds):
        answers = refs(ex.get("answers"))
        ctxs = ex.get("ctxs") or []
        top = ctxs[0] if ctxs else ""
        context = str(top.get("text") if isinstance(top, dict) else top).strip()
        question = str(ex.get("question") or "").strip()
        answerable = any(norm(answer) and norm(answer) in norm(context) for answer in answers)
        if question and context and answers:
            output.append(row(f"nq-{index}", "nq", question, context, answers, answerable, {"retrieval": "DPR top-1"}))
    return output


def knowledge_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text == "no_passages_used __knowledge__ no_passages_used":
        return ""
    if "__knowledge__" in text:
        left, right = text.split("__knowledge__", 1)
        return f"{left.strip()}: {right.strip()}".strip(": ")
    return text


def wow_rows() -> List[Dict[str, Any]]:
    output = []
    ds = load_dataset("chujiezheng/wizard_of_wikipedia", split="validation")
    for conv, ex in enumerate(ds):
        posts, responses = ex.get("post") or [], ex.get("response") or []
        knowledge, labels, topics = ex.get("knowledge") or [], ex.get("labels") or [], ex.get("topics") or []
        for turn in range(min(len(posts), len(responses), len(knowledge), len(labels))):
            items = knowledge[turn] or []
            try: selected = int(labels[turn])
            except (TypeError, ValueError): selected = -1
            context = knowledge_text(items[selected]) if 0 <= selected < len(items) else ""
            if not context:
                context = "\n".join(filter(None, (knowledge_text(item) for item in items[:3])))
            topic = str(topics[turn]).strip() if turn < len(topics) else ""
            question = str(posts[turn]).strip()
            if topic and not question.startswith(topic):
                question = f"Topic: {topic}\nConversation: {question}"
            answer = str(responses[turn]).strip()
            if question and context and answer:
                output.append(row(f"wow-{conv}-{turn}", "wow", question, context, [answer], True, {"topic": topic, "turn": turn}))
    return output


def trex_rows() -> List[Dict[str, Any]]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="relbert/t_rex", repo_type="dataset", filename=TREX_FILES["validation"])
    output = []
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            ex = json.loads(line)
            head, tail, context = str(ex.get("head") or "").strip(), str(ex.get("tail") or "").strip(), str(ex.get("text") or "").strip()
            if head and tail and context:
                question = f"What entity or value is associated with {head} in the passage?"
                output.append(row(f"trex-{index}", "trex", question, context, [tail], True, {"head": head, "title": ex.get("title", "")}))
    return output


def row(identifier: str, dataset: str, question: str, context: str, answers: List[str], answerable: bool, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": identifier, "dataset": dataset, "question": question, "context": context, "answers": answers, "answer": answers[0], "answerable": answerable, "metadata": metadata}


def main() -> None:
    options = args()
    builders = {"msmarco": msmarco_rows, "hotpotqa": hotpot_rows, "nq": nq_rows, "wow": wow_rows, "trex": trex_rows}
    rows = builders[options.dataset]()
    size = options.size or PROFILES[options.dataset]["size"]
    if len(rows) < size:
        raise RuntimeError(f"{options.dataset} produced {len(rows)} rows, fewer than requested {size}")
    if len(rows) > size:
        random.Random(options.seed).shuffle(rows)
        rows = rows[:size]
    options.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with options.output_jsonl.open("w", encoding="utf-8") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {"dataset": options.dataset, "size": len(rows), "seed": options.seed, "output": str(options.output_jsonl)}
    options.output_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
