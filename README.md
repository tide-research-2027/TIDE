# TIDE

**Target Optimization with Implicit Distillation for ContExt-Grounded RAG**

Code accompanying **Grounded without Drifting from Pretraining: Rollout-Free Target Optimization with Implicit Distillation for RAG**.

## Overview

TIDE is a rollout-free approach to post-training retrieval-augmented language models. It retains human-written supervision while weighting training examples according to:

- Compatibility with the pretrained model.
- Faithfulness to the retrieved context.

The implementation computes a value for each response and converts it into a training weight:

```text
value = mean answer-token log probability + NLI entailment probability
raw_weight = exp(alpha * value)
weight = raw_weight / mean(raw_weight)
```

Training minimizes the weighted, length-normalized response loss. Prompt and padding tokens are excluded. At `alpha = 0`, the weights become uniform and the objective reduces to supervised fine-tuning.

## Repository Contents

| File | Description |
| --- | --- |
| `tide_train.py` | Computes TIDE weights and performs full-model fine-tuning. |
| `self-demo.py` | Runs the Self-Demo baseline, including demonstration generation, selection, and training. |
| `tide_evaluate.py` | Generates responses and evaluates reference overlap and fixed-judge decisions. |
| `faithfulness_judge.py` | Scores context faithfulness using a fixed Qwen judge. |
| `kl_divergence.py` | Computes full-vocabulary KL divergence between the base and trained models along generated responses. |

## Setup

Install PyTorch appropriate for your CUDA environment, then install the core dependencies:

```bash
python -m pip install transformers datasets accelerate tqdm numpy sentencepiece bitsandbytes
```

Use a Transformers version supporting Qwen3. Self-Demo generation additionally requires a compatible vLLM installation. DeepSpeed is optional when supplying a configuration.

Dependency versions are not currently pinned. Large-model fine-tuning and evaluation require substantial GPU memory; adjust model sizes and batch sizes to your hardware.

## TIDE Training

Example using Qwen3-4B on MSMARCO:

```bash
python tide_train.py \
  --dataset msmarco \
  --base-model Qwen/Qwen3-4B \
  --alpha 0.5 \
  --max-samples 75000 \
  --weighted-jsonl runs/tide_msmarco/weighted.jsonl \
  --output-dir runs/tide_msmarco/model
```

The default base model is `Qwen/Qwen3-14B`.

The training faithfulness scorer defaults to:

```text
MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
```

This NLI score is used to construct training weights. It is separate from the Qwen-based faithfulness evaluation described below.

Use `--dataset newsqa` for the implemented NewsQA loader. Dataset access and schemas must match the script's loaders. The sample limit is applied before filtering, so the retained training set may contain fewer examples.

To reuse computed weights, add:

```bash
--reuse-weighted-jsonl
```

Only reuse a cache when the dataset, base model, faithfulness scorer, and alpha match the intended run.

For uniform-weight SFT, use `--alpha 0` with a fresh cache and output directory. The current script still runs the scoring stage.

View all options:

```bash
python tide_train.py --help
```

## Self-Demo Baseline

Self-Demo generates and selects model-produced demonstrations before fine-tuning.

```bash
python self-demo.py \
  --dataset msmarco \
  --stage all \
  --base-model Qwen/Qwen3-4B \
  --run-tag self_demo_qwen3_4b \
  --data-root runs/self_demo_qwen3_4b/data \
  --cache-root runs/self_demo_qwen3_4b/cache \
  --output-dir runs/self_demo_qwen3_4b/model
```

Available stages are:

- `prepare`: prepare training data.
- `vllm`: run demonstration generation and selection.
- `train`: fine-tune using prepared demonstrations.
- `all`: run the full pipeline.

A separate vLLM Python environment can be specified with `--vllm-python`.

```bash
python self-demo.py --help
```

## Answer Generation and Evaluation

Prepare evaluation data as JSONL, with one example per line:

```json
{"id":"example-1","question":"What is the capital of France?","context":"Paris is the capital of France.","answers":["Paris"]}
```

Run generation and validation:

```bash
python tide_evaluate.py \
  --stage all \
  --input-jsonl data/eval.jsonl \
  --model runs/tide_msmarco/model \
  --model-label TIDE \
  --judge-model Qwen/Qwen3-32B \
  --output-dir runs/evaluation
```

The script writes:

- `generations.jsonl`
- `validated.jsonl`
- `summary.json`

Generation and validation can also be run separately using `--stage generate` and `--stage validate`.

### Metric Definitions

The current evaluation script reports:

- Exact match.
- Token-overlap precision, recall, and F1.
- Judge correctness rate.
- Judge support rate.
- Faithful-and-correct rate.

**The reported `token_f1` is different from the paper's answer-level F1.** Computing the paper's answer-level metric requires counts of correct answers, attempted answers, and answerable examples; the current summary function does not produce those counts.

Judge-based summary rates exclude judge parse failures, which are reported separately.

## Context-Faithfulness Evaluation

Evaluate whether generated answers are supported by their contexts:

```bash
python faithfulness_judge.py \
  --input-jsonl runs/evaluation/generations.jsonl \
  --output-jsonl runs/evaluation/faithfulness.jsonl \
  --judge-model Qwen/Qwen3-32B \
  --dataset msmarco \
  --max-new-tokens 128
```

Input records should contain a question, context, and an answer under `model_answer`, `prediction`, or `answer`.

The output preserves the input fields and adds the raw judge response and a Boolean `faithful` decision. The aggregate score is the fraction of scored records marked faithful.

The judge evaluates context support independently of reference-answer correctness.

Do not use `--only-correct` for unconditional faithfulness. That option expects a Boolean `correct` field, whereas `tide_evaluate.py` writes `judge_correct`; an explicit field conversion is needed before using this filter.

## KL-Divergence Evaluation

Compute full-vocabulary forward KL in the direction:

```text
KL(base model || trained model)
```

The input JSONL must contain the exact rendered generation `prompt` and the generated answer under `model_answer`, `prediction`, or `answer`.

```bash
python kl_divergence.py \
  --base-model Qwen/Qwen3-4B \
  --trained-model runs/tide_msmarco/model \
  --input-jsonl data/generated_answers_with_prompts.jsonl \
  --output-json runs/evaluation/kl.json \
  --base-device cuda:0 \
  --trained-device cuda:1
```

The script averages KL over retained answer tokens, including EOS when retained. The two models must have compatible tokenizers and vocabularies.

