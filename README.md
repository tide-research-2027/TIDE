# TIDE

**Target Optimization with Implicit Distillation for ContExt-Grounded RAG**

Official implementation of **Grounded without Drifting from Pretraining:
Rollout-Free Target Optimization with Implicit Distillation for RAG**.

TIDE performs full-parameter post-training with response-level weights derived
from pretrained-policy compatibility and context faithfulness:

```text
V(X,y) = mean_answer_token_log_probability_base(y|X) + entailment(context,y)
w(X,y) = exp(alpha * V(X,y))
L = mean_i w_i * mean_answer_tokens[-log pi_theta(y_i|X_i)]
```

The default experiment uses `alpha=0.5`, Qwen3-14B as the policy, and
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` during target
weight construction. Evaluation uses Qwen3-32B as a fixed judge. The smaller
experiment uses Qwen3-4B and Qwen3-8B, respectively.

## Files

| File | Purpose |
|---|---|
| `tide_train.py` | TIDE/FIND, SFT, Less-Value, and Faith-Only full-model training |
| `self_demo.py` | Prompt optimization, Self-Demo generation, tournament selection, and training |
| `prepare_dataset.py` | Dataset-independent conversion to canonical QA JSONL |
| `prepare_benchmarks.py` | Exact five-benchmark evaluation constructors and paper sizes |
| `tide_evaluate.py` | Generation, fixed-judge evaluation, and metrics |
| `faithfulness_judge.py` | Context faithfulness with a fixed Qwen judge |
| `kl_divergence.py` | Token-weighted full-vocabulary `KL(base || trained)` |
| `deepspeed_zero3.json` | Full-parameter ZeRO-3 configuration |
| `deepspeed_zero3_offload.json` | Full-parameter ZeRO-3 CPU-offload configuration |

## Environment

Training and Transformers evaluation:

```bash
conda create -n tide python=3.11 -y
conda activate tide
python -m pip install -r requirements.txt
```

Self-Demo generation uses a separate vLLM environment:

```bash
conda create -n tide-vllm python=3.11 -y
conda activate tide-vllm
python -m pip install -r requirements-vllm.txt
```

## Canonical Data Schema

Training and evaluation artifacts use JSONL with one object per line:

```json
{"id":"example-1","question":"What is the capital of France?","context":"Paris is the capital of France.","answers":["Paris"],"answer":"Paris","metadata":{}}
```

Convert JSONL, JSON, CSV/TSV, or Hugging Face data:

```bash
python prepare_dataset.py \
  --input-file raw/evaluation.jsonl \
  --output-dir data/prepared \
  --output-prefix evaluation \
  --id-field id \
  --question-field question \
  --context-field context \
  --answers-field answers \
  --require-answer --require-context --deduplicate
```

Nested passage collections and selected-passage flags are supported:

```bash
python prepare_dataset.py \
  --input-file raw/data.jsonl \
  --output-dir data/prepared \
  --output-prefix selected \
  --question-field query \
  --context-field passages \
  --answers-field answers \
  --passage-text-field passage_text \
  --passage-selected-field is_selected \
  --selected-value 1 \
  --require-answer-in-context
```

Preparation writes a JSON summary containing accepted/skipped counts, output
paths, seed, and schema. Preserve this summary with every experiment.

Prepare the reported evaluation sets:

```bash
for dataset in msmarco hotpotqa nq wow trex; do
  python prepare_benchmarks.py --dataset "$dataset" \
    --output-jsonl "data/evaluation/${dataset}.jsonl" --seed 42
done
```

This produces MSMARCO `6,789`, HotpotQA `7,000`, NQ `3,610`, WoW `7,866`,
and T-REx `7,000` examples using the constructions described in the paper.

## Full-Parameter Training

All methods use the same base model, optimizer settings, learning rate,
effective batch construction, epoch count, seed, and data loader. No LoRA,
QLoRA, or PEFT adapter is used.

### TIDE

```bash
torchrun --standalone --nproc_per_node=8 tide_train.py \
  --dataset msmarco --mode find \
  --base-model Qwen/Qwen3-14B \
  --nli-model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli \
  --alpha 0.5 --max-samples 75000 --seed 42 \
  --weighted-jsonl runs/msmarco/tide/weighted.jsonl \
  --output-dir runs/msmarco/tide/model \
  --deepspeed deepspeed_zero3.json
```

For NewsQA, run a separate job with the reported retained count:

```bash
torchrun --standalone --nproc_per_node=8 tide_train.py \
  --dataset newsqa --mode find \
  --base-model Qwen/Qwen3-14B \
  --nli-model MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli \
  --alpha 0.5 --max-samples 71719 --seed 42 \
  --weighted-jsonl runs/newsqa/tide/weighted.jsonl \
  --output-dir runs/newsqa/tide/model \
  --deepspeed deepspeed_zero3.json
```

MSMARCO and NewsQA are always trained as separate jobs; the training loader
does not concatenate or mix datasets.

### Baselines and ablations

Use the identical command and change only `--mode`:

```bash
# Uniform-weight supervised fine-tuning
--mode sft

# Pretrained log-probability component only
--mode less_value --alpha 0.5

# Faithfulness component only
--mode faith_only --alpha 0.5
```

Alpha sensitivity uses `--mode find` and one of:

```text
--alpha 0
--alpha 0.25
--alpha 0.375
--alpha 0.5
--alpha 0.625
--alpha 0.75
--alpha 1
--alpha 2
```

Use a fresh `--weighted-jsonl` and `--output-dir` for every method, dataset,
model scale, alpha, and seed. `--reuse-weighted-jsonl` is intended only for an
exactly matching configuration.

### Qwen3-4B scale

The 4B experiment uses four GPUs and gradient accumulation eight:

```bash
torchrun --standalone --nproc_per_node=4 tide_train.py \
  --dataset msmarco --mode find \
  --base-model Qwen/Qwen3-4B --alpha 0.5 \
  --max-samples 75000 --per-device-batch-size 1 \
  --gradient-accumulation-steps 8 --seed 42 \
  --weighted-jsonl runs/4b/msmarco/tide/weighted.jsonl \
  --output-dir runs/4b/msmarco/tide/model \
  --deepspeed deepspeed_zero3.json
```

## Self-Demo

Self-Demo follows prompt optimization, model self-scoring, critique/rewrite,
No-RAG/RAG/refusal candidate generation, and tournament selection. Candidate-only
strict selection uses minimum reference F1 `0.70`, reference precision `0.50`,
support recall `0.75`, `ndocs=4`, and answer caps of 14 words for MSMARCO and
16 words for NewsQA. Gold-reference fallback is never used as a training target.

Prepare MSMARCO:

```bash
python self_demo.py --dataset msmarco --stage prepare \
  --base-model Qwen/Qwen3-14B --seed 42
```

Generate and select demonstrations with all eight GPUs:

```bash
conda run -n tide-vllm python self_demo.py \
  --dataset msmarco --stage vllm \
  --base-model Qwen/Qwen3-14B \
  --tensor-parallel-size 8 \
  --cuda-visible-devices 0,1,2,3,4,5,6,7 \
  --seed 42
```

Train the selected demonstrations:

```bash
torchrun --standalone --nproc_per_node=8 self_demo.py \
  --dataset msmarco --stage train \
  --base-model Qwen/Qwen3-14B \
  --deepspeed-config deepspeed_zero3.json \
  --cuda-visible-devices 0,1,2,3,4,5,6,7 \
  --seed 42
```

Use `--dataset newsqa` for NewsQA. Each dataset receives an isolated data,
cache, and model directory automatically.

## Generation and Evaluation

Run generation and fixed-judge validation:

```bash
python tide_evaluate.py \
  --stage all \
  --input-jsonl data/prepared/evaluation.jsonl \
  --output-dir runs/evaluation/msmarco_tide \
  --model runs/msmarco/tide/model \
  --model-label TIDE \
  --judge-model Qwen/Qwen3-32B \
  --dataset-profile msmarco \
  --seed 42 --temperature 0
```

For Qwen3-4B policies, use `--judge-model Qwen/Qwen3-8B`.

The evaluation profile automatically sets generation/judge limits to
`32/192` for MSMARCO, `24/192` for HotpotQA, `32/192` for NQ, `64/192`
for WoW, and `16/160` for T-REx. Both generation and judging are greedy.

The evaluator writes:

```text
generations.jsonl   exact prompt, raw output, normalized answer, reference metrics
validated.jsonl     raw judge JSON, parsed decisions, and final correctness
summary.json         aggregate token and answer-level metrics
```

The fixed Qwen judge evaluates semantic correctness and context support. Final
correctness also applies fixed output-validity checks implemented in the
evaluation script. Raw decisions and associated reasons are retained in
`validated.jsonl`.

### Answer-level F1

Let `C` be final-correct answers, `A` attempted answers, and `G` answerable
examples:

```text
precision = C / A
recall    = C / G
F1        = 2 * precision * recall / (precision + recall)
```

`summary.json` reports `answer_precision`, `answer_recall`, and `answer_f1`.
It also reports normalized token-overlap precision, recall, and F1 as separate
diagnostic metrics.

## Faithfulness

Faithfulness is the fraction of generated answers whose factual claims are all
entailed by their supplied contexts. Reference correctness is not used by this
metric.

```bash
python faithfulness_judge.py \
  --input-jsonl runs/evaluation/msmarco_tide/generations.jsonl \
  --output-jsonl runs/evaluation/msmarco_tide/faithfulness.jsonl \
  --judge-model Qwen/Qwen3-32B \
  --dataset msmarco
```

## Distributional Shift

The KL script computes the full-vocabulary forward divergence at every retained
generated-answer position, conditioned on the same prompt and response prefix:

```text
D_i,t = KL(pi_base(.|X_i,y_i,<t) || pi_trained(.|X_i,y_i,<t))
KL_hat = sum_i sum_t D_i,t / sum_i T_i
```

```bash
python kl_divergence.py \
  --base-model Qwen/Qwen3-14B \
  --trained-model runs/msmarco/tide/model \
  --input-jsonl runs/evaluation/msmarco_tide/generations.jsonl \
  --output-json runs/evaluation/msmarco_tide/kl_base_to_model.json \
  --base-device cuda:0 --trained-device cuda:1
```

## Reproduction Record

For every reported run, retain:

- Prepared JSONL and preparation summary
- Weighted training JSONL
- Training arguments and logs
- Full model checkpoint
- `generations.jsonl`
- `validated.jsonl`
- `summary.json`
- Faithfulness JSONL
- KL JSON
- Git commit hash, random seed, model revisions, and environment lock

The exact prompts used by generation, correctness judging, and faithfulness
judging are constants in the corresponding scripts and are stored with raw
outputs for auditability.
