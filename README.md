# SWIM

Official repository for the EMNLP paper, "SWIM: Student Writing Simulation via Proficiency-Conditioned Generation"

Given a writing prompt and a target trait-level proficiency profile, SWIM generates a student essay that reflects that profile, and measures profile alignment with a frozen multi-trait automated essay scoring (AES) model (ArTS). This repository contains the baseline (rubric-grounded prompting), SFT, and GRPO pipelines evaluated in the paper.

## Project structure

```
SWIM/
├── aes/
│   └── model.py                # ArTSScorer: score essays, compute the PAR reward
├── generation/
│   ├── sft_train.py             # Step 1: SFT on real ASAP essays
│   ├── evaluate.py              # Generate + AES-score + QWK for an SFT/GRPO checkpoint
│   ├── rescore.py               # Re-score existing outputs with an independent AES verifier
│   └── prompt_qwen.py           # Rubric-grounded prompting baseline on a Qwen backbone
├── rl/
│   └── grpo_train.py            # Step 2: GRPO with the Proficiency Alignment Reward (PAR)
├── prompting/                    # Claude/GPT prompting baseline pipeline (see prompting/README.md)
├── utils/
│   └── data_utils.py            # Prompt builder, collators, dataset loaders
├── dataset/
│   ├── asap/fold_{0..4}/         # Raw ASAP/ASAP++ splits (5-fold CV)
│   └── gen/
│       ├── prompt.jsonl          # prompt_id -> writing instruction text
│       ├── data_preprocess_for_generation.py
│       └── not_norm/fold_{0..4}/ # Preprocessed splits consumed by sft_train.py / evaluate.py / grpo_train.py
└── scripts/                      # bash/sbatch entry points for each step
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The `prompting/` pipeline (Claude/GPT baseline) pins an older, separate dependency set — install it in its own environment if you need to reproduce it (`prompting/requirements.txt`, see `prompting/README.md`).

Set `HF_TOKEN` in your environment if any of the models you point these scripts at (generator backbone, AES verifier) are private on the Hugging Face Hub.

## Pipeline

Each step also has a corresponding script under `scripts/` that works with either `bash scripts/run_*.sh` or `sbatch scripts/run_*.sh` — adjust the `#SBATCH` directives and the environment-activation block at the top of each script for your cluster.

### 1. Rubric-grounded prompting (baseline)

```bash
python generation/prompt_qwen.py \
    --mode ctp_fs \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --data_csv dataset/asap/fold_0/total_test.csv \
    --aes_model Heejindo/scorer_f0 \
    --output_dir results/prompt_qwen/fold_0 \
    --bf16
```

`--mode` is `ctp_fs` (Contrastive Trait Prompting, 5-shot) or `srlp_fs` (Score-level Rubric-Lookup Prompting, 5-shot). See `prompting/README.md` for the Claude/GPT variant of this baseline.

### 2. SFT

Fine-tunes a Qwen backbone (LoRA/QLoRA) on real ASAP essays to generate essays conditioned on trait scores.

```bash
python generation/sft_train.py \
    --data_path dataset/gen/not_norm/fold_0 \
    --output_path results/sft_qwen25-7b-instruct_f0 \
    --model Qwen/Qwen2.5-7B-Instruct \
    --mode multitrait \
    --epoch 5 --lr 1e-4 --train_batch 4 --grad_accum 2 \
    --use_4bit --bf16
```

**Input**: `total_train.csv` (real essays, `content_text` column) — **Output**: LoRA checkpoint at `--output_path`.

### 3. GRPO (online RL)

Generates `G` essays per prompt from the current policy, scores them with the frozen AES verifier, and updates the policy with the group-normalized Proficiency Alignment Reward (PAR). Initializes from an SFT checkpoint.

```bash
python rl/grpo_train.py \
    --sft_checkpoint results/sft_qwen25-7b-instruct_f0/checkpoint-XXXX \
    --aes_model Heejindo/scorer_f0 \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --data_path dataset/gen/not_norm/fold_0 \
    --output_path results/grpo_qwen25-7b-instruct_f0 \
    --num_generations 4 --lr 1e-5 \
    --bf16 --use_vllm
```

Requires TRL's `GRPOTrainer` (`trl>=1.1.0`). Reward: `r = mean_t(1 - |pred_t - target_t| / range_t) ∈ [0, 1]`.

### 4. Evaluation

Generates essays for the test split from a checkpoint (SFT or GRPO), scores them with the AES verifier, and reports per-trait and per-prompt QWK against the gold profile used for conditioning.

```bash
python generation/evaluate.py \
    --checkpoint results/sft_qwen25-7b-instruct_f0/checkpoint-XXXX \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --aes_model Heejindo/scorer_f0 \
    --data_path dataset/gen/not_norm/fold_0 \
    --data_file total_test.csv \
    --bf16
```

### 5. Cross-evaluator check (optional)

Re-scores an existing `eval_*.csv` with a second, independently-trained AES verifier (T5 or DeBERTa), to check that alignment gains generalize beyond the reward verifier used during GRPO training (§5.4 of the paper).

```bash
python generation/rescore.py \
    --input_csv results/grpo_.../checkpoint-XXXX/eval_total_test.csv \
    --aes_model Heejindo/scorer_deberta_f0
```

## AES module usage

`--aes_model` throughout this repo defaults to [`Heejindo/scorer_f0`](https://huggingface.co/Heejindo/scorer_f0), the ArTS checkpoint trained on fold 0. Each fold has its own verifier (`Heejindo/scorer_f0` … `scorer_f4`). **When working with a different fold, pass the matching `--aes_model scorer_f<N>`, since a mismatched fold's verifier has seen that fold's training essays and leaks into the score.**

```python
from aes.model import ArTSScorer

scorer = ArTSScorer("Heejindo/scorer_f0")

preds = scorer.score(["essay text..."], prompt_ids=[3])
# -> [{"overall": 2.0, "content": 2.0, "prompt_adherence": 1.0, ...}]

r = scorer.reward(preds[0], target={"overall": 2, "content": 2}, prompt_id=3)
# -> float in [0, 1]; 1.0 = perfect match
```

## ASAP/ASAP++ dataset — trait / score reference

| Prompt | Traits | Overall range |
|---|---|---|
| 1 | overall, content, organization, word_choice, sentence_fluency, conventions | 2–12 |
| 2 | overall, content, organization, word_choice, sentence_fluency, conventions | 1–6 |
| 3–4 | overall, content, prompt_adherence, language, narrativity | 0–3 |
| 5–6 | overall, content, prompt_adherence, language, narrativity | 0–4 |
| 7 | overall, content, organization, conventions | 0–30 |
| 8 | overall, content, organization, word_choice, sentence_fluency, conventions | 0–60 |

ASAP essays are sourced from the [Kaggle ASAP-AES competition](https://www.kaggle.com/c/asap-aes); trait-level annotations are from ASAP++ (Mathias and Bhattacharyya, 2018). The frozen AES verifier is [ArTS](https://aclanthology.org/2024.findings-eacl.115/) (Do et al., 2024).
