# Rubric-grounded prompting baseline

Code and assets for the CTP (Contrastive Trait Prompting) and SRLP
(Score-level Rubric-Lookup Prompting) baselines reported in Table 2 of the
paper. This pipeline is what was used to generate the Claude Sonnet results;
the same assets are reused by `generation/prompt_qwen.py` (in the repo root)
to run the identical prompts against open-source Qwen backbones.

Uses its own, older-pinned dependency set (`requirements.txt` in this
directory) — install it in a separate virtual environment from the main
repo's `requirements.txt`.

## Layout

```
prompting/
├── essay_prompts.json        # writing prompt text + grade level per ASAP prompt id
├── ranges.json                # per-prompt, per-trait (min, max) score ranges
├── trait_descriptions.json    # SRLP: per-trait, per-score rubric text
├── prompts/
│   ├── system_prompt.md
│   ├── user_prompt_template_db.md      # SRLP template
│   ├── user_prompt_template_hyp_mix.md # CTP template
│   └── prompt_{1..8}_examples.md       # 5-shot exemplars per ASAP prompt
└── src/
    ├── data_manager.py    # loads a fold's CSV + prompts/ranges into per-row context
    ├── descriptions.json  # CTP: contrastive high/low descriptor per trait
    ├── llm.py             # ClaudeGenerator — builds prompts, calls the Anthropic Batch API
    ├── generate_essays.py # submits a batch generation job
    ├── download_essays.py # downloads a finished batch job to downloaded_essays.csv
    ├── scorer.py           # T5Scorer — scores downloaded essays with the ArTS AES verifier
    ├── eval_qwk.py         # QWK Evaluator used by scorer.py
    └── evaluate_essays.py  # scores + computes QWK for a downloaded_essays.csv
```

## Running the Claude pipeline

Requires `ANTHROPIC_API_KEY` in the environment (or a `.env` file in
`prompting/src/`).

```bash
cd prompting/src

# 1. Submit a batch generation job (writes current_batch_id.txt)
python generate_essays.py

# 2. Once the batch has finished (check the Anthropic dashboard, or poll
#    generator.check_batch_status), download the results
python download_essays.py

# 3. Score the downloaded essays with the ArTS AES verifier and report QWK
python evaluate_essays.py
```

`generate_essays.py` / `download_essays.py` hardcode the test split to
`dataset/asap/fold_0/total_test.csv` (matching the paper: prompting-based
methods are evaluated on a single fold due to generation cost). To run
another fold, edit `dev_path` in both scripts.

Both scripts also hardcode `mode=GenerationMode.LOOKUP` (SRLP) when
constructing `ClaudeGenerator`. To reproduce the CTP numbers instead, edit
that argument to `mode=GenerationMode.TRAIT_BASED` in both
`generate_essays.py` and `download_essays.py` before running; this is a
source edit, not a CLI/env option.

## Running the same prompts on a Qwen backbone

See `../scripts/run_prompt_qwen.sh` and `../generation/prompt_qwen.py` — no
Anthropic API key needed, just a local (or HF Hub) causal LM.
