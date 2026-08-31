"""
Evaluation script for the essay generation model.
"""

import os
import sys
import json
import argparse

HUB = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["HF_HOME"] = HUB
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HUB, "transformers"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(HUB, "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HUB, "hub"))

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import cohen_kappa_score
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from aes.model import ArTSScorer
from utils.data_utils import PROMPT_TRAITS, build_messages, load_prompt_map


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   type=str, required=True,
                   help="LoRA checkpoint path (SFT or RL output).")
    p.add_argument("--base_model",   type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--aes_model",    type=str,
                   default="Heejindo/scorer_f0",
                   help="Must be the verifier trained on the SAME fold as --data_path/--checkpoint, "
                        "e.g. Heejindo/scorer_f1 for fold_1. Each fold has its own verifier; using a "
                        "mismatched fold leaks that fold's training essays into the score.")
    p.add_argument("--data_path",    type=str, default="dataset/gen/not_norm/fold_0")
    p.add_argument("--data_file",    type=str, default="total_test.csv")
    p.add_argument("--prompt_jsonl", type=str, default="dataset/gen/prompt.jsonl")
    p.add_argument("--output_csv",   type=str, default=None,
                   help="If not set, saved next to --checkpoint as eval_<data_file>.csv")
    p.add_argument("--mode",         type=str, default="multitrait",
                   choices=["multitrait", "holistic"])

    # Generation
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_seq_len",    type=int, default=2500)
    p.add_argument("--gen_batch",      type=int, default=4)

    # AES
    p.add_argument("--aes_batch",    type=int, default=8)
    p.add_argument("--max_src_len",  type=int, default=512)
    p.add_argument("--max_new_tokens_aes", type=int, default=64)

    # Precision
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--bf16",     action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_gen_model(args):
    tok = AutoTokenizer.from_pretrained(
        args.base_model, cache_dir=HUB, use_fast=True, trust_remote_code=True,
    )
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, cache_dir=HUB, quantization_config=bnb,
            device_map="auto", trust_remote_code=True,
        )
    else:
        dtype = torch.bfloat16 if args.bf16 else None
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, cache_dir=HUB, device_map="auto",
            torch_dtype=dtype, trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(base, args.checkpoint, is_trainable=False)
    model.eval()
    return model, tok


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_prompt(tok, row, mode, prompt_map):
    pid = int(str(row.get("prompt_id", 1)).strip())
    prompt_text = prompt_map.get(pid, "")
    messages = build_messages(
        prompt_id=pid,
        prompt_text=prompt_text,
        overall_score=row.get("overall_score"),
        trait_lines=row.get("trait_lines", ""),
        mode=mode,
    )
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.inference_mode()
def generate_batch(model, tok, prompts, max_new_tokens, max_seq_len=2500):
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=False,
    ).to(model.device)

    out = model.generate(
        **enc,
        do_sample=False,          # greedy for reproducible eval
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        # NOTE: repetition_penalty was tried at 1.15 to fix Qwen3-4B mode collapse
        # on long prompts, but it caused AES scores to drift out of distribution
        # (the T5 verifier was trained on natural essays where mild repetition is
        # expected). Left at the default (1.0) to stay consistent with GRPO
        # training, which also does not use it.
    )

    input_len = enc["input_ids"].shape[1]
    essays = []
    for i in range(len(prompts)):
        generated_ids = out[i][input_len:]
        essays.append(tok.decode(generated_ids, skip_special_tokens=True).strip())
    return essays


# ---------------------------------------------------------------------------
# QWK computation (numeric predictions vs numeric gold)
# ---------------------------------------------------------------------------

def compute_qwk_results(df, pred_col_prefix="pred_"):
    """
    Compute per-trait QWK between AES-predicted scores and gold scores.
    Predictions are in columns named f"{pred_col_prefix}{trait}".
    Gold scores are in columns named by trait name directly.
    """
    rows = []
    for pid in sorted(df["prompt_id"].unique()):
        sub = df[df["prompt_id"] == pid].copy()
        traits = PROMPT_TRAITS.get(int(pid), [])
        trait_qwk = {"prompt_id": int(pid), "n": len(sub)}

        for trait in traits:
            pred_col = f"{pred_col_prefix}{trait}"
            gold_col = trait
            if pred_col not in sub.columns or gold_col not in sub.columns:
                continue

            gold = sub[gold_col].dropna()
            pred = sub[pred_col].loc[gold.index].fillna(-1)

            # Round to int for QWK
            gold_int = gold.round().astype(int).tolist()
            pred_int = pred.round().astype(int).tolist()

            if len(set(gold_int)) < 2:
                trait_qwk[f"qwk_{trait}"] = float("nan")
                continue

            try:
                qwk = cohen_kappa_score(gold_int, pred_int, weights="quadratic")
            except Exception:
                qwk = float("nan")
            trait_qwk[f"qwk_{trait}"] = round(qwk, 4)

        # Mean QWK across traits (excluding overall)
        sub_traits = [t for t in traits if t != "overall"]
        vals = [trait_qwk[f"qwk_{t}"] for t in sub_traits
                if f"qwk_{t}" in trait_qwk and not np.isnan(trait_qwk[f"qwk_{t}"])]
        trait_qwk["mean_qwk"] = round(float(np.mean(vals)), 4) if vals else float("nan")
        rows.append(trait_qwk)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.output_csv is None:
        stem = os.path.splitext(args.data_file)[0]
        args.output_csv = os.path.join(args.checkpoint, f"eval_{stem}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    # Load data
    prompt_map = load_prompt_map(args.prompt_jsonl)
    df = pd.read_csv(os.path.join(args.data_path, args.data_file))
    print(f"Test rows: {len(df)}")

    # Load generation model
    print("Loading generation model...")
    gen_model, tok = load_gen_model(args)

    # Build prompts and generate
    prompts = [build_prompt(tok, row, args.mode, prompt_map) for _, row in df.iterrows()]

    all_essays = []
    for start in tqdm(range(0, len(prompts), args.gen_batch), desc="Generating"):
        batch = prompts[start : start + args.gen_batch]
        all_essays.extend(generate_batch(gen_model, tok, batch, args.max_new_tokens, args.max_seq_len))

    df["generated_essay"] = all_essays

    # Free generation model memory before loading AES
    del gen_model
    torch.cuda.empty_cache()

    # Score with ArTS
    print("Loading AES model and scoring...")
    scorer = ArTSScorer(
        args.aes_model,
        max_src_len=args.max_src_len,
        max_new_tokens=args.max_new_tokens_aes,
    )

    pids   = df["prompt_id"].tolist()
    essays = df["generated_essay"].tolist()

    all_preds = []
    for start in tqdm(range(0, len(essays), args.aes_batch), desc="AES scoring"):
        batch_essays = essays[start : start + args.aes_batch]
        batch_pids   = pids[start : start + args.aes_batch]
        preds = scorer.score(batch_essays, batch_pids)
        all_preds.extend(preds)

    # Store AES predictions as columns
    for trait in set(t for p in all_preds for t in p):
        df[f"pred_{trait}"] = [p.get(trait, float("nan")) for p in all_preds]

    # Save full results
    df.to_csv(args.output_csv, index=False)
    print(f"Saved results → {args.output_csv}")

    # Compute and print QWK
    qwk_df = compute_qwk_results(df, pred_col_prefix="pred_")

    print("\n" + "=" * 60)
    print("QWK (AES-predicted vs gold trait scores)")
    print("=" * 60)
    for _, row in qwk_df.iterrows():
        pid = int(row["prompt_id"])
        traits = PROMPT_TRAITS.get(pid, [])
        print(f"\nPrompt {pid}  (n={int(row['n'])})")
        for trait in traits:
            col = f"qwk_{trait}"
            if col in row:
                val = row[col]
                print(f"  {trait:<20}: {val:.4f}" if not np.isnan(val) else f"  {trait:<20}: N/A")
        print(f"  {'mean (excl. overall)':<20}: {row['mean_qwk']:.4f}")

    macro_mean = qwk_df["mean_qwk"].mean()
    print(f"\nMacro-avg QWK across prompts: {macro_mean:.4f}")
    print("=" * 60)

    qwk_path = args.output_csv.replace(".csv", "_qwk.csv")
    qwk_df.to_csv(qwk_path, index=False)
    print(f"Saved QWK summary → {qwk_path}")


if __name__ == "__main__":
    main()
