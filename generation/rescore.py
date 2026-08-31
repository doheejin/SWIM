"""
Re-score existing generation results with a different AES scorer.
"""

import os
import sys
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
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from aes.model import ArTSScorer, SCORE_RANGES
from generation.evaluate import compute_qwk_results
from utils.data_utils import PROMPT_TRAITS


# Trait order the DeBERTa regression head was trained with.
# Keeps spaces here for readability, but we map to underscore names when
# writing pred_* columns.
DEBERTA_TRAITS = [
    "overall", "content", "organization", "word choice",
    "sentence fluency", "conventions", "prompt adherence",
    "language", "narrativity", "style", "voice",
]

_TRAIT_SPACE_TO_UNDERSCORE = {
    "word choice": "word_choice",
    "sentence fluency": "sentence_fluency",
    "prompt adherence": "prompt_adherence",
}


def _canonical(trait: str) -> str:
    return _TRAIT_SPACE_TO_UNDERSCORE.get(trait, trait)


class DebertaScorer:
    """DeBERTa-v2 sequence-classification head with 11 regression outputs.

    Outputs are normalized (0..1) at training time; we denormalize per-sample
    using the prompt-specific (min, max) ranges from SCORE_RANGES.
    """

    def __init__(self, model_path: str, device: str = "cuda",
                 max_src_len: int = 512, batch_size: int = 8):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_path)
            .to(self.device)
            .eval()
        )
        self.max_src_len = max_src_len
        self.batch_size = batch_size

    @torch.inference_mode()
    def score(self, essays: list[str], prompt_ids: list[int]) -> list[dict]:
        inputs = [
            f"score the essay of the prompt {pid}: {essay}"
            for pid, essay in zip(prompt_ids, essays)
        ]
        enc = self.tokenizer(
            inputs,
            max_length=self.max_src_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits.detach().cpu().numpy()  # (B, 11)

        out = []
        for i, pid in enumerate(prompt_ids):
            ranges = SCORE_RANGES.get(int(pid), {})
            preds = {}
            for t_idx, trait_space in enumerate(DEBERTA_TRAITS):
                trait = _canonical(trait_space)
                r = ranges.get(trait)
                if r is None:
                    continue
                lo, hi = r
                preds[trait] = float(logits[i, t_idx]) * (hi - lo) + lo
            out.append(preds)
        return out


def _detect_scorer_kind(model_path: str) -> str:
    cfg = AutoConfig.from_pretrained(model_path)
    archs = getattr(cfg, "architectures", None) or []
    for arch in archs:
        if "T5" in arch:
            return "t5"
        if "Deberta" in arch or "Bert" in arch:
            return "deberta"
    model_type = getattr(cfg, "model_type", "").lower()
    if model_type == "t5":
        return "t5"
    if "deberta" in model_type or "bert" in model_type:
        return "deberta"
    raise ValueError(f"Cannot detect scorer type for {model_path} (archs={archs}, model_type={model_type})")


def _slugify(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", type=str, required=True,
                   help="Existing eval CSV with `generated_essay` + `prompt_id` columns.")
    p.add_argument("--aes_model", type=str, required=True,
                   help="HF model id or local path of the scorer to re-run with.")
    p.add_argument("--output_csv", type=str, default=None,
                   help="Defaults to <input_stem>__<scorer_slug>.csv in the same dir.")
    p.add_argument("--aes_batch", type=int, default=16)
    p.add_argument("--max_src_len", type=int, default=512)
    p.add_argument("--max_new_tokens_aes", type=int, default=64,
                   help="T5 only; ignored for DeBERTa.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_csv is None:
        stem, ext = os.path.splitext(args.input_csv)
        args.output_csv = f"{stem}__{_slugify(args.aes_model)}{ext}"
    if os.path.abspath(args.output_csv) == os.path.abspath(args.input_csv):
        raise ValueError("Refusing to overwrite the input CSV; pass a distinct --output_csv.")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if "generated_essay" not in df.columns or "prompt_id" not in df.columns:
        raise ValueError(f"{args.input_csv} must contain `generated_essay` and `prompt_id` columns.")
    print(f"Loaded {len(df)} rows from {args.input_csv}")

    # Strip stale pred_* columns so the new scorer's output isn't mixed with old.
    stale_pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if stale_pred_cols:
        df = df.drop(columns=stale_pred_cols)
        print(f"Dropped stale prediction columns: {stale_pred_cols}")

    kind = _detect_scorer_kind(args.aes_model)
    print(f"Scorer kind: {kind}  ({args.aes_model})")
    if kind == "t5":
        scorer = ArTSScorer(
            args.aes_model,
            max_src_len=args.max_src_len,
            max_new_tokens=args.max_new_tokens_aes,
        )
    else:
        scorer = DebertaScorer(
            args.aes_model,
            max_src_len=args.max_src_len,
            batch_size=args.aes_batch,
        )

    essays = df["generated_essay"].fillna("").astype(str).tolist()
    pids = df["prompt_id"].astype(int).tolist()

    all_preds = []
    for start in tqdm(range(0, len(essays), args.aes_batch), desc="AES scoring"):
        batch_essays = essays[start : start + args.aes_batch]
        batch_pids = pids[start : start + args.aes_batch]
        all_preds.extend(scorer.score(batch_essays, batch_pids))

    for trait in sorted({t for p in all_preds for t in p}):
        df[f"pred_{trait}"] = [p.get(trait, float("nan")) for p in all_preds]

    df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions → {args.output_csv}")

    qwk_df = compute_qwk_results(df, pred_col_prefix="pred_")

    print("\n" + "=" * 60)
    print(f"QWK — scorer={args.aes_model}")
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
