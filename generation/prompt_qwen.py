"""
Within-backbone prompting baseline (rubric-grounded prompting on Qwen).
"""

import os
import sys
import json
import argparse
from pathlib import Path

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
from transformers import AutoModelForCausalLM, AutoTokenizer

from aes.model import ArTSScorer
from generation.evaluate import compute_qwk_results
from utils.data_utils import PROMPT_TRAITS


PROMPTING_ROOT = Path(__file__).resolve().parent.parent / "prompting"

# Trait ordering used in the prompting assets (spaces).
TRAITS_SPACE = [
    "overall", "content", "prompt adherence", "language", "narrativity",
    "organization", "word choice", "sentence fluency", "conventions",
    "style", "voice",
]

# Bidirectional space<->underscore mapping for trait names.
_SPACE_TO_US = {
    "prompt adherence": "prompt_adherence",
    "word choice": "word_choice",
    "sentence fluency": "sentence_fluency",
}


class QwenPromptBuilder:
    """Reproduces the Claude/GPT prompt formatting logic (prompting/src/llm.py)
    locally, so the same rubric assets can be applied to a Qwen backbone.

    Both CTP+FS and SRLP+FS use the `user_prompt_template_db.md` layout
    (bullets + rubric block + examples + task), which matches the paper's
    Appendix C prompt figures. The two modes only differ in what fills the
    {characteristics} block:
      - ctp_fs: contrastive high/low descriptors from descriptions.json
      - srlp_fs: per-trait, per-score rubric descriptors from
        trait_descriptions.json (overall trait is skipped, as in the
        Claude/GPT pipeline)
    """

    def __init__(self, mode: str, root: Path = PROMPTING_ROOT):
        assert mode in ("ctp_fs", "srlp_fs"), f"Unknown mode {mode}"
        self.mode = mode
        self.root = root

        with open(root / "essay_prompts.json") as f:
            self.prompts = {item["prompt"]: item for item in json.load(f)}
        with open(root / "ranges.json") as f:
            self.ranges = json.load(f)
        with open(root / "prompts" / "system_prompt.md") as f:
            self.system_prompt = f.read()
        with open(root / "prompts" / "user_prompt_template_db.md") as f:
            self.user_template = f.read()

        if mode == "ctp_fs":
            with open(root / "src" / "descriptions.json") as f:
                self.descriptions = json.load(f)
            self.rubrics = None
        else:
            with open(root / "trait_descriptions.json") as f:
                self.rubrics = json.load(f)
            self.descriptions = None

        self._examples_cache: dict[int, str] = {}

    def _examples(self, prompt_id: int) -> str:
        if prompt_id not in self._examples_cache:
            with open(self.root / "prompts" / f"prompt_{prompt_id}_examples.md") as f:
                self._examples_cache[prompt_id] = f.read()
        return self._examples_cache[prompt_id]

    @staticmethod
    def _score_for_trait(row, trait_space: str):
        """Fetch trait score from the row, trying both spaced and underscored col names."""
        for col in (trait_space, _SPACE_TO_US.get(trait_space, trait_space)):
            if col in row:
                val = row[col]
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return None
                if isinstance(val, (int, float)):
                    return int(val)
                try:
                    return int(float(val))
                except (TypeError, ValueError):
                    return None
        return None

    def build_messages(self, row) -> list[dict]:
        pid = int(row["prompt_id"])
        prompt_key = f"prompt_{pid}"

        scores_section = ""
        characteristics_section = ""
        for trait in TRAITS_SPACE:
            score_val = self._score_for_trait(row, trait)
            if score_val is None:
                continue
            rng = self.ranges.get(prompt_key, {}).get(trait)
            range_str = f"(Range: {rng['min']}-{rng['max']})" if rng else ""
            scores_section += f"- {trait}: {score_val} {range_str}\n"

            if self.mode == "ctp_fs":
                desc = self.descriptions.get(trait, "")
                if desc:
                    characteristics_section += (
                        f"**{trait.capitalize()}:** {desc}\n\n"
                    )
            else:
                if trait == "overall":
                    continue  # overall has no per-score rubric text
                trait_rubric = (self.rubrics.get(prompt_key) or {}).get(trait) or {}
                specific = trait_rubric.get(str(score_val))
                if specific:
                    characteristics_section += (
                        f"**{trait.capitalize()} {score_val} rubric:** {specific}\n\n"
                    )

        user_prompt = (
            self.user_template
            .replace("{scores}", scores_section.strip())
            .replace("{prompt_id}", str(pid))
            .replace("{characteristics}", characteristics_section.strip())
            .replace("{instruction}", self.prompts[pid]["wrt_instruction"])
            .replace("{examples}", self._examples(pid))
        )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]


def load_gen_model(base_model: str, bf16: bool):
    tok = AutoTokenizer.from_pretrained(
        base_model, cache_dir=HUB, use_fast=True, trust_remote_code=True,
    )
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if bf16 else None
    model = AutoModelForCausalLM.from_pretrained(
        base_model, cache_dir=HUB, device_map="auto",
        torch_dtype=dtype, trust_remote_code=True,
    ).eval()
    return model, tok


@torch.inference_mode()
def generate_batch(model, tok, chat_batches, max_new_tokens, max_seq_len):
    prompts = [
        tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        for msgs in chat_batches
    ]
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
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    input_len = enc["input_ids"].shape[1]
    return [
        tok.decode(out[i][input_len:], skip_special_tokens=True).strip()
        for i in range(len(prompts))
    ]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["ctp_fs", "srlp_fs"], required=True)
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data_csv", required=True,
                   help="Test CSV. Must have `prompt_id`, `content_text`, and gold trait cols.")
    p.add_argument("--aes_model", default="Heejindo/scorer_f0",
                   help="Must be the verifier trained on the SAME fold as --data_csv, "
                        "e.g. Heejindo/scorer_f1 for fold_1's test split.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tag", default=None,
                   help="Suffix for output file names; defaults to <base_model_slug>_<mode>.")

    p.add_argument("--gen_batch",     type=int, default=4)
    p.add_argument("--aes_batch",     type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_seq_len",    type=int, default=12288)
    p.add_argument("--max_src_len",    type=int, default=1024)
    p.add_argument("--bf16", action="store_true")
    return p.parse_args()


def _slug(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1].replace(".", "").replace("-", "").lower()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_csv)
    print(f"Loaded {len(df)} rows from {args.data_csv}")

    # PROMPT_TRAITS uses underscore trait names, so gold cols must be underscored
    # for compute_qwk_results to find them. Rename in-place.
    df = df.rename(columns=_SPACE_TO_US)

    builder = QwenPromptBuilder(mode=args.mode)
    print(f"Building {args.mode} prompts...")
    all_msgs = [builder.build_messages(row) for _, row in df.iterrows()]

    print(f"Loading generation model: {args.base_model}")
    model, tok = load_gen_model(args.base_model, bf16=args.bf16)

    essays = []
    for start in tqdm(range(0, len(all_msgs), args.gen_batch), desc="Generating"):
        batch = all_msgs[start : start + args.gen_batch]
        essays.extend(generate_batch(
            model, tok, batch, args.max_new_tokens, args.max_seq_len,
        ))
    df["generated_essay"] = essays

    del model
    torch.cuda.empty_cache()

    print(f"Loading AES model: {args.aes_model}")
    scorer = ArTSScorer(args.aes_model, max_src_len=args.max_src_len)

    pids = df["prompt_id"].astype(int).tolist()
    all_preds = []
    for start in tqdm(range(0, len(essays), args.aes_batch), desc="AES scoring"):
        b_essays = essays[start : start + args.aes_batch]
        b_pids = pids[start : start + args.aes_batch]
        all_preds.extend(scorer.score(b_essays, b_pids))

    for trait in sorted({t for p in all_preds for t in p}):
        df[f"pred_{trait}"] = [p.get(trait, float("nan")) for p in all_preds]

    tag = args.tag or f"{_slug(args.base_model)}_{args.mode}"
    stem = os.path.splitext(os.path.basename(args.data_csv))[0]
    out_csv = os.path.join(args.output_dir, f"eval_{stem}__{tag}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions → {out_csv}")

    qwk_df = compute_qwk_results(df, pred_col_prefix="pred_")

    print("\n" + "=" * 60)
    print(f"QWK — backbone={args.base_model}  mode={args.mode}")
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

    qwk_path = out_csv.replace(".csv", "_qwk.csv")
    qwk_df.to_csv(qwk_path, index=False)
    print(f"Saved QWK summary → {qwk_path}")


if __name__ == "__main__":
    main()
