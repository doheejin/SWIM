# preprocess_multitrait_generation.py
# - cleans CSV
# - drops style/voice
# - loads prompt.jsonl and fills prompt_text (+ prompt_type)
# - builds trait_lines with raw scores + per-prompt min/max range (EXCLUDES overall)

import re
import json
import argparse
import pandas as pd
import numpy as np


PROMPT_MINMAX = {
    1: {'overall': (2, 12), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
        'sentence_fluency': (1, 6), 'conventions': (1, 6)},
    2: {'overall': (1, 6), 'content': (1, 6), 'organization': (1, 6), 'word_choice': (1, 6),
        'sentence_fluency': (1, 6), 'conventions': (1, 6)},
    3: {'overall': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
    4: {'overall': (0, 3), 'content': (0, 3), 'prompt_adherence': (0, 3), 'language': (0, 3), 'narrativity': (0, 3)},
    5: {'overall': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
    6: {'overall': (0, 4), 'content': (0, 4), 'prompt_adherence': (0, 4), 'language': (0, 4), 'narrativity': (0, 4)},
    7: {'overall': (0, 30), 'content': (0, 6), 'organization': (0, 6), 'conventions': (0, 6)},
    8: {'overall': (0, 60), 'content': (2, 12), 'organization': (2, 12), 'word_choice': (2, 12),
        'sentence_fluency': (2, 12), 'conventions': (2, 12)},
}


def to_snake(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]+", "", s)
    s = re.sub(r"\s+", "_", s)
    return s


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    x = str(x)
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()


def load_prompt_jsonl(prompt_jsonl_path: str):
    """
    Expects lines like:
    {"prompt": 1, "wrt_instruction": "...", "type": "argumentative"}
    """
    id2inst = {}
    id2type = {}
    with open(prompt_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = int(obj["prompt"])
            id2inst[pid] = clean_text(obj.get("wrt_instruction", ""))
            id2type[pid] = str(obj.get("type", "")).strip()
    return id2inst, id2type


def preprocess(in_csv: str, out_csv: str, prompt_jsonl: str | None = None):
    df = pd.read_csv(in_csv)

    # 1) clean columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")].copy()
    df.rename(columns={c: to_snake(c) for c in df.columns}, inplace=True)

    # 2) required cols
    if "content_text" not in df.columns:
        raise ValueError("Missing content_text")
    if "prompt_id" not in df.columns:
        raise ValueError("Missing prompt_id (needed for prompt-specific ranges)")

    # coerce prompt_id to numeric (safe)
    df["prompt_id"] = pd.to_numeric(df["prompt_id"], errors="coerce")

    # clean essay text
    df["content_text"] = df["content_text"].map(clean_text)

    # 3) Fill prompt_text (+ prompt_type) from prompt.jsonl if provided
    if prompt_jsonl is not None:
        id2inst, id2type = load_prompt_jsonl(prompt_jsonl)
        df["prompt_text"] = df["prompt_id"].map(lambda x: id2inst.get(int(x), "") if pd.notna(x) else "")
        df["prompt_type"] = df["prompt_id"].map(lambda x: id2type.get(int(x), "") if pd.notna(x) else "")
    else:
        if "prompt_text" not in df.columns:
            df["prompt_text"] = ""
        df["prompt_text"] = df["prompt_text"].map(clean_text)
        if "prompt_type" not in df.columns:
            df["prompt_type"] = ""

    # 4) drop style / voice
    for c in ["style", "voice"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # 5) trait columns present in the CSV
    all_traits = sorted(set().union(*[set(v.keys()) for v in PROMPT_MINMAX.values()]))
    existing_traits = [c for c in all_traits if c in df.columns]

    # cast existing trait cols to numeric
    for c in existing_traits:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def build_overall_score(row):
        pid = row["prompt_id"]
        if pd.isna(pid):
            return ""
        try:
            pid = int(pid)
        except Exception:
            return ""

        # overall이 컬럼에 없으면 빈 문자열
        if "overall" not in df.columns:
            return ""

        val = row.get("overall", np.nan)
        if pd.isna(val):
            return ""

        lo, hi = PROMPT_MINMAX.get(pid, {}).get("overall", (None, None))
        if lo is None or hi is None:
            return ""

        # int처럼 보이면 정수로
        sval = str(int(val)) if float(val).is_integer() else str(val)
        return f"Overall: {sval} ({lo}-{hi})"
    
    df["overall_score"] = df.apply(build_overall_score, axis=1)


    # 6) build trait_lines (raw score + range), EXCLUDING overall
    def build_trait_lines(row):
        pid = row["prompt_id"]
        if pd.isna(pid):
            return ""
        try:
            pid = int(pid)
        except Exception:
            return ""

        spec = PROMPT_MINMAX.get(pid, {})
        lines = []
        for trait, (lo, hi) in spec.items():
            if trait not in df.columns:
                continue
            val = row.get(trait, np.nan)
            if pd.isna(val):
                continue

            # pretty name: sentence_fluency -> Sentence Fluency
            name = trait.replace("_", " ").title()

            # show as int if it looks like int
            if float(val).is_integer():
                sval = str(int(val))
            else:
                sval = str(val)

            lines.append(f"- {name}: {sval} ({lo}-{hi})")
        return "\n".join(lines)

    df["trait_lines"] = df.apply(build_trait_lines, axis=1)

    # 7) filter
    df = df[df["content_text"].str.len() > 0].copy()
    df = df[df["trait_lines"].str.len() > 0].copy()
    df = df[df["prompt_text"].str.len() > 0].copy()

    # 8) keep
    keep = [
        "essay_id", "prompt_id", "prompt_type", "prompt_text",
        "content_text",
        "overall", "overall_score",          # keep raw overall if present
        "trait_lines",      # new field
    ]
    keep += [t for t in existing_traits if t != "overall"]

    df = df[[c for c in keep if c in df.columns]].copy()
    df.to_csv(out_csv, index=False)

    print(f"Saved {out_csv} | rows={len(df)}")
    if prompt_jsonl is not None:
        missing_prompt = (df["prompt_text"].str.len() == 0).sum()
        print("Missing prompt_text rows after merge:", missing_prompt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--prompt_jsonl", default="prompt.jsonl", help="Path to prompt.jsonl (ASAP prompts)")
    args = ap.parse_args()

    preprocess(args.in_csv, args.out_csv, prompt_jsonl=args.prompt_jsonl)
