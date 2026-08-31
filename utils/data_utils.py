"""
Shared prompt formatting and data loading utilities.
Used by generation/sft_train.py and all rl/ scripts.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import load_dataset

# Trait list per prompt ID
PROMPT_TRAITS = {
    1: ["overall", "content", "organization", "word_choice", "sentence_fluency", "conventions"],
    2: ["overall", "content", "organization", "word_choice", "sentence_fluency", "conventions"],
    3: ["overall", "content", "prompt_adherence", "language", "narrativity"],
    4: ["overall", "content", "prompt_adherence", "language", "narrativity"],
    5: ["overall", "content", "prompt_adherence", "language", "narrativity"],
    6: ["overall", "content", "prompt_adherence", "language", "narrativity"],
    7: ["overall", "content", "organization", "conventions", "style"],
    8: ["overall", "content", "organization", "word_choice", "sentence_fluency", "conventions", "voice"],
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_messages(
    prompt_id: str,
    prompt_text: str,
    overall_score,
    trait_lines: str,
    mode: str = "multitrait",
) -> list[dict]:
    """
    Build the system + user messages for the generation model.
    The assistant message is NOT included here; add it separately for SFT.
    """
    prompt_id = str(prompt_id).strip()
    prompt_text = (prompt_text or "").strip()
    trait_lines = (trait_lines or "").strip()

    if mode == "holistic":
        system_msg = "You simulate student writing behavior based on overall writing proficiency."
        user_msg = (
            "You are simulating a student's writing based on the overall proficiency.\n"
            f"The student receives the following overall score for this writing task: {overall_score}\n\n"
            "Given the writing instruction below, write an essay that reflects this student's overall writing ability.\n\n"
            "Important:\n"
            "- Do NOT mention scores or proficiency levels in the essay.\n"
            "- Output ONLY the essay text.\n\n"
            f"Prompt ID: {prompt_id}\n"
            f"Writing Instruction:\n{prompt_text}"
        )
    else:  # multitrait (default)
        system_msg = "You simulate student writing behavior conditioned on trait-level proficiency."
        user_msg = (
            "You are simulating a student's writing based on trait-level proficiency.\n"
            "The student receives the following trait scores for this task:\n\n"
            f"{trait_lines}\n\n"
            "Given the writing instruction below, write an essay that matches this student's trait profile.\n"
            "Important:\n"
            "- Do NOT mention the scores in the essay.\n"
            "- Output ONLY the essay text.\n\n"
            f"Prompt ID: {prompt_id}\n"
            f"Writing Instruction:\n{prompt_text}\n"
        )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]


def format_for_sft(tokenizer, ex: dict, mode: str = "multitrait", prompt_map: dict = None) -> dict:
    """
    Format one CSV row into a full chat string (system + user + assistant).
    The assistant turn contains the gold essay.
    Returns {"text": <full chat string>}.
    prompt_map: {prompt_id(int) -> wrt_instruction} loaded from prompt.jsonl.
                If provided, overrides prompt_text from the CSV row.
    """
    pid = int(str(ex.get("prompt_id", 1)).strip())
    prompt_text = prompt_map.get(pid, "") if prompt_map else ex.get("prompt_text", "")
    messages = build_messages(
        prompt_id=pid,
        prompt_text=prompt_text,
        overall_score=ex.get("overall_score"),
        trait_lines=ex.get("trait_lines", ""),
        mode=mode,
    )
    essay_col = "generated_essay" if "generated_essay" in ex else "content_text"
    gold_essay = (ex.get(essay_col) or "").strip()
    messages.append({"role": "assistant", "content": gold_essay})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    return {"text": text}


def format_for_sft_weighted(tokenizer, ex: dict, mode: str = "multitrait", prompt_map: dict = None) -> dict:
    """
    Same as format_for_sft but also propagates sample_weight.
    Used by rl/filtered_retrain.py.
    """
    result = format_for_sft(tokenizer, ex, mode, prompt_map)
    w = ex.get("sample_weight", 1.0)
    try:
        w = float(w)
    except (TypeError, ValueError):
        w = 1.0
    w = max(0.2, min(0.8, w))
    result["sample_weight"] = w
    return result


def format_prompt_only(tokenizer, ex: dict, mode: str = "multitrait", prompt_map: dict = None) -> dict:
    """
    Format one row into a prompt-only string (no assistant turn).
    Used by GRPO dataset (model generates the completion).
    Also stores prompt_id and target traits as JSON for the reward function.
    prompt_map: {prompt_id(int) -> wrt_instruction} loaded from prompt.jsonl.
    """
    pid = int(str(ex.get("prompt_id", 1)).strip())
    prompt_text = prompt_map.get(pid, "") if prompt_map else ex.get("prompt_text", "")
    messages = build_messages(
        prompt_id=pid,
        prompt_text=prompt_text,
        overall_score=ex.get("overall_score"),
        trait_lines=ex.get("trait_lines", ""),
        mode=mode,
    )
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    # Build target trait dict from individual columns
    pid = int(ex.get("prompt_id", 1))
    traits = PROMPT_TRAITS.get(pid, [])
    target = {}
    for t in traits:
        val = ex.get(t)
        if val is not None:
            try:
                target[t] = float(val)
            except (TypeError, ValueError):
                pass

    return {
        "prompt": prompt,
        "prompt_id": pid,
        "target_json": json.dumps(target),
    }


# ---------------------------------------------------------------------------
# Data collator: mask prompt tokens, pass sample_weight through
# ---------------------------------------------------------------------------

@dataclass
class CompletionOnlyCollator:
    """
    Tokenizes a batch of {"text": str, "sample_weight": float} examples.
    Masks everything up to and including the assistant header so loss is
    computed only on the completion (essay) tokens.
    """
    tokenizer: Any
    response_template_ids: List[int]
    max_length: Optional[int] = None

    def _find_sublist(self, haystack: list, needle: list) -> int:
        n = len(needle)
        for i in range(len(haystack) - n + 1):
            if haystack[i : i + n] == needle:
                return i
        return -1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts          = [f["text"] for f in features]
        sample_weights = [f.get("sample_weight", 1.0) for f in features]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=False,
        )

        import torch
        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels         = input_ids.clone()

        for i in range(input_ids.size(0)):
            ids = input_ids[i].tolist()
            pos = self._find_sublist(ids, self.response_template_ids)
            if pos == -1:
                labels[i, :] = -100
                continue
            cut = pos + len(self.response_template_ids)
            labels[i, :cut] = -100
            labels[i, attention_mask[i] == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
            "sample_weight":  torch.tensor(sample_weights, dtype=torch.float),
        }


def get_response_template_ids(tokenizer, model_name: str) -> list[int]:
    """Return the token IDs for the assistant-turn header."""
    if "Mistral" in model_name:
        return tokenizer.encode("[/INST]", add_special_tokens=False)
    # Qwen chat template uses "<|im_start|>assistant\n"
    return tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_prompt_map(jsonl_path: str) -> dict:
    """
    Load prompt_id -> wrt_instruction mapping from prompt.jsonl.
    Returns {1: "Write a letter ...", 2: "Censorship ...", ...}
    """
    prompt_map = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompt_map[int(obj["prompt"])] = obj["wrt_instruction"]
    return prompt_map


def load_sft_dataset(data_path: str):
    """Load real ASAP essays for basic SFT. Expects total_train.csv / total_dev.csv."""
    train = load_dataset("csv", data_files=os.path.join(data_path, "total_train.csv"))["train"]
    dev   = load_dataset("csv", data_files=os.path.join(data_path, "total_dev.csv"))["train"]
    return train, dev


def load_retrain_dataset(data_path: str, train_file: str = "retrain_total_train_1_00.csv"):
    """
    Load synthetic essays for filtered retraining.
    The CSV must have a 'sample_weight' column (output of retraining_data_construct.py).
    """
    train = load_dataset("csv", data_files=os.path.join(data_path, train_file))["train"]
    dev   = load_dataset("csv", data_files=os.path.join(data_path, "total_dev.csv"))["train"]
    return train, dev


def load_grpo_dataset(data_path: str):
    """
    Load dataset for GRPO.
    Requires individual trait score columns (content, organization, etc.)
    alongside standard prompt columns.
    Uses total_train.csv as conditioning targets.
    """
    train = load_dataset("csv", data_files=os.path.join(data_path, "total_train.csv"))["train"]
    return train


def load_grpo_eval_dataset(data_path: str):
    """Load eval split (total_dev.csv) for GRPO reward monitoring."""
    dev = load_dataset("csv", data_files=os.path.join(data_path, "total_dev.csv"))["train"]
    return dev
