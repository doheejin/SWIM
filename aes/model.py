"""
AES (Automated Essay Scoring) wrapper around the ArTS T5 model.

Usage:
    scorer = ArTSScorer("/path/to/checkpoint-25000")
    preds  = scorer.score(["essay text..."], prompt_ids=[3])
    reward = scorer.reward(preds[0], target={"overall": 1, "content": 1}, prompt_id=3)
    rewards = scorer.batch_reward(essays, prompt_ids, target_list)
"""

import re
import json
import torch
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Per-prompt score ranges (min, max) for each trait
SCORE_RANGES = {
    1: {"overall": (2, 12), "content": (1, 6), "organization": (1, 6),
        "word_choice": (1, 6), "sentence_fluency": (1, 6), "conventions": (1, 6)},
    2: {"overall": (1, 6),  "content": (1, 6), "organization": (1, 6),
        "word_choice": (1, 6), "sentence_fluency": (1, 6), "conventions": (1, 6)},
    3: {"overall": (0, 3),  "content": (0, 3), "prompt_adherence": (0, 3),
        "language": (0, 3), "narrativity": (0, 3)},
    4: {"overall": (0, 3),  "content": (0, 3), "prompt_adherence": (0, 3),
        "language": (0, 3), "narrativity": (0, 3)},
    5: {"overall": (0, 4),  "content": (0, 4), "prompt_adherence": (0, 4),
        "language": (0, 4), "narrativity": (0, 4)},
    6: {"overall": (0, 4),  "content": (0, 4), "prompt_adherence": (0, 4),
        "language": (0, 4), "narrativity": (0, 4)},
    7: {"overall": (0, 30), "content": (0, 6), "organization": (0, 6),
        "conventions": (0, 6), "style": (0, 6)},
    8: {"overall": (0, 60), "content": (2, 12), "organization": (2, 12),
        "word_choice": (2, 12), "sentence_fluency": (2, 12),
        "conventions": (2, 12), "voice": (2, 12)},
}


def _normalize_trait_name(name: str) -> str:
    name = str(name).strip().lower().replace("-", " ")
    name = re.sub(r"\s+", " ", name)
    mapping = {
        "word choice": "word_choice",
        "sentence fluency": "sentence_fluency",
        "prompt adherence": "prompt_adherence",
    }
    return mapping.get(name, name.replace(" ", "_"))


def parse_pred_text(pred_text: str) -> dict:
    """Parse ArTS output 'content 2, language 1, overall 2, ...' into {trait: score}."""
    out = {}
    for chunk in str(pred_text).lower().split(","):
        chunk = chunk.strip()
        m = re.match(r"^(.+?)\s+([0-9]+(?:\.[0-9]+)?|nan)$", chunk)
        if not m:
            continue
        trait = _normalize_trait_name(m.group(1))
        val = m.group(2)
        if val != "nan":
            try:
                out[trait] = float(val)
            except ValueError:
                pass
    return out


class ArTSScorer:
    """
    Wrapper around the ArTS T5 checkpoint for scoring essays.

    Args:
        model_path: Path to the T5 checkpoint directory.
        device:     "cuda" or "cpu".
        max_src_len:    Max tokenized input length (essay + prompt prefix).
        max_new_tokens: Max tokens to generate (score string).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_src_len: int = 512,
        max_new_tokens: int = 64,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = (
            T5ForConditionalGeneration.from_pretrained(model_path)
            .to(self.device)
            .eval()
        )
        self.max_src_len = max_src_len
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def score(self, essays: list[str], prompt_ids: list[int]) -> list[dict]:
        """
        Score a batch of essays.

        Returns:
            List of {trait_name: float} dicts, one per essay.
        """
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
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return [parse_pred_text(d) for d in decoded]

    def reward(self, pred_traits: dict, target: dict, prompt_id: int) -> float:
        """
        Normalized trait accuracy reward.

        r = mean_t( 1 - |pred_t - target_t| / range_t )

        Bounded [0, 1].  1.0 = perfect match across all traits.
        Traits missing from either pred or target are skipped.
        """
        ranges = SCORE_RANGES.get(int(prompt_id), {})
        scores = []
        for trait, target_val in target.items():
            if target_val is None or (isinstance(target_val, float) and np.isnan(target_val)):
                continue
            pred_val = pred_traits.get(trait)
            if pred_val is None or (isinstance(pred_val, float) and np.isnan(pred_val)):
                continue
            r = ranges.get(trait)
            if r is None:
                continue
            range_size = r[1] - r[0]
            if range_size <= 0:
                continue
            scores.append(1.0 - abs(float(pred_val) - float(target_val)) / range_size)
        return float(np.mean(scores)) if scores else 0.0

    def batch_reward(
        self,
        essays: list[str],
        prompt_ids: list[int],
        targets: list[dict],
    ) -> list[float]:
        """Score essays and compute rewards against target trait dicts."""
        preds = self.score(essays, prompt_ids)
        return [
            self.reward(pred, target, pid)
            for pred, target, pid in zip(preds, targets, prompt_ids)
        ]
