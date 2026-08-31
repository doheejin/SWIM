import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from eval_qwk import Evaluator as QWKEvaluator

class T5Scorer:
    """Handles the T5 model inference and QWK calculation."""
    def __init__(self, model_path="Heejindo/ArTS-T5-f0", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        print(f"Loading {model_path} from cache: {cache_dir}...")
        
        # 1. Clean, original loading method
        self.tokenizer = T5Tokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            use_fast=False,         # Force the Python implementation
            extra_ids=0,            # Tell it not to expect those extra tokens from a config
            legacy=True             # Critical for older T5 weights
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir=cache_dir
        ).to(self.device)
        
        # 2. Trait arrays
        self.trait_sets = [
            ['overall','content','word choice','organization','sentence fluency','conventions'],
            ['overall','content','word choice','organization','sentence fluency','conventions'],
            ['overall','content','prompt adherence','language','narrativity'],
            ['overall','content','prompt adherence','language','narrativity'],
            ['overall','content','prompt adherence','language','narrativity'],
            ['overall','content','prompt adherence','language','narrativity'],
            ['overall','content','organization','conventions','style'],
            ['overall','content','word choice','organization','sentence fluency','conventions', 'voice']
        ]

    def score_essays(self, essays, prompt_ids, batch_size=4):
        all_preds = []
        for i in range(0, len(essays), batch_size):
            batch_texts = essays[i:i+batch_size]
            batch_pids = prompt_ids[i:i+batch_size]
            
            inputs = [f"score the essay of the prompt {pid}: {txt}" for pid, txt in zip(batch_pids, batch_texts)]
            enc = self.tokenizer(inputs, max_length=512, padding=True, truncation=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(input_ids=enc.input_ids, attention_mask=enc.attention_mask, max_length=162)
            
            all_preds.extend([self.tokenizer.decode(o, skip_special_tokens=True) for o in out])
        return all_preds

    def calculate_metrics(self, df):
        results = []
        for p_id in range(1, 9):
            sub = df[df['prompt_id'] == p_id].reset_index(drop=True)
            
            if sub.empty: continue
            
            evaluator = QWKEvaluator(self.trait_sets[p_id-1])
            res = evaluator.evaluate_notnull(sub['pred'], sub['target'].astype(str))
            res['prompt_id'] = p_id
            results.append(res)
            
        return pd.DataFrame(results)