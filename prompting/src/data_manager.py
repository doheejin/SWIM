from datasets import load_dataset
import json
import pandas as pd

class DataManager:
    def __init__(self, dev_path, prompts_path, ranges_path):
        dataset_dict = load_dataset("csv", data_files={"data": dev_path})
        self.ds = dataset_dict["data"]
        
        with open(prompts_path, 'r') as f:
            self.prompts = {item["prompt"]: item for item in json.load(f)}
        with open(ranges_path, 'r') as f:
            self.ranges = json.load(f)
        
        self.score_cols = [
            'overall', 'content', 'prompt adherence', 'language', 'narrativity', 
            'organization', 'word choice', 'sentence fluency', 'conventions', 'style', 'voice'
        ]

    def get_row_context(self, index):
        """
        In HF Datasets, indexing ds[index] returns a dictionary 
        representing that row. No more .loc vs .iloc confusion!
        """
        row = self.ds[index]
        
        # prompt_id is often loaded as a string or int; we cast to be safe
        p_id = int(row['prompt_id'])
        prompt_key = f"prompt_{p_id}"
        
        scores_dict = {}
        ranges_dict = {}
        for trait in self.score_cols:
            val = row.get(trait)
            # HF Datasets represents empty CSV cells as None
            if val is None or (isinstance(val, float) and pd.isna(val)):
                scores_dict[trait] = "Not Applicable"
            else:
                info = self.ranges.get(prompt_key, {}).get(trait)
                score_range = f"(Range: {info['min']}-{info['max']})" if info else ""
                ranges_dict[trait] = score_range
                clean_val = int(val) if isinstance(val, (float, int)) else val
                scores_dict[trait] = clean_val
        
        return {
            "scores": scores_dict,
            "ranges": ranges_dict,
            "grade": self.prompts[p_id]['grade'],
            "instruction": self.prompts[p_id]['wrt_instruction'],
            "prompt_id": p_id,
            "target": row['target']
        }