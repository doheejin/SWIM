import os
import pandas as pd
from data_manager import DataManager
from llm import ClaudeGenerator, GenerationMode

def main():
    if not os.path.exists("current_batch_id.txt"):
        print("Error: current_batch_id.txt not found. Run the generate script first.")
        return
        
    with open("current_batch_id.txt", "r") as f:
        batch_id = f.read().strip()
        
    generator = ClaudeGenerator(mode = GenerationMode.LOOKUP, model="claude-sonnet-4-6")
    
    status = generator.check_batch_status(batch_id)
    if status.processing_status != "ended":
        print(f"Batch is currently: {status.processing_status}. Please run this script again when it is 'ended'.")
        return
        
    print("Batch complete! Downloading results...")
    
    dm = DataManager(
        dev_path = "../../dataset/asap/fold_0/total_test.csv",
        prompts_path="../essay_prompts.json",
        ranges_path="../ranges.json"
    )
    dataset = [row for row in dm.ds] 
    results_list = []
    
    for result in generator.get_batch_results(batch_id):
        idx = int(result.custom_id)
        original_row = dataset[idx] 
        
        if result.result.type == "succeeded":
            essay_text = result.result.message.content[0].text
        else:
            essay_text = f"ERROR: {result.result.error.message}"
            
        # Build the exact dictionary matching your target format
        results_list.append({
            "idx": idx, # We keep this temporarily for sorting
            "essay_id": original_row.get("essay_id", ""),
            "prompt_id": original_row.get("prompt_id", ""),
            "overall": original_row.get("overall", ""),
            "content": original_row.get("content", ""),
            "prompt adherence": original_row.get("prompt adherence", ""),
            "language": original_row.get("language", ""),
            "narrativity": original_row.get("narrativity", ""),
            "content_text": essay_text,
            "organization": original_row.get("organization", ""),
            "word choice": original_row.get("word choice", ""),
            "sentence fluency": original_row.get("sentence fluency", ""),
            "conventions": original_row.get("conventions", ""),
            "style": original_row.get("style", ""),
            "voice": original_row.get("voice", ""),
            "target": original_row.get("target", "")
        })
        
    # Convert to DataFrame
    df = pd.DataFrame(results_list)
    
    # Sort by the original index, then set it as the DataFrame's actual index
    df = df.sort_values("idx").set_index("idx")
    df.index.name = None # Removes the "idx" header name to match your exact format
    
    # Force the exact column order you requested
    columns_order = [
        "essay_id", "prompt_id", "overall", "content", "prompt adherence", 
        "language", "narrativity", "content_text", "organization", 
        "word choice", "sentence fluency", "conventions", "style", "voice", "target"
    ]
    df = df[columns_order]
    
    # index=True is what creates that leading comma (e.g., ",essay_id,...")
    df.to_csv("downloaded_essays.csv", index=True)
    print(f"Success! Saved {len(df)} essays to downloaded_essays.csv in the exact requested format.")

if __name__ == "__main__":
    main()