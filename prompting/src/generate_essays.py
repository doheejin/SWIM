from data_manager import DataManager
from llm import ClaudeGenerator, GenerationMode

def main():
    print("Loading dataset...")
    dm = DataManager(
        dev_path = "../../dataset/asap/fold_0/total_test.csv",
        prompts_path = "../essay_prompts.json",
        ranges_path = "../ranges.json"
    )
    dataset = [row for row in dm.ds]
    
    generator = ClaudeGenerator(mode = GenerationMode.LOOKUP, model="claude-sonnet-4-6")
    requests = []
    
    print(f"Preparing {len(dataset)} items for batch processing...")
    for i in range(len(dm.ds)):
        context = dm.get_row_context(i)
        req = generator.prepare_batch_line(custom_id=i, context=context)
        requests.append(req)

    print("Submitting to Anthropic...")
    batch_id = generator.submit_batch(requests)
    
    # Save the batch ID to a file so Pipeline 2 can read it
    with open("current_batch_id.txt", "w") as f:
        f.write(batch_id)
        
    print(f"Success! Batch ID '{batch_id}' saved to current_batch_id.txt")
    print("Check your Anthropic dashboard to monitor progress.")

if __name__ == "__main__":
    main()