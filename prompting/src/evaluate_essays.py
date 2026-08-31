import pandas as pd
from scorer import T5Scorer

def main():
    print("Loading downloaded essays from local CSV...")
    # 1. Load the data you already downloaded
    path = "downloaded_essays.csv"
    df = pd.read_csv(path)

    df = df.copy()

    # Clean the essays: Cut off anything after "Analysis", "Explanation", or "---"
    def clean_essay(text):
        if not isinstance(text, str): return text
        # Split at the first sign of a footer and take the top part
        for marker in ["## Analysis", "---", "# Explanation", "**Analysis"]:
            text = text.split(marker)[0]
        return text.strip()

    df['content_text'] = df['content_text'].apply(clean_essay)

    # 2. Score the Essays
    print("\nLoading T5 Scorer...")
    scorer = T5Scorer()
    
    print(f"Scoring {len(df)} essays on GPU...")
    # Passing 'content_text' exactly as we fixed earlier
    preds = scorer.score_essays(df['content_text'].tolist(), df['prompt_id'].tolist(), batch_size=8)
    
    df['pred'] = preds
    
    # Saving as a 'test' file so we don't accidentally overwrite your future full run
    df.to_csv("predictions.csv", index=False)
    print("Saved predictions to predictions.csv")
    
    # 3. Calculate Metrics
    print("\nCalculating Final QWK Metrics...")
    metrics_df = scorer.calculate_metrics(df)
    
    if not metrics_df.empty:
        print(metrics_df)
        metrics_df.to_csv("qwk_metrics.csv", index=False)
        print("Saved metrics to qwk_metrics.csv")
    else:
        print("Metrics dataframe is empty.")

if __name__ == "__main__":
    main()