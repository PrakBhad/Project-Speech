import pandas as pd

# Step 1: Load CSV file
# Change the file path to your actual CSV file
file_path = "fleurs_whisper_small_results.csv"
df = pd.read_csv(file_path)

# Step 2: Compute averages grouped by language
summary = (
    df.groupby("Language")[["WER", "BLEU", "BERTScore", "Processing Time (s)"]]
    .mean()
    .reset_index()
)

# Step 3: Round for readability
summary = summary.round(3)

# Step 4: Display summary table
print("\n=== Average Performance Metrics by Language ===")
print(summary.to_string(index=False))

# Step 5: Save results
summary.to_csv("average_metrics_by_language.csv", index=False)
summary.to_markdown("average_metrics_by_language.md", index=False)

print("\nAverages saved to 'average_metrics_by_language.csv' and '.md'")
