import pandas as pd
import glob
import os

OUTPUT_DIR = "phase2_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all df*.csv files
csv_files = glob.glob(os.path.expanduser("~/Downloads/GEPA/data/df*.csv"))


if not csv_files:
    raise RuntimeError("No df*.csv files found in directory.")

print(f"Found files: {csv_files}")

all_dfs = []

for file in csv_files:
    df = pd.read_csv(file)

    # Standardize column names
    df = df.rename(columns={
        "Sentences": "text",
        "Label": "label"
    })

    df["source"] = file
    all_dfs.append(df[["text", "label", "source"]])

# Combine all datasets
combined_df = pd.concat(all_dfs, ignore_index=True)

# Remove duplicates
combined_df = combined_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print(f"Total combined rows: {len(combined_df)}")
print(f"Label distribution:\n{combined_df['label'].value_counts()}")

# Separate positives and negatives
positives = combined_df[combined_df["label"] == "Positive"].copy()
negatives = combined_df[combined_df["label"] == "Negative"].copy()

# Convert labels to binary
positives["label"] = 1
negatives["label"] = 0

# Save files
combined_df.to_csv(os.path.join(OUTPUT_DIR, "stereotype_combined_all.csv"), index=False)
positives.to_csv(os.path.join(OUTPUT_DIR, "stereotype_positive_only.csv"), index=False)
negatives.to_csv(os.path.join(OUTPUT_DIR, "stereotype_negative_only.csv"), index=False)

print("\nSaved:")
print(" - stereotype_combined_all.csv")
print(" - stereotype_positive_only.csv")
print(" - stereotype_negative_only.csv")
