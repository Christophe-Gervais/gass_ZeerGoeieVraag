import pandas as pd

# Load CSVs
df_cls = pd.read_csv("csv/bottle_classification_1344.csv")
df_ocr = pd.read_csv("bottle_ocr_filtered_13_44.csv")

# Replace empty strings or NaN in OCR with "?"
df_ocr["tarra"] = df_ocr["tarra"].fillna("?").replace("", "?")
df_ocr["year"] = df_ocr["year"].fillna("?").replace("", "?")

# Merge classification rows with OCR information
df_merged = df_cls.merge(df_ocr, on="bottle_id", how="left")

# After merge, fill missing tarra/year with "?"
df_merged["tarra"] = df_merged["tarra"].fillna("?")
df_merged["year"] = df_merged["year"].fillna("?")

# Save result
df_merged.to_csv("merged_output.csv", index=False)

print("Merged CSV saved as merged_output.csv")
