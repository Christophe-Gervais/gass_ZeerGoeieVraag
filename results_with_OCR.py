import pandas as pd
import numpy as np

# -----------------------------
# ---- LOAD CSV FILES ---------
# -----------------------------
gt = pd.read_csv("csv/groundtruth_1455.csv")
my = pd.read_csv("csv/merge_results_1455.csv")  

# -----------------------------
# ---- CLEAN CLASSIFICATION COLUMNS ----
# -----------------------------
gt["classification_clean"] = gt["classification"].astype(str).str.strip().str.upper()
my["classification_clean"] = my["classification"].astype(str).str.strip().str.upper()

# -----------------------------
# ---- AGGREGATE PER BOTTLE FOR PUSH METRICS ----
# -----------------------------
def aggregate_push(g):
    return pd.Series({
        "pushed_by_ai": g["pushed_by_ai"].all()
    })

my_agg = my.groupby("bottle_id").apply(aggregate_push)
my_agg.index.name = "id"

# -----------------------------
# ---- PUSH PERFORMANCE METRICS ----
# -----------------------------
TOTAL = len(my_agg)

gt_indexed = gt.set_index("id")
gt_indexed["is_nok_push"] = gt_indexed["Primagaz_status"].astype(str).str.strip().str.upper() == "NOK"
df_push = my_agg.join(gt_indexed[["is_nok_push"]])

h2 = df_push["pushed_by_ai"].sum()
correct_pushes = ((df_push["pushed_by_ai"]) & (df_push["is_nok_push"])).sum()
x1 = df_push["is_nok_push"].sum()

recall_push = correct_pushes / x1 if x1 > 0 else 0
precision_push = correct_pushes / h2 if h2 > 0 else 0
f1 = 2 * precision_push * recall_push / (precision_push + recall_push) if (precision_push + recall_push) > 0 else 0
beta = 2
fbeta = (1 + beta**2) * precision_push * recall_push / ((beta**2 * precision_push) + recall_push) if (precision_push + recall_push) > 0 else 0

# -----------------------------
# ---- STRICT CLASSIFICATION METRICS ----
# -----------------------------
gt_agg_class = gt.groupby("id").agg({
    "classification_clean": lambda x: (x == "NOK").any()
}).rename(columns={"classification_clean": "is_nok_true_class"})

def is_nok_by_ai(g):
    return (g["classification_clean"] == "NOK").sum() >= 2

my_agg_class = my.groupby("bottle_id").apply(is_nok_by_ai).rename("is_nok_pred_strict")
df_class = my_agg_class.to_frame().join(gt_agg_class, how="left")

x = df_class["is_nok_true_class"].sum()        
z = df_class["is_nok_pred_strict"].sum()      
y = ((df_class["is_nok_pred_strict"]) & (df_class["is_nok_true_class"])).sum()  

recall_classification = y / x if x > 0 else 0
precision_classification = y / z if z > 0 else 0
accuracy_classification = (df_class["is_nok_pred_strict"] == df_class["is_nok_true_class"]).mean()

# -----------------------------
# ---- TARA AND YEAR AGGREGATION ----
# -----------------------------
def aggregate_tara_year(g):
    # Take the first non-? value for tara and year (both rows should have same values)
    tara_vals = g["tarra"].replace("?", np.nan).dropna()
    year_vals = g["year"].replace("?", np.nan).dropna()
    
    return pd.Series({
        "tarra_ai": tara_vals.iloc[0] if len(tara_vals) > 0 else np.nan,
        "year_ai": year_vals.iloc[0] if len(year_vals) > 0 else np.nan
    })

my_tara_year = my.groupby("bottle_id").apply(aggregate_tara_year)
my_tara_year.index.name = "id"

# Join with groundtruth
df_tara_year = my_tara_year.join(gt_indexed[["tarra", "year"]], rsuffix="_gt")

# Convert to numeric
df_tara_year["tarra_ai"] = pd.to_numeric(df_tara_year["tarra_ai"], errors="coerce")
df_tara_year["tarra_gt"] = pd.to_numeric(df_tara_year["tarra"], errors="coerce")
df_tara_year["year_ai"] = pd.to_numeric(df_tara_year["year_ai"], errors="coerce")
df_tara_year["year_gt"] = pd.to_numeric(df_tara_year["year"], errors="coerce")

# -----------------------------
# ---- DANGEROUS FILLS (>500g overfill) ----
# -----------------------------
# Dangerous fill: AI tara is >0.5kg (500g) LESS than groundtruth tara
# (meaning the bottle would be filled too much because we think it's lighter)
df_tara_year["dangerous_fill_gt"] = (df_tara_year["tarra_gt"] - df_tara_year["tarra_ai"]) > 0.5
df_tara_year["dangerous_fill_detected"] = df_tara_year["dangerous_fill_gt"] & df_tara_year["tarra_ai"].notna()

dangerous_fills_total = df_tara_year["dangerous_fill_gt"].sum()
dangerous_fills_detected = df_tara_year["dangerous_fill_detected"].sum()

# Among bottles where AI read tara, how many would cause overfill?
tara_read_mask = df_tara_year["tarra_ai"].notna()
dangerous_among_read = df_tara_year[tara_read_mask]["dangerous_fill_gt"].sum()
tara_read_total = tara_read_mask.sum()

# -----------------------------
# ---- TARA METRICS ----
# -----------------------------
tara_readable = df_tara_year["tarra_ai"].notna()
tara_read_count = tara_readable.sum()
tara_read_pct = tara_read_count / TOTAL * 100

# Accuracy: exact match (within tolerance)
tara_correct = np.abs(df_tara_year["tarra_ai"] - df_tara_year["tarra_gt"]) < 0.1
tara_accuracy = tara_correct.sum() / tara_read_count * 100 if tara_read_count > 0 else 0

# Deviation/spread for read values
tara_diff = df_tara_year["tarra_ai"] - df_tara_year["tarra_gt"]
tara_deviation = tara_diff[tara_readable].abs().mean()
tara_std = tara_diff[tara_readable].std()

# Precision, Recall, F2 for tara
# For this, we consider "positive" as having the correct tara value
tara_gt_available = df_tara_year["tarra_gt"].notna()
tara_tp = (tara_readable & tara_gt_available & tara_correct).sum()
tara_fp = (tara_readable & (~tara_correct | df_tara_year["tarra_gt"].isna())).sum()
tara_fn = (tara_gt_available & ~tara_readable).sum()

tara_precision = tara_tp / (tara_tp + tara_fp) if (tara_tp + tara_fp) > 0 else 0
tara_recall = tara_tp / (tara_tp + tara_fn) if (tara_tp + tara_fn) > 0 else 0
tara_f2 = (1 + 4) * tara_precision * tara_recall / (4 * tara_precision + tara_recall) if (tara_precision + tara_recall) > 0 else 0

# -----------------------------
# ---- YEAR METRICS ----
# -----------------------------
year_readable = df_tara_year["year_ai"].notna()
year_read_count = year_readable.sum()
year_read_pct = year_read_count / TOTAL * 100

# Accuracy: exact match
year_correct = df_tara_year["year_ai"] == df_tara_year["year_gt"]
year_accuracy = year_correct.sum() / year_read_count * 100 if year_read_count > 0 else 0

# Deviation for read values
year_diff = df_tara_year["year_ai"] - df_tara_year["year_gt"]
year_deviation = year_diff[year_readable].abs().mean()
year_std = year_diff[year_readable].std()

# Precision, Recall, F2 for year
year_gt_available = df_tara_year["year_gt"].notna()
year_tp = (year_readable & year_gt_available & year_correct).sum()
year_fp = (year_readable & (~year_correct | df_tara_year["year_gt"].isna())).sum()
year_fn = (year_gt_available & ~year_readable).sum()

year_precision = year_tp / (year_tp + year_fp) if (year_tp + year_fp) > 0 else 0
year_recall = year_tp / (year_tp + year_fn) if (year_tp + year_fn) > 0 else 0
year_f2 = (1 + 4) * year_precision * year_recall / (4 * year_precision + year_recall) if (year_precision + year_recall) > 0 else 0

# -----------------------------
# ---- PRINT RESULTS ---------
# -----------------------------
print("=== PUSH PERFORMANCE ===")
print(f"Total bottles: {TOTAL}")
print(f"Groundtruth pushed: {x1}/{TOTAL}")
print(f"Bottles also pushed by AI: {correct_pushes}/{x1}")
print(f"Recall push: {recall_push:.2%}")
print(f"Total pushed by AI: {h2}/{TOTAL}")
print(f"Correct pushes by AI: {correct_pushes}/{h2}")
print(f"Precision push: {precision_push:.2%}")
print(f"F1 score: {f1:.4f}")
print(f"F-beta (beta=2): {fbeta:.4f}")

print("\n=== CLASSIFICATION PERFORMANCE ===")
print(f"Total bottles: {len(df_class)}")
print(f"Total bottles actually NOK (classification column): {x}")
print(f"Bottles flagged as NOK by AI: {z}")
print(f"Correctly flagged NOK bottles: {y}")
print(f"Recall (fraction of actual NOK detected): {recall_classification:.2%}")
print(f"Precision (fraction of flagged bottles correct): {precision_classification:.2%}")

print("\n=== DANGEROUS FILLS (>500g overfill) ===")
print(f"Among {tara_read_total} bottles where AI read tara:")
dangerous_among_read_pct = (dangerous_among_read / tara_read_total * 100) if tara_read_total > 0 else 0
print(f"  Dangerous overfills: {dangerous_among_read}/{tara_read_total} ({dangerous_among_read_pct:.1f}%)")

print("\n=== TARA READING PERFORMANCE ===")
print(f"AI was able to read tara: {tara_read_pct:.1f}% ({tara_read_count}/{TOTAL})")
print(f"AI accuracy for bottles read: {tara_accuracy:.1f}%")
print(f"Bottles with correct tara: {tara_correct.sum()}/{tara_read_count}")
print(f"Precision: {tara_precision:.2%}")
print(f"Recall: {tara_recall:.2%}")
print(f"F2 score: {tara_f2:.4f}")

print("\n=== YEAR READING PERFORMANCE ===")
print(f"AI was able to read year: {year_read_pct:.1f}% ({year_read_count}/{TOTAL})")
print(f"AI accuracy for bottles read: {year_accuracy:.1f}%")
print(f"Bottles with correct year: {year_correct.sum()}/{year_read_count}")
print(f"Precision: {year_precision:.2%}")
print(f"Recall: {year_recall:.2%}")
print(f"F2 score: {year_f2:.4f}")