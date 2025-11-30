import pandas as pd
import numpy as np

# =============================================================================
# LOAD CSV FILES
# =============================================================================

gt = pd.read_csv("csv/groundtruth_1344.csv")
my = pd.read_csv("csv/merge_results_1344.csv")

# =============================================================================
# CLEAN COLUMNS & IDS
# =============================================================================

gt.columns = gt.columns.str.strip().str.lower()
my.columns = my.columns.str.strip().str.lower()

# Convert ID columns to numeric for merging
gt["id"] = pd.to_numeric(gt["id"], errors="coerce").astype("Int64")
my["bottle_id"] = pd.to_numeric(my["bottle_id"], errors="coerce").astype("Int64")

# Clean classification columns
gt["classification_clean"] = gt["classification"].astype(str).str.strip().str.upper()
my["classification_clean"] = my["classification"].astype(str).str.strip().str.upper()

# =============================================================================
# AGGREGATE AI TARA/YEAR
# =============================================================================

def aggregate_tara_year(g):
    tara_vals = g["tarra"].replace("?", np.nan).dropna()
    year_vals = g["year"].replace("?", np.nan).dropna()
    return pd.Series({
        "tarra_ai": tara_vals.iloc[0] if len(tara_vals) > 0 else np.nan,
        "year_ai": year_vals.iloc[0] if len(year_vals) > 0 else np.nan
    })

my_tara_year = my.groupby("bottle_id").apply(aggregate_tara_year)
my_tara_year.index.name = "id"

# =============================================================================
# AGGREGATE AI PUSH (STRICT - ALL READINGS MUST BE TRUE OR YEAR < 2025)
# =============================================================================

def strict_push_aggregation(g):
    # All readings must be True for the bottle to be considered "pushed by AI"
    all_true = (g["pushed_by_ai"] == True).all()
    
    # OR check if year < 2025
    year_vals = pd.to_numeric(g["year"].replace("?", np.nan), errors="coerce")
    has_old_year = (year_vals < 2025).any()
    
    return pd.Series({"pushed_by_ai": all_true or has_old_year})

my_push = my.groupby("bottle_id").apply(strict_push_aggregation)
my_push.index.name = "id"

# =============================================================================
# PREPARE GROUNDTRUTH
# =============================================================================

gt["year_gt_clean"] = pd.to_numeric(gt["year"], errors="coerce")
gt["primagaz_status_clean"] = gt["primagaz_status"].astype(str).str.strip().str.upper()

# Create a separate dataframe for push metrics
gt_for_push = gt.copy()
gt_for_push_indexed = gt_for_push.merge(
    my_tara_year[["year_ai"]], 
    left_on="id", 
    right_on="id", 
    how="left"
)
gt_for_push_indexed["year_ai_clean"] = pd.to_numeric(gt_for_push_indexed["year_ai"], errors="coerce")
gt_for_push_indexed["year_final"] = gt_for_push_indexed["year_gt_clean"].combine_first(
    gt_for_push_indexed["year_ai_clean"]
)

# GT push label: NOK or year_final < 2025
gt_for_push_indexed["is_nok_push"] = (
    (gt_for_push_indexed["primagaz_status_clean"] == "NOK") | 
    (gt_for_push_indexed["year_final"] < 2025)
)

# =============================================================================
# PUSH METRICS (with F2 and F3 scores)
# =============================================================================

df_push = gt_for_push_indexed.set_index("id").join(my_push, how="left")
df_push["pushed_by_ai"] = df_push["pushed_by_ai"].fillna(False)

TOTAL = len(df_push)
h2 = df_push["pushed_by_ai"].sum()
correct_pushes = (df_push["pushed_by_ai"] & df_push["is_nok_push"]).sum()
x1 = df_push["is_nok_push"].sum()

recall_push = correct_pushes / x1 if x1 > 0 else 0
precision_push = correct_pushes / h2 if h2 > 0 else 0
f1 = (
    2 * precision_push * recall_push / (precision_push + recall_push) 
    if (precision_push + recall_push) > 0 else 0
)

# F2 score
beta2 = 2
f2 = (
    (1 + beta2**2) * precision_push * recall_push / 
    ((beta2**2 * precision_push) + recall_push) 
    if (precision_push + recall_push) > 0 else 0
)

# F3 score
beta3 = 3
f3 = (
    (1 + beta3**2) * precision_push * recall_push / 
    ((beta3**2 * precision_push) + recall_push) 
    if (precision_push + recall_push) > 0 else 0
)

# =============================================================================
# STRICT CLASSIFICATION METRICS
# =============================================================================

gt_agg_class = gt.groupby("id").agg({
    "classification_clean": lambda x: (x == "NOK").any()
}).rename(columns={"classification_clean": "is_nok_true_class"})

def is_nok_by_ai(g):
    return (g["classification_clean"] == "NOK").sum() >= 2

my_agg_class = my.groupby("bottle_id").apply(is_nok_by_ai).rename("is_nok_pred_strict")
df_class = my_agg_class.to_frame().join(gt_agg_class, how="left")

x = df_class["is_nok_true_class"].sum()
z = df_class["is_nok_pred_strict"].sum()
y = (df_class["is_nok_pred_strict"] & df_class["is_nok_true_class"]).sum()

recall_classification = y / x if x > 0 else 0
precision_classification = y / z if z > 0 else 0
accuracy_classification = (
    df_class["is_nok_pred_strict"] == df_class["is_nok_true_class"]
).mean()

# =============================================================================
# DANGEROUS FILLS - ONLY for bottles FILLED (not pushed) with >500g gas
# =============================================================================

# Use ORIGINAL groundtruth indexed (not the modified one for push)
gt_indexed = gt.set_index("id")
df_tara_year = my_tara_year.join(gt_indexed[["tarra", "year"]], rsuffix="_gt")

# Convert to numeric
df_tara_year["tarra_ai"] = pd.to_numeric(df_tara_year["tarra_ai"], errors="coerce")
df_tara_year["tarra_gt"] = pd.to_numeric(df_tara_year["tarra"], errors="coerce")
df_tara_year["year_ai"] = pd.to_numeric(df_tara_year["year_ai"], errors="coerce")
df_tara_year["year_gt"] = pd.to_numeric(df_tara_year["year"], errors="coerce")

# Join with push decisions
df_tara_year = df_tara_year.join(df_push[["pushed_by_ai"]])

# Filter: only bottles that were FILLED (not pushed) AND have tara read
filled_mask = ~df_tara_year["pushed_by_ai"].fillna(False)
tara_read_and_filled = df_tara_year["tarra_ai"].notna() & filled_mask

# Dangerous fill: AI would fill >500g (0.5kg) more than safe
# This happens when GT tara - AI tara > 0.5
df_tara_year["dangerous_fill_gt"] = (
    (df_tara_year["tarra_gt"] - df_tara_year["tarra_ai"]) > 0.5
)

# Count dangerous fills only among FILLED bottles with tara read
dangerous_among_filled = df_tara_year[tara_read_and_filled]["dangerous_fill_gt"].sum()
tara_read_and_filled_total = tara_read_and_filled.sum()

# =============================================================================
# TARA METRICS with RMS-E
# =============================================================================

tara_readable = df_tara_year["tarra_ai"].notna()
tara_read_count = tara_readable.sum()
tara_read_pct = tara_read_count / TOTAL * 100

tara_correct = np.abs(df_tara_year["tarra_ai"] - df_tara_year["tarra_gt"]) < 0.1
tara_accuracy = (
    tara_correct.sum() / tara_read_count * 100 if tara_read_count > 0 else 0
)

# RMS-E calculation (only where both AI and GT tara are available and GT is not ?)
tara_diff = df_tara_year["tarra_ai"] - df_tara_year["tarra_gt"]
tara_gt_available = df_tara_year["tarra_gt"].notna()
both_available = tara_readable & tara_gt_available
tara_rmse = np.sqrt((tara_diff[both_available]**2).mean()) if both_available.sum() > 0 else np.nan

# Old metrics for comparison
tara_deviation = tara_diff[tara_readable].abs().mean()
tara_std = tara_diff[tara_readable].std()

tara_tp = (tara_readable & tara_gt_available & tara_correct).sum()
tara_fp = (tara_readable & (~tara_correct | df_tara_year["tarra_gt"].isna())).sum()
tara_fn = (tara_gt_available & ~tara_readable).sum()

tara_precision = tara_tp / (tara_tp + tara_fp) if (tara_tp + tara_fp) > 0 else 0
tara_recall = tara_tp / (tara_tp + tara_fn) if (tara_tp + tara_fn) > 0 else 0
tara_f2 = (
    (1 + 4) * tara_precision * tara_recall / (4 * tara_precision + tara_recall) 
    if (tara_precision + tara_recall) > 0 else 0
)

# =============================================================================
# YEAR METRICS
# =============================================================================

year_readable = df_tara_year["year_ai"].notna()
year_read_count = year_readable.sum()
year_read_pct = year_read_count / TOTAL * 100

year_correct = df_tara_year["year_ai"] == df_tara_year["year_gt"]
year_accuracy = (
    year_correct.sum() / year_read_count * 100 if year_read_count > 0 else 0
)

year_diff = df_tara_year["year_ai"] - df_tara_year["year_gt"]
year_deviation = year_diff[year_readable].abs().mean()
year_std = year_diff[year_readable].std()

year_gt_available = df_tara_year["year_gt"].notna()
year_tp = (year_readable & year_gt_available & year_correct).sum()
year_fp = (year_readable & (~year_correct | df_tara_year["year_gt"].isna())).sum()
year_fn = (year_gt_available & ~year_readable).sum()

year_precision = year_tp / (year_tp + year_fp) if (year_tp + year_fp) > 0 else 0
year_recall = year_tp / (year_tp + year_fn) if (year_tp + year_fn) > 0 else 0
year_f2 = (
    (1 + 4) * year_precision * year_recall / (4 * year_precision + year_recall) 
    if (year_precision + year_recall) > 0 else 0
)

# =============================================================================
# PRINT RESULTS
# =============================================================================

print("=== PUSH PERFORMANCE ===")
print(f"Total bottles: {TOTAL}")
print(f"Groundtruth pushed: {x1}/{TOTAL}")
print(f"Bottles also pushed by AI: {correct_pushes}/{x1}")
print(f"Recall push: {recall_push:.2%}")
print(f"Total pushed by AI: {h2}/{TOTAL}")
print(f"Correct pushes by AI: {correct_pushes}/{h2}")
print(f"Precision push: {precision_push:.2%}")
print(f"F1 score: {f1:.4f}")
print(f"F2 score: {f2:.4f}")
print(f"F3 score: {f3:.4f}")

print("\n=== CLASSIFICATION PERFORMANCE ===")
print(f"Total bottles: {len(df_class)}")
print(f"Total bottles actually NOK (classification column): {x}")
print(f"Bottles flagged as NOK by AI: {z}")
print(f"Correctly flagged NOK bottles: {y}")
print(f"Recall (fraction of actual NOK detected): {recall_classification:.2%}")
print(f"Precision (fraction of flagged bottles correct): {precision_classification:.2%}")

print("\n=== DANGEROUS FILLS (>500g overfill) - FILLED BOTTLES ONLY ===")
print(f"Bottles filled by AI (not pushed): {(~df_push['pushed_by_ai']).sum()}/{TOTAL}")
print(f"Among {tara_read_and_filled_total} filled bottles where AI read tara:")
dangerous_among_filled_pct = (dangerous_among_filled / tara_read_and_filled_total * 100) if tara_read_and_filled_total > 0 else 0
print(f"  Dangerous overfills (>500g): {dangerous_among_filled}/{tara_read_and_filled_total} ({dangerous_among_filled_pct:.1f}%)")

print("\n=== TARA READING PERFORMANCE ===")
print(f"AI was able to read tara: {tara_read_pct:.1f}% ({tara_read_count}/{TOTAL})")
print(f"AI accuracy for bottles read: {tara_accuracy:.1f}%")
print(f"Bottles with correct tara: {tara_correct.sum()}/{tara_read_count}")
print(f"RMS-E: {tara_rmse:.4f} kg" if not np.isnan(tara_rmse) else "RMS-E: N/A")
print(f"STDEV (old metric): {tara_std:.4f} kg" if not np.isnan(tara_std) else "STDEV: N/A")
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