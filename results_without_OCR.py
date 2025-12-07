import pandas as pd
import numpy as np

# -----------------------------
# ---- LOAD CSV FILES ---------
# -----------------------------
gt = pd.read_csv("csv/groundtruth_1344.csv")
my = pd.read_csv("csv/bottle_classification_1344.csv")  

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

print("\n=== CLASSIFICATION PERFORMANCE (STRICT NOK) ===")
print(f"Total bottles: {len(df_class)}")
print(f"Total bottles actually NOK (classification column): {x}")
print(f"Bottles flagged as NOK by AI: {z}")
print(f"Correctly flagged NOK bottles: {y}")
print(f"Recall (fraction of actual NOK detected): {recall_classification:.2%}")
print(f"Precision (fraction of flagged bottles correct): {precision_classification:.2%}")
