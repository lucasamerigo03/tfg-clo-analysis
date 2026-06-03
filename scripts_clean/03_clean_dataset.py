"""
03_clean_dataset.py
-------------------
Reads clo_dataset_raw.csv, applies type casting, derives new variables,
and writes clo_dataset_clean.csv. Also prints a cleaning log for section 2.5.

Input:  data/processed/clo_dataset_raw.csv
Output: data/processed/clo_dataset_clean.csv
"""

import pandas as pd
import numpy as np
import os

INPUT_PATH  = "data/processed/clo_dataset_raw.csv"
OUTPUT_PATH = "data/processed/clo_dataset_clean.csv"

df = pd.read_csv(INPUT_PATH)

print("=" * 60)
print("CLEANING LOG — clo_dataset_clean.csv")
print("=" * 60)
print(f"\n[INPUT]  Rows: {len(df)} | Columns: {df.shape[1]}")

# ---------------------------------------------------------------------------
# 1. Type casting
# ---------------------------------------------------------------------------

df["vintage"] = df["vintage"].astype(int)

df["reinvestment_end_date"] = pd.to_datetime(df["reinvestment_end_date"], format="%Y-%m-%d")

float_cols = [
    "total_deal_size_mn",
    "reinvestment_period",
    "non_call_period",
    "ccc_limit_pct",
    "oc_ratio_class_a",
    "num_tranches",
    "class_a_size_mn",
    "sub_notes_size_mn",
]
for col in float_cols:
    before = df[col].isna().sum()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    after = df[col].isna().sum()
    if after > before:
        print(f"  [WARNING] {col}: {after - before} value(s) could not be coerced to float → set to NaN")

print("\n[1] Type casting applied:")
print(f"    vintage              → int")
print(f"    reinvestment_end_date→ datetime (YYYY-MM-DD)")
print(f"    {', '.join(float_cols)}")
print(f"    → float (pd.to_numeric, errors='coerce')")

# ---------------------------------------------------------------------------
# 2. Derived variables
# ---------------------------------------------------------------------------

df["class_a_pct"] = (df["class_a_size_mn"] / df["total_deal_size_mn"]) * 100
df["sub_notes_pct"] = (df["sub_notes_size_mn"] / df["total_deal_size_mn"]) * 100
df["deal_size_log"] = np.log(df["total_deal_size_mn"])

print("\n[2] Derived variables created:")
print(f"    class_a_pct    = class_a_size_mn / total_deal_size_mn * 100")
print(f"    sub_notes_pct  = sub_notes_size_mn / total_deal_size_mn * 100")
print(f"    deal_size_log  = ln(total_deal_size_mn)")

# ---------------------------------------------------------------------------
# 3. Notes on variables excluded from quantitative analysis
# ---------------------------------------------------------------------------
print("\n[3] Variables retained but excluded from cross-sectional analysis:")
print(f"    ccc_limit_pct: constant at 7.5% across all 36 deals.")
print(f"    No cross-sectional variation — documented but not used as regressor.")

# ---------------------------------------------------------------------------
# 4. Final validation
# ---------------------------------------------------------------------------
nan_counts = df.isna().sum()
nan_cols   = nan_counts[nan_counts > 0]

if nan_cols.empty:
    print("\n[4] NaN check: 0 missing values across all columns. Dataset complete.")
else:
    print("\n[4] NaN check: missing values detected:")
    print(nan_cols.to_string())

print(f"\n[OUTPUT] Rows: {len(df)} | Columns: {df.shape[1]}")
print(f"         New columns: class_a_pct, sub_notes_pct, deal_size_log")

# ---------------------------------------------------------------------------
# 5. Descriptive summary of derived variables (for section 2.5 reference)
# ---------------------------------------------------------------------------
print("\n[5] Descriptive summary of derived variables:")
print(df[["class_a_pct", "sub_notes_pct", "deal_size_log"]].describe().round(3).to_string())

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\n[SAVED] {OUTPUT_PATH}")
print("=" * 60)
