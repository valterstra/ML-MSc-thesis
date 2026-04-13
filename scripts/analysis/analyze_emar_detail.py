"""
Analyze emar_detail.csv (87M rows) in chunks.
Answers 8 questions about dose/route/infusion completeness.
"""
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

CSV = Path(r"C:\Users\ValterAdmin\Documents\VS code projects\TemporaryMLthesis\downloads\hosp\emar_detail.csv")
CHUNKSIZE = 1_000_000

# Accumulators
total_rows = 0
non_null = Counter()  # column -> count of non-null
dose_given_unit_counts = Counter()
route_counts = Counter()
admin_type_counts = Counter()
product_desc_counts = Counter()
infusion_rate_unit_counts = Counter()
complete_dose_not_given_counts = Counter()

# For Q8: dose_due vs dose_given comparison
dose_compare_total = 0
dose_less_given = 0
dose_equal = 0
dose_more_given = 0
dose_compare_examples = []

TRACK_COLS = [
    "dose_given", "dose_given_unit", "route",
    "administration_type", "infusion_rate", "infusion_rate_unit",
    "dose_due", "dose_due_unit", "complete_dose_not_given",
    "product_description",
]

print(f"Reading {CSV} in chunks of {CHUNKSIZE:,} ...")

for i, chunk in enumerate(pd.read_csv(CSV, chunksize=CHUNKSIZE, low_memory=False)):
    n = len(chunk)
    total_rows += n

    # Non-null counts
    for col in TRACK_COLS:
        if col in chunk.columns:
            non_null[col] += chunk[col].notna().sum()

    # Q2: dose_given_unit
    if "dose_given_unit" in chunk.columns:
        vals = chunk["dose_given_unit"].dropna()
        dose_given_unit_counts.update(vals.value_counts().to_dict())

    # Q3: route
    if "route" in chunk.columns:
        vals = chunk["route"].dropna()
        route_counts.update(vals.value_counts().to_dict())

    # Q4: administration_type
    if "administration_type" in chunk.columns:
        vals = chunk["administration_type"].dropna()
        admin_type_counts.update(vals.value_counts().to_dict())

    # Q5: product_description (top values)
    if "product_description" in chunk.columns:
        vals = chunk["product_description"].dropna()
        product_desc_counts.update(vals.value_counts().to_dict())

    # Q6: infusion_rate_unit
    if "infusion_rate_unit" in chunk.columns:
        vals = chunk["infusion_rate_unit"].dropna()
        infusion_rate_unit_counts.update(vals.value_counts().to_dict())

    # Q7: complete_dose_not_given
    if "complete_dose_not_given" in chunk.columns:
        vals = chunk["complete_dose_not_given"].dropna()
        complete_dose_not_given_counts.update(vals.value_counts().to_dict())

    # Q8: dose_due vs dose_given comparison
    if "dose_due" in chunk.columns and "dose_given" in chunk.columns:
        # Convert to numeric, coerce errors
        dd = pd.to_numeric(chunk["dose_due"], errors="coerce")
        dg = pd.to_numeric(chunk["dose_given"], errors="coerce")
        mask = dd.notna() & dg.notna() & (dd > 0)
        both = mask.sum()
        dose_compare_total += both
        if both > 0:
            less = ((dg[mask] < dd[mask] * 0.99)).sum()  # 1% tolerance
            equal = ((dg[mask] >= dd[mask] * 0.99) & (dg[mask] <= dd[mask] * 1.01)).sum()
            more = ((dg[mask] > dd[mask] * 1.01)).sum()
            dose_less_given += less
            dose_equal += equal
            dose_more_given += more

            # Collect some examples where less was given
            if len(dose_compare_examples) < 20:
                less_mask = mask & (dg < dd * 0.99)
                sample = chunk.loc[less_mask, ["dose_due", "dose_due_unit", "dose_given", "dose_given_unit", "product_description"]].head(5)
                for _, row in sample.iterrows():
                    if len(dose_compare_examples) < 20:
                        dose_compare_examples.append(row.to_dict())

    if (i + 1) % 10 == 0:
        print(f"  ... processed {total_rows:,} rows so far")

print(f"\nTotal rows: {total_rows:,}\n")

# ---- Q1 ----
print("=" * 70)
print("Q1: dose_given and dose_given_unit completeness")
print("=" * 70)
for col in ["dose_given", "dose_given_unit"]:
    nn = non_null[col]
    pct = 100.0 * nn / total_rows
    print(f"  {col}: {nn:,} non-null / {total_rows:,} = {pct:.1f}%")

# ---- Q2 ----
print("\n" + "=" * 70)
print("Q2: Most common dose_given_unit values")
print("=" * 70)
for val, cnt in dose_given_unit_counts.most_common(25):
    pct = 100.0 * cnt / total_rows
    print(f"  {val!r:30s}  {cnt:>12,}  ({pct:.2f}%)")

# ---- Q3 ----
print("\n" + "=" * 70)
print("Q3: Route completeness and top values")
print("=" * 70)
nn = non_null["route"]
pct = 100.0 * nn / total_rows
print(f"  route non-null: {nn:,} / {total_rows:,} = {pct:.1f}%")
print("  Top route values:")
for val, cnt in route_counts.most_common(25):
    pct_v = 100.0 * cnt / total_rows
    print(f"    {val!r:30s}  {cnt:>12,}  ({pct_v:.2f}%)")

# ---- Q4 ----
print("\n" + "=" * 70)
print("Q4: administration_type completeness and values")
print("=" * 70)
nn = non_null["administration_type"]
pct = 100.0 * nn / total_rows
print(f"  administration_type non-null: {nn:,} / {total_rows:,} = {pct:.1f}%")
print("  Values:")
for val, cnt in admin_type_counts.most_common(20):
    pct_v = 100.0 * cnt / total_rows
    print(f"    {val!r:30s}  {cnt:>12,}  ({pct_v:.2f}%)")

# ---- Q5 ----
print("\n" + "=" * 70)
print("Q5: Top 20 product_description values")
print("=" * 70)
nn = non_null["product_description"]
pct = 100.0 * nn / total_rows
print(f"  product_description non-null: {nn:,} / {total_rows:,} = {pct:.1f}%")
print(f"  Unique values: ~{len(product_desc_counts):,}")
print("  Top 20:")
for val, cnt in product_desc_counts.most_common(20):
    pct_v = 100.0 * cnt / total_rows
    print(f"    {str(val)[:50]:50s}  {cnt:>12,}  ({pct_v:.2f}%)")

# ---- Q6 ----
print("\n" + "=" * 70)
print("Q6: infusion_rate completeness and units")
print("=" * 70)
nn_rate = non_null["infusion_rate"]
pct_rate = 100.0 * nn_rate / total_rows
print(f"  infusion_rate non-null: {nn_rate:,} / {total_rows:,} = {pct_rate:.1f}%")
nn_unit = non_null.get("infusion_rate_unit", 0)
pct_unit = 100.0 * nn_unit / total_rows
print(f"  infusion_rate_unit non-null: {nn_unit:,} / {total_rows:,} = {pct_unit:.1f}%")
print("  Common infusion_rate_unit values:")
for val, cnt in infusion_rate_unit_counts.most_common(15):
    pct_v = 100.0 * cnt / total_rows
    print(f"    {val!r:30s}  {cnt:>12,}  ({pct_v:.2f}%)")

# ---- Q7 ----
print("\n" + "=" * 70)
print("Q7: complete_dose_not_given values")
print("=" * 70)
nn = non_null["complete_dose_not_given"]
pct = 100.0 * nn / total_rows
print(f"  complete_dose_not_given non-null: {nn:,} / {total_rows:,} = {pct:.1f}%")
print("  Values:")
for val, cnt in complete_dose_not_given_counts.most_common(20):
    pct_v = 100.0 * cnt / total_rows
    print(f"    {val!r:30s}  {cnt:>12,}  ({pct_v:.2f}%)")

# ---- Q8 ----
print("\n" + "=" * 70)
print("Q8: dose_due vs dose_given comparison")
print("=" * 70)
nn_due = non_null["dose_due"]
nn_given = non_null["dose_given"]
print(f"  dose_due non-null:   {nn_due:,} ({100.0*nn_due/total_rows:.1f}%)")
print(f"  dose_given non-null: {nn_given:,} ({100.0*nn_given/total_rows:.1f}%)")
print(f"  Both numeric & dose_due>0: {dose_compare_total:,}")
if dose_compare_total > 0:
    print(f"  Less given than due (<99%):  {dose_less_given:,} ({100.0*dose_less_given/dose_compare_total:.1f}%)")
    print(f"  Approximately equal (+-1%):  {dose_equal:,} ({100.0*dose_equal/dose_compare_total:.1f}%)")
    print(f"  More given than due (>101%): {dose_more_given:,} ({100.0*dose_more_given/dose_compare_total:.1f}%)")
    print("\n  Example rows where less was given than ordered:")
    for ex in dose_compare_examples[:10]:
        print(f"    due={ex.get('dose_due')!r} {ex.get('dose_due_unit')!r}  "
              f"given={ex.get('dose_given')!r} {ex.get('dose_given_unit')!r}  "
              f"product={str(ex.get('product_description',''))[:40]}")

print("\nDone.")
