"""
Comprehensive data quality audit for states_actions.csv
(sepsis pipeline output after column-scrambling bug fix)
"""
import pandas as pd
import numpy as np
import sys

CSV = "data/interim/sepsis/intermediates/states_actions.csv"

print("=" * 80)
print("DATA QUALITY AUDIT: states_actions.csv")
print("=" * 80)

df = pd.read_csv(CSV)

# ============================================================
# 1. SHAPE, UNIQUE PATIENTS, BLOCS PER PATIENT
# ============================================================
print("\n" + "=" * 80)
print("1. SHAPE & PATIENT SUMMARY")
print("=" * 80)
n_rows, n_cols = df.shape
n_patients = df["icustayid"].nunique()
blocs = df.groupby("icustayid")["bloc"].count()
print(f"Rows:           {n_rows:,}")
print(f"Columns:        {n_cols}")
print(f"Unique patients (icustayid): {n_patients:,}")
print(f"\nBlocs per patient:")
print(f"  min:    {blocs.min()}")
print(f"  max:    {blocs.max()}")
print(f"  mean:   {blocs.mean():.1f}")
print(f"  median: {blocs.median():.0f}")
print(f"  p5:     {blocs.quantile(0.05):.0f}")
print(f"  p25:    {blocs.quantile(0.25):.0f}")
print(f"  p75:    {blocs.quantile(0.75):.0f}")
print(f"  p95:    {blocs.quantile(0.95):.0f}")

# ============================================================
# 2. PER-COLUMN STATS
# ============================================================
print("\n" + "=" * 80)
print("2. PER-COLUMN STATISTICS")
print("=" * 80)
print(f"{'Column':<35} {'dtype':<10} {'non-null':>10} {'%miss':>7} {'min':>12} {'max':>12} {'median':>12}")
print("-" * 100)

cols_100_nan = []
for col in df.columns:
    nn = df[col].notna().sum()
    pct_miss = (1 - nn / n_rows) * 100
    dt = str(df[col].dtype)
    if nn == 0:
        cols_100_nan.append(col)
        print(f"{col:<35} {dt:<10} {nn:>10,} {pct_miss:>6.1f}% {'N/A':>12} {'N/A':>12} {'N/A':>12}")
    elif np.issubdtype(df[col].dtype, np.number):
        mn = df[col].min()
        mx = df[col].max()
        med = df[col].median()
        print(f"{col:<35} {dt:<10} {nn:>10,} {pct_miss:>6.1f}% {mn:>12.4g} {mx:>12.4g} {med:>12.4g}")
    else:
        print(f"{col:<35} {dt:<10} {nn:>10,} {pct_miss:>6.1f}% {'(non-num)':>12} {'(non-num)':>12} {'(non-num)':>12}")

# ============================================================
# 3. KEY CLINICAL FEATURES - RANGE CHECK
# ============================================================
print("\n" + "=" * 80)
print("3. KEY CLINICAL FEATURES - RANGE CHECK")
print("=" * 80)

expected = {
    # (col_name, lo_normal, hi_normal, lo_plausible, hi_plausible)
    "HR":                (60, 100,   0, 300),
    "SysBP":             (90, 140,   0, 300),
    "MeanBP":            (70, 105,   0, 250),
    "DiaBP":             (60, 90,    0, 200),
    "SpO2":              (95, 100,   0, 100),
    "Temp_C":            (36.5, 37.5, 25, 45),
    "RR":                (12, 20,    0, 60),
    "GCS":               (3, 15,     3, 15),
    "RASS":              (-5, 4,    -5, 4),
    "Potassium":         (3.5, 5.0,  1.0, 12.0),
    "Sodium":            (136, 145,  100, 180),
    "Chloride":          (96, 106,   60, 150),
    "Glucose":           (70, 180,   10, 1500),
    "BUN":               (7, 20,     0, 200),
    "Creatinine":        (0.6, 1.2,  0, 30),
    "Magnesium":         (1.7, 2.2,  0.5, 10),
    "Calcium":           (8.4, 10.2, 4, 20),
    "Hb":                (12, 17.5,  2, 25),
    "WBC_count":         (4.5, 11,   0, 200),
    "Platelets_count":   (150, 400,  0, 2000),
    "Arterial_lactate":  (0.5, 2.0,  0, 30),
    "Arterial_pH":       (7.35, 7.45, 6.5, 8.0),
    "INR":               (0.8, 1.2,  0.5, 20),
    "PTT":               (25, 35,    10, 200),
    "PT":                (11, 13.5,  5, 150),
    "paO2":              (80, 100,   20, 700),
    "paCO2":             (35, 45,    10, 150),
    "HCO3":              (22, 28,    5, 60),
    "Total_bili":        (0.1, 1.2,  0, 60),
    "Albumin":           (3.5, 5.5,  0.5, 7),
}

print(f"{'Feature':<22} {'N':>8} {'%miss':>7} {'min':>10} {'p1':>10} {'p5':>10} {'median':>10} {'p95':>10} {'p99':>10} {'max':>10} {'%normal':>8} {'%implaus':>8} FLAG")
print("-" * 150)

for col, (lo_n, hi_n, lo_p, hi_p) in expected.items():
    if col not in df.columns:
        print(f"{col:<22} *** COLUMN NOT FOUND ***")
        continue
    s = df[col].dropna()
    n = len(s)
    pct_miss = (1 - n / n_rows) * 100
    if n == 0:
        print(f"{col:<22} {0:>8} {pct_miss:>6.1f}% {'ALL NaN':>10}")
        continue
    mn, mx = s.min(), s.max()
    p1, p5, med, p95, p99 = s.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    pct_normal = ((s >= lo_n) & (s <= hi_n)).mean() * 100
    pct_implaus = ((s < lo_p) | (s > hi_p)).mean() * 100

    flags = []
    if pct_miss > 50:
        flags.append("HIGH_MISS")
    if pct_implaus > 1:
        flags.append(f"IMPLAUSIBLE({pct_implaus:.1f}%)")
    if mn < lo_p:
        flags.append(f"BELOW_FLOOR(min={mn:.2f})")
    if mx > hi_p:
        flags.append(f"ABOVE_CEIL(max={mx:.2f})")
    flag_str = " | ".join(flags) if flags else "OK"

    print(f"{col:<22} {n:>8,} {pct_miss:>6.1f}% {mn:>10.3f} {p1:>10.3f} {p5:>10.3f} {med:>10.3f} {p95:>10.3f} {p99:>10.3f} {mx:>10.3f} {pct_normal:>7.1f}% {pct_implaus:>7.1f}% {flag_str}")

# ============================================================
# 4. ACTION COLUMNS
# ============================================================
print("\n" + "=" * 80)
print("4. ACTION COLUMNS")
print("=" * 80)

action_cols = ["input_step", "output_step", "input_total", "output_total",
               "max_dose_vaso", "median_dose_vaso", "mechvent", "cumulated_balance"]
for col in action_cols:
    if col not in df.columns:
        print(f"  {col}: NOT FOUND")
        continue
    s = df[col]
    nn = s.notna().sum()
    pct_miss = (1 - nn / n_rows) * 100
    pct_zero = (s == 0).mean() * 100
    pct_neg = (s < 0).mean() * 100
    print(f"\n  {col}:")
    print(f"    non-null: {nn:,} ({pct_miss:.1f}% missing)")
    print(f"    min={s.min():.4g}  max={s.max():.4g}  median={s.median():.4g}  mean={s.mean():.4g}")
    print(f"    %zero={pct_zero:.1f}%  %negative={pct_neg:.1f}%")
    print(f"    p5={s.quantile(0.05):.4g}  p25={s.quantile(0.25):.4g}  p75={s.quantile(0.75):.4g}  p95={s.quantile(0.95):.4g}")
    if col in ["max_dose_vaso", "median_dose_vaso"]:
        pct_pos = (s > 0).mean() * 100
        print(f"    %positive (on vasopressors)={pct_pos:.1f}%")
    if pct_neg > 0 and col in ["input_step", "input_total", "max_dose_vaso", "median_dose_vaso"]:
        print(f"    *** FLAG: negative values found! ***")

# ============================================================
# 5. DEMOGRAPHICS
# ============================================================
print("\n" + "=" * 80)
print("5. DEMOGRAPHICS (first row per patient)")
print("=" * 80)

demo_cols = ["gender", "age", "elixhauser", "re_admission", "morta_90",
             "died_in_hosp", "died_within_48h_of_out_time"]
first = df.groupby("icustayid").first()

for col in demo_cols:
    if col not in first.columns:
        print(f"  {col}: NOT FOUND")
        continue
    s = first[col].dropna()
    print(f"\n  {col} (N={len(s):,}, {(1-len(s)/n_patients)*100:.1f}% missing):")
    if s.nunique() <= 10:
        vc = s.value_counts().sort_index()
        for v, c in vc.items():
            print(f"    {v}: {c:,} ({c/len(s)*100:.1f}%)")
    else:
        print(f"    min={s.min():.4g}  max={s.max():.4g}  mean={s.mean():.4g}  median={s.median():.4g}")
        print(f"    p5={s.quantile(0.05):.4g}  p25={s.quantile(0.25):.4g}  p75={s.quantile(0.75):.4g}  p95={s.quantile(0.95):.4g}")

# Age sanity
if "age" in first.columns:
    age = first["age"].dropna()
    print(f"\n  Age sanity:")
    print(f"    <18: {(age < 18).sum()}, 18-65: {((age >= 18) & (age <= 65)).sum()}, 65-90: {((age > 65) & (age <= 90)).sum()}, >90: {(age > 90).sum()}")

# ============================================================
# 6. 100% NaN COLUMNS
# ============================================================
print("\n" + "=" * 80)
print("6. COLUMNS THAT ARE 100% NaN")
print("=" * 80)
if cols_100_nan:
    for c in cols_100_nan:
        print(f"  *** {c} ***")
else:
    print("  None found.")

# High missingness
print("\n  Columns with >80% missing:")
for col in df.columns:
    pct_miss = df[col].isna().mean() * 100
    if pct_miss > 80:
        print(f"    {col}: {pct_miss:.1f}%")

# ============================================================
# 7. SUSPICIOUS DISTRIBUTIONS
# ============================================================
print("\n" + "=" * 80)
print("7. SUSPICIOUS DISTRIBUTIONS")
print("=" * 80)

# Mostly zeros (>90%)
print("\n  Columns >90% zero:")
for col in df.select_dtypes(include=[np.number]).columns:
    if col in ["bloc", "icustayid", "timestep"]:
        continue
    pct_zero = (df[col] == 0).mean() * 100
    if pct_zero > 90:
        print(f"    {col}: {pct_zero:.1f}% zero")

# Very high max/median ratio (heavy tails)
print("\n  Columns with max > 100x median (heavy tails):")
for col in df.select_dtypes(include=[np.number]).columns:
    if col in ["bloc", "icustayid", "timestep"]:
        continue
    s = df[col].dropna()
    if len(s) == 0:
        continue
    med = s.median()
    mx = s.max()
    if med > 0 and mx / med > 100:
        print(f"    {col}: median={med:.4g}, max={mx:.4g}, ratio={mx/med:.0f}x")

# Constant columns
print("\n  Constant columns (std=0):")
for col in df.select_dtypes(include=[np.number]).columns:
    s = df[col].dropna()
    if len(s) > 0 and s.std() == 0:
        print(f"    {col}: constant value = {s.iloc[0]}")

# ============================================================
# 8. DUPLICATES
# ============================================================
print("\n" + "=" * 80)
print("8. DUPLICATE CHECK")
print("=" * 80)

n_dup_rows = df.duplicated().sum()
print(f"  Fully duplicate rows: {n_dup_rows:,}")

dup_bloc = df.duplicated(subset=["icustayid", "bloc"]).sum()
print(f"  Duplicate (icustayid, bloc) pairs: {dup_bloc:,}")

dup_ts = df.duplicated(subset=["icustayid", "timestep"]).sum()
print(f"  Duplicate (icustayid, timestep) pairs: {dup_ts:,}")

if dup_bloc > 0:
    print(f"  *** FLAG: duplicate (icustayid, bloc) pairs exist! ***")
    ex = df[df.duplicated(subset=["icustayid", "bloc"], keep=False)].head(6)[["icustayid", "bloc", "timestep"]]
    print(ex.to_string())

# ============================================================
# 9. DATA TYPE ISSUES
# ============================================================
print("\n" + "=" * 80)
print("9. DATA TYPE CHECK")
print("=" * 80)

obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
if obj_cols:
    print(f"  Object/string columns: {obj_cols}")
    for col in obj_cols:
        print(f"    {col}: unique values = {df[col].nunique()}, examples = {df[col].dropna().unique()[:5]}")
else:
    print("  No object/string columns found. All numeric. Good.")

# Check int columns that might have float issues
print("\n  Integer columns:")
int_cols = df.select_dtypes(include=["int64", "int32"]).columns.tolist()
print(f"    {int_cols}")

print("\n  Float columns:")
float_cols = df.select_dtypes(include=["float64", "float32"]).columns.tolist()
print(f"    {float_cols}")

# ============================================================
# 10. COMPARISON TO TYPICAL AI-CLINICIAN
# ============================================================
print("\n" + "=" * 80)
print("10. COMPARISON TO TYPICAL AI-CLINICIAN OUTPUT")
print("=" * 80)
print(f"  This dataset:        {n_patients:,} patients, {n_rows:,} rows, {n_cols} columns")
print(f"  Typical AI-Clinician: ~17,000 patients, ~180,000 rows, ~50 columns")
print(f"  Patient ratio:       {n_patients/17000:.2f}x")
print(f"  Row ratio:           {n_rows/180000:.2f}x")
print(f"  Row/patient ratio:   {n_rows/n_patients:.1f} (typical ~10-11)")

if n_patients < 10000:
    print("  *** FLAG: Fewer patients than expected. Check cohort inclusion criteria. ***")
elif n_patients > 25000:
    print("  *** FLAG: More patients than expected. Check if cohort is too broad. ***")
else:
    print("  Patient count in reasonable range.")

if n_cols > 80:
    print(f"  *** NOTE: {n_cols} columns is more than standard 47. Extra columns may be intermediate features. ***")

# ============================================================
# SUMMARY OF FLAGS
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY OF FLAGS")
print("=" * 80)

flags_summary = []
if cols_100_nan:
    flags_summary.append(f"100% NaN columns: {cols_100_nan}")
if n_dup_rows > 0:
    flags_summary.append(f"Duplicate rows: {n_dup_rows}")
if dup_bloc > 0:
    flags_summary.append(f"Duplicate (icustayid, bloc) pairs: {dup_bloc}")
if obj_cols:
    flags_summary.append(f"Object columns that may need conversion: {obj_cols}")

# Check key features present
missing_key = [c for c in ["HR", "SysBP", "MeanBP", "SpO2", "GCS", "Potassium",
                            "Sodium", "Creatinine", "Arterial_lactate", "WBC_count",
                            "Platelets_count", "Arterial_pH"] if c not in df.columns]
if missing_key:
    flags_summary.append(f"Missing key sepsisRL features: {missing_key}")

if flags_summary:
    for f in flags_summary:
        print(f"  ** {f}")
else:
    print("  No critical flags raised.")

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)
