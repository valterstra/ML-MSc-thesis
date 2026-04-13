"""NOTEARS causal discovery — all 18 actions combined.

Actions (18):
  dc_facility (binary: home=0, facility=1)
  los_days, discharge_service
  8 consults (consult_addiction dropped - near-zero variance)
  8 discharge meds (dc_med_insulin dropped - zero variance)

Confounders (34): selected by LightGBM double selection across all 19 action
  targets + readmit_30d.

Rows: ~231k train (after dropping dc_location='other'/'died').

Lambda: 0.005 (same as first NOTEARS run).
Tier mask enforced post-hoc.

Usage:
    # Smoke test (5k rows)
    python scripts/readmission_v4/run_notears_all_actions.py --sample-only

    # Full run
    python scripts/readmission_v4/run_notears_all_actions.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

log = logging.getLogger(__name__)

OUTCOME_COL = "readmit_30d"

FACILITY_CATS = ["snf", "rehab", "psych", "hospice", "other_facility"]

ACTIONS = [
    "dc_facility",
    "los_days",
    "discharge_service",
    "consult_pt", "consult_ot", "consult_social_work", "consult_followup",
    "consult_palliative", "consult_diabetes_education", "consult_speech",
    "dc_med_statin", "dc_med_antihypertensive", "dc_med_antiplatelet",
    "dc_med_anticoagulant", "dc_med_diuretic", "dc_med_antibiotic",
    "dc_med_opiate", "dc_med_steroid",
]

CONFOUNDERS = [
    "age_at_admit", "dc_lab_bun", "dc_lab_calcium", "dc_lab_hemoglobin",
    "dc_lab_platelets", "dc_lab_wbc", "ed_dwell_hours", "first_service",
    "icu_days", "n_cultures", "charlson_score", "dc_lab_bicarbonate",
    "dc_lab_creatinine", "dc_lab_potassium", "dc_lab_sodium", "admission_type",
    "dc_lab_anion_gap", "drg_severity", "dc_lab_albumin", "n_transfers",
    "n_abnormal_dc_labs", "race_cat", "drg_mortality", "flag_atrial_fibrillation",
    "n_distinct_services", "flag_cancer", "n_distinct_careunits",
    "flag_malnutrition", "flag_schizophrenia", "insurance_cat",
    "rx_antibiotic", "rx_anticoagulant", "rx_diuretic", "rx_steroid",
]

CATEGORICAL_COLS = [
    "admission_type", "first_service", "discharge_service",
    "insurance_cat", "race_cat",
]


# ---------------------------------------------------------------------------
# Tier mask
# ---------------------------------------------------------------------------

def build_tier_mask(var_names, confounders, actions, outcome):
    confounder_set = set(confounders)
    action_set     = set(actions)

    def tier(v):
        if v in confounder_set: return 0
        if v in action_set:     return 1
        return 2

    n = len(var_names)
    mask = np.ones((n, n))
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if tier(vj) >= tier(vi):
                mask[i, j] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------

def extract_edges(W, var_names, w_threshold):
    rows = []
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if i == j:
                continue
            w = W[i, j]
            if abs(w) >= w_threshold:
                rows.append({
                    "source":     vj,
                    "target":     vi,
                    "weight":     round(float(w), 6),
                    "abs_weight": round(abs(float(w)), 6),
                })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "target", "weight", "abs_weight"]
    )
    return df.sort_values("abs_weight", ascending=False).reset_index(drop=True)


def extract_action_edges(edge_df, actions, outcome):
    action_set = set(actions)
    mask = (
        (edge_df["source"].isin(action_set) & (edge_df["target"] == outcome))
        | ((edge_df["source"] == outcome) & edge_df["target"].isin(action_set))
    )
    return edge_df[mask].copy()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df, var_names):
    X = df[var_names].copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category").cat.codes.astype(float)
            X.loc[X[col] == -1, col] = np.nan
    for col in X.columns:
        if X[col].isna().any():
            if X[col].nunique(dropna=True) <= 2:
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(X[col].median())
    scaler = StandardScaler()
    return scaler.fit_transform(X.values.astype(np.float64))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "readmission_v4_admissions.csv"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "readmission_v4" / "notears_all_actions"),
    )
    parser.add_argument("--lam",         type=float, default=0.005)
    parser.add_argument("--w-threshold", type=float, default=0.01)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--sample-n",    type=int,   default=5_000)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if args.sample_only:
        report_dir = report_dir.parent / "notears_all_actions_smoke"
    report_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(report_dir / "run_log.txt"), mode="w", encoding="utf-8"),
        ],
    )

    log.info("=" * 70)
    log.info("V4 NOTEARS -- all 18 actions combined")
    log.info("=" * 70)
    log.info("Actions: %d | Confounders: %d | lambda: %.4f | sample_only: %s",
             len(ACTIONS), len(CONFOUNDERS), args.lam, args.sample_only)

    # ── Load and encode ──────────────────────────────────────────────
    log.info("Loading: %s", args.csv)
    df = pd.read_csv(args.csv, low_memory=False)
    train = df[df["split"] == "train"].copy()
    log.info("Train rows (all): %d", len(train))

    keep = ["home"] + FACILITY_CATS
    train = train[train["dc_location"].isin(keep)].copy()
    train["dc_facility"] = (train["dc_location"] != "home").astype(int)
    log.info("After dropping 'other'/'died': %d rows", len(train))
    log.info("dc_facility=1: %.1f%%", train["dc_facility"].mean() * 100)

    if args.sample_only:
        train = train.sample(n=min(args.sample_n, len(train)), random_state=args.seed)
        log.info("Sampled %d rows (seed=%d)", len(train), args.seed)

    var_names = CONFOUNDERS + ACTIONS + [OUTCOME_COL]

    zero_var = [c for c in var_names if train[c].nunique(dropna=True) <= 1]
    if zero_var:
        log.info("Dropping zero-variance columns: %s", zero_var)
        var_names = [v for v in var_names if v not in zero_var]
        actions_clean = [a for a in ACTIONS if a not in zero_var]
    else:
        actions_clean = ACTIONS

    log.info("Preprocessing (label-encode, impute, standardize) ...")
    X = preprocess(train, var_names)
    log.info("Data matrix: %d rows x %d columns", X.shape[0], X.shape[1])

    tier_mask   = build_tier_mask(var_names, CONFOUNDERS, actions_clean, OUTCOME_COL)
    n_forbidden = int((tier_mask == 0).sum())
    log.info("Tier mask: %d entries forbidden", n_forbidden)

    from notears.linear import notears_linear

    log.info("")
    log.info("Starting NOTEARS (lambda=%.4f) ...", args.lam)
    t0    = time.time()
    W_raw = notears_linear(X, lambda1=args.lam, loss_type="l2", w_threshold=0.0)
    elapsed = time.time() - t0
    log.info("NOTEARS done in %.1f seconds (%.1f min)", elapsed, elapsed / 60)

    W         = W_raw * tier_mask
    n_raw     = int((np.abs(W_raw) >= args.w_threshold).sum())
    n_masked  = int((np.abs(W)     >= args.w_threshold).sum())
    log.info("Edges (|w| >= %.3f): %d total (%d removed by tier mask)",
             args.w_threshold, n_masked, n_raw - n_masked)

    # Masked edges (tier-constrained)
    edge_df        = extract_edges(W, var_names, w_threshold=args.w_threshold)
    action_edge_df = extract_action_edges(edge_df, actions_clean, OUTCOME_COL)

    # Unmasked edges (raw NOTEARS, no tier constraint)
    edge_df_raw        = extract_edges(W_raw, var_names, w_threshold=args.w_threshold)
    action_edge_df_raw = extract_action_edges(edge_df_raw, actions_clean, OUTCOME_COL)

    edge_df.to_csv(report_dir / "edges.csv", index=False)
    action_edge_df.to_csv(report_dir / "action_edges.csv", index=False)
    edge_df_raw.to_csv(report_dir / "edges_unmasked.csv", index=False)
    action_edge_df_raw.to_csv(report_dir / "action_edges_unmasked.csv", index=False)

    log.info("")
    log.info("=" * 70)
    log.info("ACTION <-> readmit_30d EDGES -- UNMASKED (raw NOTEARS, %d)", len(action_edge_df_raw))
    log.info("=" * 70)
    if len(action_edge_df_raw) > 0:
        for _, row in action_edge_df_raw.iterrows():
            log.info("  %-35s --> %-35s  w=%.4f", row["source"], row["target"], row["weight"])
    else:
        log.info("  (none)")

    log.info("")
    log.info("=" * 70)
    log.info("ACTION -> readmit_30d EDGES -- MASKED (tier-constrained, %d)", len(action_edge_df))
    log.info("=" * 70)
    if len(action_edge_df) > 0:
        for _, row in action_edge_df.iterrows():
            log.info("  %-35s --> %-35s  w=%.4f", row["source"], row["target"], row["weight"])
    else:
        log.info("  (none)")

    log.info("")
    log.info("V4 NOTEARS all_actions complete. Results in: %s", report_dir)
    log.info("Total runtime: %.1f minutes", elapsed / 60)


if __name__ == "__main__":
    main()
