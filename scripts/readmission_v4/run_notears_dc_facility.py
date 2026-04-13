"""NOTEARS causal discovery — dc_facility action only.

Single binary action: dc_facility (home=0, facility=1).
Facility = SNF + rehab + psych + hospice + other_facility.
Rows with dc_location='other' or 'died' are dropped.

Confounders: 23 selected by LightGBM double selection
  (predicts dc_facility OR readmit_30d from all Tier 0-2 candidates).

Same NOTEARS setup as run_notears.py:
  - tier mask enforced post-hoc (action/outcome cannot cause confounders)
  - lambda sweep: 0.05, 0.01, 0.005
  - outputs: edges, action_edges, lambda_sweep, run_log

Usage:
    # Smoke test (5k rows)
    python scripts/readmission_v4/run_notears_dc_facility.py --sample-only

    # Full run
    python scripts/readmission_v4/run_notears_dc_facility.py
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
ACTION_COL  = "dc_facility"

FACILITY_CATS = ["snf", "rehab", "psych", "hospice", "other_facility"]

# 23 confounders from LightGBM double selection
CONFOUNDERS = [
    "admission_type",
    "age_at_admit",
    "charlson_score",
    "dc_lab_albumin",
    "dc_lab_anion_gap",
    "dc_lab_bicarbonate",
    "dc_lab_bun",
    "dc_lab_calcium",
    "dc_lab_creatinine",
    "dc_lab_hemoglobin",
    "dc_lab_platelets",
    "dc_lab_potassium",
    "dc_lab_sodium",
    "dc_lab_wbc",
    "drg_severity",
    "ed_dwell_hours",
    "first_service",
    "icu_days",
    "insurance_cat",
    "los_days",
    "n_abnormal_dc_labs",
    "n_cultures",
    "race_cat",
]

CATEGORICAL_COLS = ["admission_type", "first_service", "insurance_cat", "race_cat"]


# ---------------------------------------------------------------------------
# Tier mask
# ---------------------------------------------------------------------------

def build_tier_mask(var_names, confounders, actions, outcome):
    """Post-hoc mask: confounders -> action -> outcome only."""
    confounder_set = set(confounders)
    action_set     = set(actions)

    def tier(v):
        if v in confounder_set: return 0
        if v in action_set:     return 1
        return 2  # outcome

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
        default=str(PROJECT_ROOT / "reports" / "readmission_v4" / "notears_dc_facility"),
    )
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=[0.05, 0.01, 0.005],
    )
    parser.add_argument("--w-threshold", type=float, default=0.01)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--sample-n",    type=int, default=5_000)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if args.sample_only:
        report_dir = report_dir.parent / "notears_dc_facility_smoke"
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
    log.info("V4 NOTEARS -- dc_facility action")
    log.info("=" * 70)
    log.info("Action: %s | Confounders: %d | lambdas: %s",
             ACTION_COL, len(CONFOUNDERS), args.lambdas)
    log.info("sample_only=%s", args.sample_only)

    # ── Load and encode ──────────────────────────────────────────────
    log.info("Loading: %s", args.csv)
    df = pd.read_csv(args.csv, low_memory=False)
    train = df[df["split"] == "train"].copy()
    log.info("Train rows (all): %d", len(train))

    # Encode dc_facility, drop 'other' and 'died'
    keep = ["home"] + FACILITY_CATS
    train = train[train["dc_location"].isin(keep)].copy()
    train[ACTION_COL] = (train["dc_location"] != "home").astype(int)
    log.info("After dropping 'other'/'died': %d rows", len(train))
    log.info("dc_facility=1 (facility): %.1f%%", train[ACTION_COL].mean() * 100)

    if args.sample_only:
        train = train.sample(n=min(args.sample_n, len(train)), random_state=args.seed)
        log.info("Sampled %d rows (seed=%d)", len(train), args.seed)

    var_names = CONFOUNDERS + [ACTION_COL, OUTCOME_COL]

    # Drop zero-variance columns
    zero_var = [c for c in var_names if train[c].nunique(dropna=True) <= 1]
    if zero_var:
        log.info("Dropping zero-variance columns: %s", zero_var)
        var_names = [v for v in var_names if v not in zero_var]

    log.info("Preprocessing (label-encode, impute, standardize) ...")
    X = preprocess(train, var_names)
    log.info("Data matrix: %d rows x %d columns", X.shape[0], X.shape[1])

    actions    = [ACTION_COL]
    tier_mask  = build_tier_mask(var_names, CONFOUNDERS, actions, OUTCOME_COL)
    n_forbidden = int((tier_mask == 0).sum())
    log.info("Tier mask: %d entries forbidden", n_forbidden)

    from notears.linear import notears_linear

    all_results = {}
    total_t0 = time.time()

    for i, lam in enumerate(sorted(args.lambdas, reverse=True)):
        log.info("")
        log.info("=" * 70)
        log.info("LAMBDA = %.4f  (%d/%d)", lam, i + 1, len(args.lambdas))
        log.info("=" * 70)
        log.info("Starting NOTEARS optimisation ...")

        t0 = time.time()
        W_raw = notears_linear(X, lambda1=lam, loss_type="l2", w_threshold=0.0)
        elapsed = time.time() - t0

        W = W_raw * tier_mask

        n_raw     = int((np.abs(W_raw) >= args.w_threshold).sum())
        n_masked  = int((np.abs(W)     >= args.w_threshold).sum())
        n_removed = n_raw - n_masked

        log.info("NOTEARS done in %.1f seconds (%.1f min)", elapsed, elapsed / 60)
        log.info("  Edges (|w| >= %.3f): %d total (%d removed by tier mask)",
                 args.w_threshold, n_masked, n_removed)

        edge_df        = extract_edges(W, var_names, w_threshold=args.w_threshold)
        action_edge_df = extract_action_edges(edge_df, actions, OUTCOME_COL)
        log.info("  dc_facility -> readmit_30d edges: %d", len(action_edge_df))

        lam_str = f"{lam:.4f}".replace(".", "p")
        edge_df.to_csv(report_dir / f"edges_lambda_{lam_str}.csv", index=False)
        action_edge_df.to_csv(report_dir / f"action_edges_lambda_{lam_str}.csv", index=False)

        if len(action_edge_df) > 0:
            for _, row in action_edge_df.iterrows():
                log.info("    %-35s --> %-15s  w=%.4f",
                         row["source"], row["target"], row["weight"])
        else:
            log.info("  No dc_facility -> readmit_30d edge at this lambda.")

        all_results[lam] = {
            "edge_df": edge_df,
            "action_edge_df": action_edge_df,
            "elapsed": elapsed,
        }

    total_elapsed = time.time() - total_t0
    log.info("")
    log.info("All lambdas completed in %.1f seconds (%.1f min)",
             total_elapsed, total_elapsed / 60)

    # ── Lambda sensitivity table ─────────────────────────────────────
    all_pairs = set()
    for lam in all_results:
        for _, row in all_results[lam]["action_edge_df"].iterrows():
            all_pairs.add((row["source"], row["target"]))

    sweep_rows = []
    for src, tgt in sorted(all_pairs):
        row_data = {"source": src, "target": tgt}
        for lam in sorted(args.lambdas, reverse=True):
            adf   = all_results[lam]["action_edge_df"]
            match = adf[((adf["source"] == src) & (adf["target"] == tgt)) |
                        ((adf["source"] == tgt) & (adf["target"] == src))]
            row_data[f"lam={lam}"] = f"{match.iloc[0]['weight']:.4f}" if len(match) > 0 else "-"
        sweep_rows.append(row_data)

    sweep_df = pd.DataFrame(sweep_rows) if sweep_rows else pd.DataFrame()
    sweep_df.to_csv(report_dir / "lambda_sweep.csv", index=False)
    log.info("")
    log.info("=" * 70)
    log.info("LAMBDA SENSITIVITY TABLE (dc_facility -> readmit_30d)")
    log.info("=" * 70)
    if len(sweep_df) > 0:
        log.info("\n%s", sweep_df.to_string(index=False))
    else:
        log.info("  (no dc_facility -> readmit_30d edge at any lambda)")

    log.info("")
    log.info("V4 NOTEARS dc_facility complete. Results in: %s", report_dir)
    log.info("Total runtime: %.1f minutes", total_elapsed / 60)


if __name__ == "__main__":
    main()
