"""NOTEARS causal discovery on V4 readmission dataset.

NOTEARS reformulates DAG learning as continuous optimisation:
  minimize  ||X - XW||^2  +  lambda * ||W||_1
  subject to  tr(e^(W o W)) - d = 0   (acyclicity)

Tier background knowledge is enforced POST-HOC by zeroing out edges
that violate the causal ordering: confounders -> actions -> outcome.
NOTEARS has no built-in background knowledge support.

Lambda sweep rationale:
  0.05  -- high sparsity, only strongest edges survive
  0.01  -- moderate, main edges appear
  0.005 -- lenient, borderline edges (may include noise)

Outputs (--report-dir):
  edges_lambda_{X}.csv        -- all edges per lambda
  action_edges_lambda_{X}.csv -- action <-> readmit_30d edges per lambda
  lambda_sweep.csv            -- which action edges appear at each lambda
  run_log.txt

Usage:
    # Smoke test (5k rows)
    python scripts/readmission_v4/run_notears.py --sample-only

    # Full run (all train rows)
    python scripts/readmission_v4/run_notears.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUTCOME_COL = "readmit_30d"

CATEGORICAL_COLS = [
    "gender", "insurance_cat", "race_cat", "dc_location",
    "admission_type", "first_service", "discharge_service",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier mask — enforce confounder -> action -> outcome ordering
# ---------------------------------------------------------------------------

def build_tier_mask(var_names, confounders, actions, outcome):
    """Post-hoc mask for W matrix enforcing tier ordering.

    W[i, j] = effect of var_names[j] on var_names[i]  (j -> i).

    Forbidden:
      - action    -> confounder   (tier 1 cannot cause tier 0)
      - outcome   -> confounder   (tier 2 cannot cause tier 0)
      - outcome   -> action       (tier 2 cannot cause tier 1)
    """
    confounder_set = set(confounders)
    action_set     = set(actions)

    def tier(v):
        if v in confounder_set:
            return 0
        if v in action_set:
            return 1
        return 2  # outcome

    n = len(var_names)
    mask = np.ones((n, n))
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if tier(vj) >= tier(vi):  # j cannot cause i if j is same/later tier
                mask[i, j] = 0.0
    return mask


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------

def extract_edges(W, var_names, w_threshold):
    """Extract edges where |W[i,j]| >= w_threshold. W[i,j] = j -> i."""
    rows = []
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if i == j:
                continue
            w = W[i, j]
            if abs(w) >= w_threshold:
                rows.append({
                    "source": vj,
                    "target": vi,
                    "weight": round(float(w), 6),
                    "abs_weight": round(abs(float(w)), 6),
                })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "target", "weight", "abs_weight"]
    )
    return df.sort_values("abs_weight", ascending=False).reset_index(drop=True)


def extract_action_edges(edge_df, actions, outcome):
    """Filter to action -> readmit_30d edges (and any action <-> action edges)."""
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
    """Label-encode categoricals, impute missing, standardize."""
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NOTEARS causal discovery on V4 readmission dataset."
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "readmission_v4_admissions.csv"),
    )
    parser.add_argument(
        "--varsel-json",
        default=str(PROJECT_ROOT / "reports" / "readmission_v4" / "variable_selection.json"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "readmission_v4" / "notears"),
    )
    parser.add_argument(
        "--max-confounders", type=int, default=18,
        help="Keep only top-N confounders by selection frequency (default: 18)",
    )
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=[0.05, 0.01, 0.005],
        help="L1 penalty values to sweep (default: 0.05 0.01 0.005)",
    )
    parser.add_argument(
        "--w-threshold", type=float, default=0.01,
        help="Min |weight| to report an edge (default: 0.01)",
    )
    parser.add_argument(
        "--sample-only", action="store_true",
        help="Smoke test: run on 5k rows only",
    )
    parser.add_argument("--sample-n", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if args.sample_only:
        report_dir = report_dir.parent / "notears_smoke"
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
    log.info("V4 NOTEARS Causal Discovery")
    log.info("=" * 70)
    log.info("max_confounders=%s | lambdas=%s | w_threshold=%.3f | sample_only=%s",
             args.max_confounders, args.lambdas, args.w_threshold, args.sample_only)

    # ── Load variable selection ──────────────────────────────────────
    log.info("Loading variable selection: %s", args.varsel_json)
    with open(args.varsel_json) as f:
        varsel = json.load(f)

    confounders = varsel["selected_confounders"]
    if args.max_confounders is not None:
        confounders = confounders[:args.max_confounders]
        log.info("Truncated confounders to top %d by selection frequency", args.max_confounders)
    actions = varsel["action_cols"]
    outcome = varsel["outcome_col"]
    var_names = confounders + actions + [outcome]

    log.info("Confounders: %d | Actions: %d | Outcome: %s",
             len(confounders), len(actions), outcome)
    log.info("Total variables: %d", len(var_names))

    # ── Load data ────────────────────────────────────────────────────
    log.info("Loading: %s", args.csv)
    df = pd.read_csv(args.csv, low_memory=False)
    train = df[df["split"] == "train"].copy()
    log.info("Train rows: %d", len(train))

    if args.sample_only or args.sample_n < len(train):
        n = args.sample_n
        train = train.sample(n=min(n, len(train)), random_state=args.seed)
        log.info("Sampled %d rows (seed=%d)", len(train), args.seed)

    # ── Drop zero-variance columns ────────────────────────────────────
    zero_var = [c for c in var_names if c != outcome and train[c].nunique(dropna=True) <= 1]
    if zero_var:
        log.info("Dropping zero-variance columns: %s", zero_var)
        var_names = [v for v in var_names if v not in zero_var]
        actions   = [a for a in actions   if a not in zero_var]

    # ── Preprocess ───────────────────────────────────────────────────
    log.info("Preprocessing (label-encode, impute, standardize) ...")
    X = preprocess(train, var_names)
    log.info("Data matrix: %d rows x %d columns", X.shape[0], X.shape[1])

    # ── Tier mask ────────────────────────────────────────────────────
    tier_mask = build_tier_mask(var_names, confounders, actions, outcome)
    n_forbidden = int((tier_mask == 0).sum())
    log.info("Tier mask: %d entries forbidden (actions/outcome cannot cause confounders)",
             n_forbidden)

    # Import NOTEARS as black box
    from notears.linear import notears_linear

    # ── Lambda sweep ─────────────────────────────────────────────────
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

        # Apply tier mask post-hoc
        W = W_raw * tier_mask

        n_raw    = int((np.abs(W_raw) >= args.w_threshold).sum())
        n_masked = int((np.abs(W)     >= args.w_threshold).sum())
        n_removed = n_raw - n_masked

        log.info("NOTEARS done in %.1f seconds (%.1f min)", elapsed, elapsed / 60)
        log.info("  Edges (|w| >= %.3f): %d total (%d removed by tier mask)",
                 args.w_threshold, n_masked, n_removed)

        edge_df       = extract_edges(W, var_names, w_threshold=args.w_threshold)
        action_edge_df = extract_action_edges(edge_df, actions, outcome)
        log.info("  Action -> readmit_30d edges: %d", len(action_edge_df))

        lam_str = f"{lam:.4f}".replace(".", "p")
        edge_df.to_csv(report_dir / f"edges_lambda_{lam_str}.csv", index=False)
        action_edge_df.to_csv(report_dir / f"action_edges_lambda_{lam_str}.csv", index=False)

        if len(action_edge_df) > 0:
            log.info("  Action edges found:")
            for _, row in action_edge_df.iterrows():
                log.info("    %-35s --> %-15s  w=%.4f",
                         row["source"], row["target"], row["weight"])
        else:
            log.info("  No action -> readmit_30d edges at this lambda.")

        all_results[lam] = {
            "edge_df": edge_df,
            "action_edge_df": action_edge_df,
            "elapsed": elapsed,
            "n_total_edges": n_masked,
        }

    total_elapsed = time.time() - total_t0
    log.info("")
    log.info("All lambdas completed in %.1f seconds (%.1f min)",
             total_elapsed, total_elapsed / 60)

    # ── Lambda sensitivity table ─────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("LAMBDA SENSITIVITY TABLE (action -> readmit_30d edges)")
    log.info("=" * 70)

    action_set = set(actions)
    all_action_pairs = set()
    for lam in all_results:
        for _, row in all_results[lam]["action_edge_df"].iterrows():
            src = row["source"]
            tgt = row["target"]
            all_action_pairs.add((src, tgt))

    sweep_rows = []
    for src, tgt in sorted(all_action_pairs):
        row_data = {"source": src, "target": tgt}
        for lam in sorted(args.lambdas, reverse=True):
            adf = all_results[lam]["action_edge_df"]
            match = adf[((adf["source"] == src) & (adf["target"] == tgt)) |
                        ((adf["source"] == tgt) & (adf["target"] == src))]
            if len(match) > 0:
                w = match.iloc[0]["weight"]
                row_data[f"lam={lam}"] = f"{w:.4f}"
            else:
                row_data[f"lam={lam}"] = "-"
        sweep_rows.append(row_data)

    sweep_df = pd.DataFrame(sweep_rows) if sweep_rows else pd.DataFrame()
    sweep_df.to_csv(report_dir / "lambda_sweep.csv", index=False)
    log.info("Saved: lambda_sweep.csv")
    if len(sweep_df) > 0:
        log.info("\n%s", sweep_df.to_string(index=False))
    else:
        log.info("  (no action -> readmit_30d edges found at any lambda)")

    log.info("")
    log.info("V4 NOTEARS complete. Results in: %s", report_dir)
    log.info("Total runtime: %.1f minutes", total_elapsed / 60)


if __name__ == "__main__":
    main()
