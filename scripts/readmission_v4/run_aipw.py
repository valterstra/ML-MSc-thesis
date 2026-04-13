"""AIPW effect estimation for V4 readmission pipeline.

Estimates doubly-robust AIPW ATE for each of the 6 actions identified
by PC causal discovery as having directed edges into readmit_30d.

Both the propensity model P(action=1|X) and outcome model P(readmit=1|X,A)
use logistic regression. Confounders are the 18 selected by LightGBM
double selection.

Bootstrap CIs: 500 resamples, fixed fitted models (propensity + outcome
refitted on each bootstrap sample for valid inference).

Overlap check: reports min/max/mean propensity for treated vs untreated.

Outputs (--report-dir):
  aipw_results.csv   -- ATE, 95% CI, p-value, overlap stats per action
  aipw_results.txt   -- human-readable summary
  run_log.txt

Usage:
    python scripts/readmission_v4/run_aipw.py

    # Faster (fewer bootstrap samples)
    python scripts/readmission_v4/run_aipw.py --n-boot 100
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTCOME = "readmit_30d"

ACTIONS = [
    "consult_ot",
    "consult_social_work",
    "consult_diabetes_education",
    "dc_med_antibiotic",
    "dc_med_statin",
    "dc_med_steroid",
]

CONFOUNDERS = [
    "discharge_service",
    "los_days",
    "first_service",
    "age_at_admit",
    "admission_type",
    "ed_dwell_hours",
    "rx_anticoagulant",
    "dc_location",
    "charlson_score",
    "flag_chf",
    "drg_mortality",
    "flag_hypertension",
    "icu_days",
    "rx_insulin",
    "dc_lab_creatinine",
    "dc_lab_hemoglobin",
    "dc_lab_platelets",
    "flag_alcohol",
]

CATEGORICAL_COLS = [
    "discharge_service", "first_service", "admission_type", "dc_location",
]

CLIP_LOW  = 0.01
CLIP_HIGH = 0.99

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Label-encode categoricals, impute missing, return float64 array."""
    X = df[feature_cols].copy()
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
    return X.values.astype(np.float64)


def make_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=1)),
    ])


# ---------------------------------------------------------------------------
# AIPW estimator
# ---------------------------------------------------------------------------

def aipw_ate(
    X_conf: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    prop_model: Pipeline,
    out_model: Pipeline,
) -> float:
    """AIPW ATE using pre-fitted propensity and outcome models.

    prop_model: fitted on X_conf -> t
    out_model:  fitted on [X_conf | t] -> y
    """
    p = np.clip(prop_model.predict_proba(X_conf)[:, 1], CLIP_LOW, CLIP_HIGH)

    X_1 = np.hstack([X_conf, np.ones((len(X_conf), 1))])
    X_0 = np.hstack([X_conf, np.zeros((len(X_conf), 1))])

    mu1 = out_model.predict_proba(X_1)[:, 1]
    mu0 = out_model.predict_proba(X_0)[:, 1]

    psi = (
        mu1 - mu0
        + (t / p) * (y - mu1)
        - ((1 - t) / (1 - p)) * (y - mu0)
    )
    return float(np.mean(psi))


def fit_models(X_conf, t, y):
    prop_model = make_lr_pipeline()
    prop_model.fit(X_conf, t)

    X_with_t = np.hstack([X_conf, t.reshape(-1, 1)])
    out_model = make_lr_pipeline()
    out_model.fit(X_with_t, y)

    return prop_model, out_model


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    X_conf: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float, float, list[float]]:
    """Returns (point_estimate, ci_lower, ci_upper, boot_estimates)."""
    prop_model, out_model = fit_models(X_conf, t, y)
    point = aipw_ate(X_conf, t, y, prop_model, out_model)

    rng = np.random.default_rng(seed)
    n = len(y)
    boot_estimates: list[float] = []

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb, tb, yb = X_conf[idx], t[idx], y[idx]
        if tb.sum() < 5 or (1 - tb).sum() < 5:
            continue
        try:
            pm, om = fit_models(Xb, tb, yb)
            est = aipw_ate(Xb, tb, yb, pm, om)
            boot_estimates.append(est)
        except Exception:
            continue
        if (i + 1) % 50 == 0:
            log.info("    bootstrap %d/%d ...", i + 1, n_boot)

    arr = np.array(boot_estimates)
    ci_lower = float(np.percentile(arr, 2.5))
    ci_upper = float(np.percentile(arr, 97.5))
    return point, ci_lower, ci_upper, boot_estimates


# ---------------------------------------------------------------------------
# Overlap check
# ---------------------------------------------------------------------------

def overlap_check(X_conf: np.ndarray, t: np.ndarray, prop_model: Pipeline) -> dict:
    p = prop_model.predict_proba(X_conf)[:, 1]
    p1 = p[t == 1]
    p0 = p[t == 0]
    return {
        "n_treated":    int(t.sum()),
        "n_control":    int((1 - t).sum()),
        "ps_mean_treated":  round(float(p1.mean()), 4),
        "ps_mean_control":  round(float(p0.mean()), 4),
        "ps_min_treated":   round(float(p1.min()),  4),
        "ps_max_control":   round(float(p0.max()),  4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "readmission_v4_admissions.csv"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "readmission_v4" / "aipw"),
    )
    parser.add_argument("--n-boot",   type=int, default=500)
    parser.add_argument("--sample-n", type=int, default=50_000,
                        help="Subsample train rows for estimation (default: 50000, 0=all)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
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
    log.info("V4 AIPW Effect Estimation")
    log.info("=" * 70)
    log.info("Actions: %d | Confounders: %d | n_boot: %d",
             len(ACTIONS), len(CONFOUNDERS), args.n_boot)

    # ── Load data ────────────────────────────────────────────────────
    log.info("Loading: %s", args.csv)
    df = pd.read_csv(args.csv, low_memory=False)
    train = df[df["split"] == "train"].copy()
    log.info("Train rows: %d", len(train))

    if args.sample_n > 0 and args.sample_n < len(train):
        train = train.sample(n=args.sample_n, random_state=args.seed)
        log.info("Subsampled to %d rows", len(train))

    # ── Preprocess confounders once ───────────────────────────────────
    log.info("Preprocessing confounders ...")
    X_conf = preprocess(train, CONFOUNDERS)
    y = train[OUTCOME].fillna(0).astype(int).values
    log.info("Outcome prevalence: %.1f%%", 100 * y.mean())

    # ── Run AIPW per action ───────────────────────────────────────────
    results = []
    total_t0 = time.time()

    for action in ACTIONS:
        log.info("")
        log.info("=" * 70)
        log.info("ACTION: %s", action)
        log.info("=" * 70)
        t = train[action].fillna(0).astype(int).values
        prevalence = t.mean()
        log.info("  Prevalence: %.1f%% treated (%d / %d)",
                 100 * prevalence, t.sum(), len(t))

        if t.sum() < 50 or (1 - t).sum() < 50:
            log.info("  SKIPPED: insufficient treated or control observations")
            continue

        t0 = time.time()
        point, ci_lo, ci_hi, boot_ests = bootstrap_ci(
            X_conf, t, y, n_boot=args.n_boot, seed=args.seed
        )
        elapsed = time.time() - t0

        # p-value: proportion of bootstrap estimates with opposite sign
        arr = np.array(boot_ests)
        if point > 0:
            p_val = float(np.mean(arr <= 0)) * 2
        else:
            p_val = float(np.mean(arr >= 0)) * 2
        p_val = min(p_val, 1.0)

        # Overlap check using a freshly fitted propensity model
        prop_model, _ = fit_models(X_conf, t, y)
        overlap = overlap_check(X_conf, t, prop_model)

        log.info("  ATE:  %.4f  (%.4f, %.4f)  p=%.4f  [%.1f sec]",
                 point, ci_lo, ci_hi, p_val, elapsed)
        log.info("  Overlap: n_treated=%d  ps_mean_treated=%.3f  ps_mean_control=%.3f",
                 overlap["n_treated"], overlap["ps_mean_treated"], overlap["ps_mean_control"])

        results.append({
            "action":           action,
            "ate":              round(point, 4),
            "ci_lower":         round(ci_lo, 4),
            "ci_upper":         round(ci_hi, 4),
            "p_value":          round(p_val, 4),
            "n_treated":        overlap["n_treated"],
            "n_control":        overlap["n_control"],
            "ps_mean_treated":  overlap["ps_mean_treated"],
            "ps_mean_control":  overlap["ps_mean_control"],
            "ps_min_treated":   overlap["ps_min_treated"],
            "ps_max_control":   overlap["ps_max_control"],
        })

    total_elapsed = time.time() - total_t0

    # ── Save results ─────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(report_dir / "aipw_results.csv", index=False)
    log.info("")
    log.info("Saved: aipw_results.csv")

    # Human-readable summary
    txt_path = report_dir / "aipw_results.txt"
    with open(txt_path, "w") as f:
        f.write("V4 AIPW EFFECT ESTIMATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Outcome: {OUTCOME}\n")
        f.write(f"Confounders: {len(CONFOUNDERS)}\n")
        f.write(f"Bootstrap samples: {args.n_boot}\n")
        f.write(f"Train rows: {len(train)}\n\n")
        f.write(f"{'Action':<30} {'ATE':>8} {'95% CI':>20} {'p-value':>8} {'n_treated':>10}\n")
        f.write("-" * 80 + "\n")
        for row in results:
            ci_str = f"({row['ci_lower']:.4f}, {row['ci_upper']:.4f})"
            sig = "*" if row["p_value"] < 0.05 else ""
            f.write(f"{row['action']:<30} {row['ate']:>8.4f} {ci_str:>20} "
                    f"{row['p_value']:>8.4f}{sig}  n={row['n_treated']}\n")

    log.info("Saved: aipw_results.txt")
    log.info("")
    log.info("=" * 70)
    log.info("FINAL RESULTS")
    log.info("=" * 70)
    log.info("%-30s %8s %20s %8s", "Action", "ATE", "95% CI", "p-value")
    log.info("-" * 70)
    for row in results:
        ci_str = f"({row['ci_lower']:.4f}, {row['ci_upper']:.4f})"
        sig = " *" if row["p_value"] < 0.05 else ""
        log.info("%-30s %8.4f %20s %8.4f%s",
                 row["action"], row["ate"], ci_str, row["p_value"], sig)
    log.info("")
    log.info("Total runtime: %.1f minutes", total_elapsed / 60)


if __name__ == "__main__":
    main()
