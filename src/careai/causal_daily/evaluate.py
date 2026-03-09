"""Full causal analysis pipeline: fit propensity, estimate ATEs, format results."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .balance import check_overlap
from .estimators import bootstrap_ci, naive_ate
from .features import EXPECTED_DIRECTION, TREATMENT_OUTCOME_PAIRS
from .propensity import PropensityModel, fit_propensity_models, predict_propensity


def run_causal_analysis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    treatment_outcome_pairs: list[tuple[str, str]] | None = None,
    n_boot: int = 500,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full doubly-robust ATE estimation pipeline.

    Parameters
    ----------
    train_df:
        Training split — used to fit propensity models.
    test_df:
        Test split — used to estimate ATEs (avoids overfitting propensity).
    treatment_outcome_pairs:
        List of (treatment, outcome) tuples. Defaults to TREATMENT_OUTCOME_PAIRS.
    n_boot:
        Number of bootstrap resamples for CIs.
    seed:
        Random seed.
    verbose:
        Print progress to stdout.

    Returns
    -------
    pd.DataFrame with columns:
        treatment, outcome, n_treated, n_control,
        naive_ate, causal_ate, ci_lower, ci_upper,
        expected_direction, causal_direction, verdict
    """
    if treatment_outcome_pairs is None:
        treatment_outcome_pairs = TREATMENT_OUTCOME_PAIRS

    if verbose:
        print("Fitting propensity models on training data...")
    prop_model = fit_propensity_models(train_df)

    rows: list[dict] = []

    for treatment, outcome in treatment_outcome_pairs:
        next_col = f"next_{outcome}"
        if next_col not in test_df.columns:
            warnings.warn(f"Outcome column '{next_col}' not found — skipping ({treatment}, {outcome})")
            continue
        if treatment not in test_df.columns:
            warnings.warn(f"Treatment column '{treatment}' not found — skipping")
            continue

        if verbose:
            print(f"  Estimating: {treatment} -> {outcome} ...", end=" ", flush=True)

        ps = predict_propensity(prop_model, test_df, treatment)

        # Overlap diagnostics
        overlap = check_overlap(test_df, ps, treatment)
        if overlap["poor_overlap"]:
            warnings.warn(
                f"Poor overlap for {treatment}: "
                f"{overlap['frac_outside_0.1_0.9']:.1%} of rows outside [0.1, 0.9]"
            )

        # Naive estimate
        n_ate = naive_ate(test_df, treatment, outcome)

        # AIPW + bootstrap CI
        try:
            c_ate, ci_lo, ci_hi = bootstrap_ci(
                test_df, prop_model, treatment, outcome,
                n_boot=n_boot, seed=seed,
            )
        except Exception as exc:
            warnings.warn(f"Bootstrap failed for ({treatment}, {outcome}): {exc}")
            c_ate, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")

        # Direction verdict
        exp = EXPECTED_DIRECTION.get((treatment, outcome), "unknown")
        if np.isnan(c_ate):
            causal_dir = "unknown"
            verdict = "error"
        elif c_ate > 0:
            causal_dir = "up"
        else:
            causal_dir = "down"

        if not np.isnan(c_ate):
            if exp == "unknown":
                verdict = "unknown"
            elif causal_dir == exp:
                verdict = "correct"
            else:
                verdict = "wrong"

        rows.append({
            "treatment":          treatment,
            "outcome":            outcome,
            "n_treated":          overlap["n_treated"],
            "n_control":          overlap["n_control"],
            "naive_ate":          round(n_ate, 4),
            "causal_ate":         round(c_ate, 4),
            "ci_lower":           round(ci_lo, 4),
            "ci_upper":           round(ci_hi, 4),
            "expected_direction": exp,
            "causal_direction":   causal_dir,
            "verdict":            verdict,
        })

        if verbose:
            sign_flip = (n_ate * c_ate < 0)
            flip_str = "  *** SIGN FLIP ***" if sign_flip else ""
            print(
                f"naive={n_ate:+.2f}  causal={c_ate:+.2f} "
                f"[{ci_lo:+.2f}, {ci_hi:+.2f}]  "
                f"expected={exp}  verdict={verdict}{flip_str}"
            )

    return pd.DataFrame(rows)


def print_results_table(results: pd.DataFrame) -> None:
    """Print a human-readable side-by-side comparison of naive vs causal ATEs."""
    if results.empty:
        print("No results to display.")
        return

    header = (
        f"{'Treatment':<25} {'Outcome':<32} "
        f"{'Naive ATE':>10} {'Causal ATE':>11} "
        f"{'95% CI':>20} {'Expected':>9} {'Verdict':>8}"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("  CAUSAL TREATMENT EFFECTS (AIPW, doubly robust)")
    print("=" * len(header))
    print(header)
    print(sep)

    for _, row in results.iterrows():
        ci_str = f"[{row['ci_lower']:+.2f}, {row['ci_upper']:+.2f}]"
        flag = ""
        if row["verdict"] == "correct":
            flag = " [OK]"
        elif row["verdict"] == "wrong":
            flag = " [!!]"
        # Mark sign flips (confounding visible)
        if row["naive_ate"] * row["causal_ate"] < 0:
            flag += " (FLIP)"

        print(
            f"{row['treatment']:<25} {row['outcome']:<32} "
            f"{row['naive_ate']:>+10.2f} {row['causal_ate']:>+11.2f} "
            f"{ci_str:>20} {row['expected_direction']:>9} {row['verdict']:>8}{flag}"
        )

    print(sep)
    n_correct = (results["verdict"] == "correct").sum()
    n_wrong = (results["verdict"] == "wrong").sum()
    n_total = len(results)
    print(f"  {n_correct}/{n_total} pairs match expected direction  |  {n_wrong} wrong")
    print("=" * len(header))
    print()
