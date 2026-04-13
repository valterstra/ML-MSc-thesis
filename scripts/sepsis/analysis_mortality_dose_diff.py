"""
Mortality by dose difference analysis.

Replicates the primary clinical result from Raghu et al. 2017:
  For each timestep, compute the difference between what the RL agent
  recommended and what the clinician actually did. Plot observed
  in-hospital mortality against that dose difference, stratified by
  SOFA severity (low/medium/high).

If the RL policy is clinically meaningful, mortality should be lowest
when the clinician matched the agent (dose difference = 0) and higher
when they deviated.

Inputs:
  data/processed/sepsis/rl_test_set_original.csv
  models/sepsis_rl/continuous/dqn/dqn_actions_test.pkl
  models/sepsis_rl/continuous/sarsa_phys/phys_actions_test.pkl (optional)

Outputs:
  reports/sepsis_rl/mortality_dose_diff.csv
  reports/sepsis_rl/mortality_dose_diff_dqn.png
  reports/sepsis_rl/mortality_dose_diff_sarsa.png  (if SARSA available)
"""
import argparse
import logging
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

# ── SOFA severity strata ──────────────────────────────────────────────
SOFA_STRATA = {
    "Low (0-6)":    (0,  6),
    "Medium (7-13)": (7, 13),
    "High (14+)":   (14, 99),
}
SOFA_COLORS = {
    "Low (0-6)":     "#2196F3",
    "Medium (7-13)": "#FF9800",
    "High (14+)":    "#F44336",
}


def decode_action(action_id):
    """Decode action_id -> (iv_level, vaso_level). action_id = 5*iv + vaso."""
    iv = int(action_id) // 5
    vaso = int(action_id) % 5
    return iv, vaso


def compute_dose_diffs(df, agent_actions):
    """Add agent iv/vaso levels and dose differences to dataframe."""
    df = df.copy()
    agent_iv = np.array([decode_action(a)[0] for a in agent_actions])
    agent_vaso = np.array([decode_action(a)[1] for a in agent_actions])

    df["agent_iv"] = agent_iv
    df["agent_vaso"] = agent_vaso
    df["iv_diff"] = agent_iv - df["iv_input"].values
    df["vaso_diff"] = agent_vaso - df["vaso_input"].values
    return df


def mortality_by_diff(df, diff_col, sofa_col="SOFA", outcome_col="died_in_hosp",
                      min_count=50):
    """
    For each SOFA stratum and each dose difference value,
    compute observed mortality rate and sample size.

    Returns dict: {stratum_name: DataFrame(diff, mortality, n)}
    """
    results = {}
    for stratum, (lo, hi) in SOFA_STRATA.items():
        mask = (df[sofa_col] >= lo) & (df[sofa_col] <= hi)
        sub = df[mask]

        rows = []
        for diff_val in range(-4, 5):
            grp = sub[sub[diff_col] == diff_val]
            n = len(grp)
            if n >= min_count:
                mort = grp[outcome_col].mean()
                rows.append({"diff": diff_val, "mortality": mort, "n": n})

        if rows:
            results[stratum] = pd.DataFrame(rows)

    return results


def plot_mortality_diff(iv_results, vaso_results, policy_name, save_path):
    """Two-panel plot: IV dose diff (left) and vasopressor dose diff (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, results, title in zip(
        axes,
        [iv_results, vaso_results],
        ["IV Fluid Dose Difference", "Vasopressor Dose Difference"],
    ):
        for stratum, df_strat in results.items():
            color = SOFA_COLORS[stratum]
            ax.plot(
                df_strat["diff"],
                df_strat["mortality"] * 100,
                marker="o",
                color=color,
                label=stratum,
                linewidth=2,
                markersize=6,
            )

        ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5,
                   label="Agent = Clinician")
        ax.set_xlabel("Agent Dose Level - Clinician Dose Level", fontsize=11)
        ax.set_ylabel("Observed In-Hospital Mortality (%)", fontsize=11)
        ax.set_title(f"{policy_name}: {title}", fontsize=12)
        ax.set_xticks(range(-4, 5))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 60)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Plot saved: %s", save_path)


def run_analysis(df, agent_actions, policy_name, report_dir, min_count=50):
    """Full analysis pipeline for one policy."""
    logging.info("=== %s ===", policy_name)
    logging.info("Test set: %d timesteps, %d patients",
                 len(df), df["icustayid"].nunique())

    df = compute_dose_diffs(df, agent_actions)

    # Overall action agreement
    agree_iv = (df["iv_diff"] == 0).mean() * 100
    agree_vaso = (df["vaso_diff"] == 0).mean() * 100
    agree_both = ((df["iv_diff"] == 0) & (df["vaso_diff"] == 0)).mean() * 100
    logging.info("Action agreement: IV=%.1f%%, Vaso=%.1f%%, Both=%.1f%%",
                 agree_iv, agree_vaso, agree_both)

    # Mortality by dose difference
    iv_results = mortality_by_diff(df, "iv_diff", min_count=min_count)
    vaso_results = mortality_by_diff(df, "vaso_diff", min_count=min_count)

    # Log results table
    for drug, results in [("IV", iv_results), ("Vaso", vaso_results)]:
        logging.info("%s dose difference mortality:", drug)
        for stratum, df_strat in results.items():
            logging.info("  %s:", stratum)
            for _, row in df_strat.iterrows():
                logging.info("    diff=%+d  mortality=%.1f%%  n=%d",
                             row["diff"], row["mortality"] * 100, row["n"])

    # Plot
    plot_path = os.path.join(
        report_dir, f"mortality_dose_diff_{policy_name.lower().replace(' ', '_')}.png"
    )
    plot_mortality_diff(iv_results, vaso_results, policy_name, plot_path)

    # Save CSV
    rows = []
    for drug, results in [("IV", iv_results), ("Vaso", vaso_results)]:
        for stratum, df_strat in results.items():
            for _, row in df_strat.iterrows():
                rows.append({
                    "policy": policy_name,
                    "drug": drug,
                    "sofa_stratum": stratum,
                    "dose_diff": int(row["diff"]),
                    "mortality_pct": round(row["mortality"] * 100, 2),
                    "n": int(row["n"]),
                })

    return pd.DataFrame(rows), df[["icustayid", "bloc", "iv_diff", "vaso_diff",
                                    "iv_input", "vaso_input", "agent_iv", "agent_vaso",
                                    "SOFA", "died_in_hosp"]]


def main():
    parser = argparse.ArgumentParser(description="Mortality by dose difference analysis")
    parser.add_argument("--data-dir", default="data/processed/sepsis")
    parser.add_argument("--model-dir", default="models/sepsis_rl/continuous")
    parser.add_argument("--report-dir", default="reports/sepsis_rl")
    parser.add_argument("--min-count", type=int, default=50,
                        help="Minimum timesteps per bin to include in plot")
    parser.add_argument("--log", default="logs/analysis_mortality_dose_diff.log")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(args.log, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    os.makedirs(args.report_dir, exist_ok=True)

    # ── Load test set (original unscaled for SOFA values) ─────────────
    logging.info("Loading test set...")
    df = pd.read_csv(f"{args.data_dir}/rl_test_set_original.csv")
    logging.info("  %d rows, %d patients", len(df), df["icustayid"].nunique())

    # ── Load precomputed agent actions ────────────────────────────────
    dqn_path = f"{args.model_dir}/dqn/dqn_actions_test.pkl"
    sarsa_path = f"{args.model_dir}/sarsa_phys/phys_actions_test.pkl"

    all_results = []

    # DQN analysis
    if os.path.exists(dqn_path):
        with open(dqn_path, "rb") as f:
            dqn_actions = pickle.load(f)
        logging.info("DQN actions loaded: %d", len(dqn_actions))
        assert len(dqn_actions) == len(df), \
            f"DQN actions length {len(dqn_actions)} != test set {len(df)}"
        result_df, detail_df = run_analysis(df, dqn_actions, "DQN", args.report_dir,
                                             args.min_count)
        all_results.append(result_df)
        detail_df.to_csv(f"{args.report_dir}/dose_diff_detail_dqn.csv", index=False)
    else:
        logging.warning("DQN actions not found: %s", dqn_path)

    # SARSA physician analysis
    if os.path.exists(sarsa_path):
        with open(sarsa_path, "rb") as f:
            sarsa_actions = pickle.load(f)
        logging.info("SARSA actions loaded: %d", len(sarsa_actions))
        assert len(sarsa_actions) == len(df), \
            f"SARSA actions length {len(sarsa_actions)} != test set {len(df)}"
        result_df, detail_df = run_analysis(df, sarsa_actions, "SARSA Physician",
                                             args.report_dir, args.min_count)
        all_results.append(result_df)
        detail_df.to_csv(f"{args.report_dir}/dose_diff_detail_sarsa.csv", index=False)
    else:
        logging.warning("SARSA actions not found: %s", sarsa_path)

    # Combined CSV
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out_path = f"{args.report_dir}/mortality_dose_diff.csv"
        combined.to_csv(out_path, index=False)
        logging.info("Results saved: %s", out_path)

    logging.info("Analysis complete.")


if __name__ == "__main__":
    main()
