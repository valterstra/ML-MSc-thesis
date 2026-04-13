"""
Step 12b -- Policy analysis: qualitative evaluation following Raghu 2017 Figures 1 & 2.

Chains from:
  step 11 outputs  -- models/icu_readmit/continuous/ddqn/dqn_actions_test.pkl
  step 10b outputs -- data/processed/icu_readmit/rl_dataset_broad.parquet
                      data/processed/icu_readmit/scaler_params_broad.json

Produces:
  reports/icu_readmit/
    fig1_action_distribution.png      -- physician vs DDQN drug usage by SOFA severity
    fig2_readmission_vs_disagreement.png  -- readmission rate vs action disagreement
    policy_analysis_stats.json        -- underlying numbers for both figures

Raghu 2017 used qualitative plots as PRIMARY evaluation, with DR as secondary.
This script builds the ICU readmission equivalents of their two key figures.

  Raghu Fig 1: 2D histogram (IV fluid x vasopressor) for physician vs RL, by SOFA severity
  Our   Fig 1: Per-drug usage frequency for physician vs DDQN, by SOFA severity (5 binary drugs)

  Raghu Fig 2: Mortality rate vs (RL dose - physician dose), binned by SOFA severity
  Our   Fig 2: Readmission rate vs Hamming distance(DDQN action, physician action), by severity
               Hamming distance = number of drugs where DDQN and physician disagree (0-5)

Key claim (Raghu Fig 2): if the learned policy is valid, outcomes should be worst when the
clinician and RL disagree most, and best when they agree (dose difference = 0).
"""
import argparse
import json
import logging
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DRUG_NAMES  = ["vasopressor", "ivfluid", "antibiotic", "sedation", "diuretic"]
SOFA_BINS   = {"low": (None, 5), "medium": (5, 15), "high": (15, None)}
SOFA_COLORS = {"low": "#2196F3", "medium": "#FF9800", "high": "#F44336"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def decode_actions(actions_int):
    """Decode integer actions 0-31 to (N, 5) binary array."""
    arr = np.asarray(actions_int, dtype=np.int32)
    return ((arr[:, None] >> np.arange(5)) & 1).astype(np.float32)


def sofa_severity(sofa_raw):
    """Map raw SOFA scores to 'low' / 'medium' / 'high' severity labels."""
    labels = np.full(len(sofa_raw), "medium", dtype=object)
    labels[sofa_raw <  5] = "low"
    labels[sofa_raw > 15] = "high"
    return labels


# ---------------------------------------------------------------------------
# Figure 1 -- Action distribution comparison
# ---------------------------------------------------------------------------

def plot_action_distribution(phys_bin, ddqn_bin, severity_labels, report_dir):
    """Per-drug usage frequency: physician vs DDQN, split by SOFA severity.

    phys_bin: (N, 5) binary array of physician actions
    ddqn_bin: (N, 5) binary array of DDQN recommended actions
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Fig 1 -- Drug Usage Frequency: Physician vs DDQN Policy by SOFA Severity",
                 fontsize=13, fontweight="bold")

    stats = {}
    x = np.arange(len(DRUG_NAMES))
    width = 0.35

    for ax, (sev_name, _) in zip(axes, SOFA_BINS.items()):
        mask = severity_labels == sev_name
        n    = mask.sum()

        phys_rates = phys_bin[mask].mean(axis=0) * 100   # % of timesteps
        ddqn_rates = ddqn_bin[mask].mean(axis=0) * 100

        bars_phys = ax.bar(x - width / 2, phys_rates, width,
                           label="Physician", color="#455A64", alpha=0.85)
        bars_ddqn = ax.bar(x + width / 2, ddqn_rates, width,
                           label="DDQN policy", color=SOFA_COLORS[sev_name], alpha=0.85)

        ax.set_title(f"SOFA severity: {sev_name.upper()}\n(n={n:,} timesteps)",
                     fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(DRUG_NAMES, rotation=20, ha="right")
        ax.set_ylabel("% timesteps drug active" if sev_name == "low" else "")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        stats[sev_name] = {
            "n_timesteps": int(n),
            "physician": {d: float(r) for d, r in zip(DRUG_NAMES, phys_rates)},
            "ddqn":      {d: float(r) for d, r in zip(DRUG_NAMES, ddqn_rates)},
        }

    plt.tight_layout()
    path = os.path.join(report_dir, "fig1_action_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Fig 1 saved: %s", path)
    return stats


# ---------------------------------------------------------------------------
# Figure 2 -- Readmission rate vs action disagreement
# ---------------------------------------------------------------------------

def plot_readmission_vs_disagreement(phys_bin, ddqn_bin, readmit, severity_labels,
                                     report_dir):
    """Readmission rate vs Hamming distance between DDQN and physician actions.

    Hamming distance = number of drugs (0-5) where the two policies disagree.
    Readmit is a patient-level label propagated to all timesteps of that stay.

    Equivalent to Raghu 2017 Figure 2 (mortality vs dose difference).
    """
    hamming = (phys_bin != ddqn_bin).sum(axis=1)  # 0-5 disagreement per timestep

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle(
        "Fig 2 -- Readmission Rate vs DDQN-Physician Action Disagreement by SOFA Severity\n"
        "(Raghu 2017 Fig 2 equivalent: outcome should be worst where disagreement is highest)",
        fontsize=12, fontweight="bold",
    )

    stats = {}

    for ax, (sev_name, _) in zip(axes, SOFA_BINS.items()):
        mask = severity_labels == sev_name
        h_sev = hamming[mask]
        r_sev = readmit[mask]

        bin_rates = []
        bin_ns    = []
        bin_labels = []

        for d in range(6):  # 0-5 drugs different
            bin_mask = h_sev == d
            n_bin    = bin_mask.sum()
            if n_bin > 0:
                rate = r_sev[bin_mask].mean() * 100
            else:
                rate = np.nan
            bin_rates.append(rate)
            bin_ns.append(n_bin)
            bin_labels.append(str(d))

        bin_rates = np.array(bin_rates)
        valid     = ~np.isnan(bin_rates)

        ax.bar(np.arange(6)[valid], bin_rates[valid],
               color=SOFA_COLORS[sev_name], alpha=0.8, edgecolor="black", linewidth=0.5)

        # Annotate n per bar
        for d in range(6):
            if valid[d] and bin_ns[d] > 0:
                ax.text(d, bin_rates[d] + 0.5, f"n={bin_ns[d]:,}",
                        ha="center", va="bottom", fontsize=7, rotation=45)

        ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2, alpha=0.6,
                   label="Physician agrees with DDQN")

        ax.set_title(f"SOFA severity: {sev_name.upper()}", fontsize=11)
        ax.set_xlabel("Drugs where DDQN and physician disagree (Hamming distance)")
        ax.set_ylabel("Readmission rate (%)" if sev_name == "low" else "")
        ax.set_xticks(range(6))
        ax.set_xticklabels([f"{d} drug{'s' if d != 1 else ''}" for d in range(6)],
                           rotation=20, ha="right", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        stats[sev_name] = {
            "hamming_bins": {
                str(d): {"readmission_rate_pct": float(bin_rates[d]) if valid[d] else None,
                         "n_timesteps": int(bin_ns[d])}
                for d in range(6)
            }
        }

    plt.tight_layout()
    path = os.path.join(report_dir, "fig2_readmission_vs_disagreement.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Fig 2 saved: %s", path)
    return stats


# ---------------------------------------------------------------------------
# Agreement summary
# ---------------------------------------------------------------------------

def compute_agreement_summary(phys_bin, ddqn_bin, severity_labels):
    """Overall and per-severity agreement rate between DDQN and physician."""
    exact_match = (phys_bin == ddqn_bin).all(axis=1)  # all 5 drugs identical
    summary = {"overall": float(exact_match.mean() * 100)}

    for sev_name in SOFA_BINS:
        mask = severity_labels == sev_name
        summary[sev_name] = float(exact_match[mask].mean() * 100)

    logging.info("Exact action agreement (DDQN == physician):")
    for k, v in summary.items():
        logging.info("  %-10s %.1f%%", k, v)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 12b: Policy analysis (Raghu Fig 1 & 2)")
    parser.add_argument("--data-dir",   default=str(PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "legacy" / "step_20_24"))
    parser.add_argument("--model-dir",  default=str(PROJECT_ROOT / "models" / "icu_readmit" / "legacy" / "step_25_33" / "continuous" / "ddqn"))
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "reports" / "icu_readmit" / "legacy" / "step_30_33" / "policy_analysis_broad"))
    parser.add_argument("--log",        default=str(PROJECT_ROOT / "logs" / "legacy" / "icu_readmit" / "step_31_policy_analysis_legacy.log"))
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(args.log, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Step 12b started.")

    # ── Load parquet (test split only) ───────────────────────────────
    parquet_path = os.path.join(args.data_dir, "rl_dataset_broad.parquet")
    logging.info("Loading %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    df = df[df["split"] == "test"].reset_index(drop=True)
    logging.info("  Test split: %d rows, %d stays",
                 len(df), df["icustayid"].nunique())

    # ── Load SOFA scaler params to recover raw SOFA ───────────────────
    scaler_path = os.path.join(args.data_dir, "scaler_params_broad.json")
    with open(scaler_path) as f:
        scaler = json.load(f)

    sofa_mean = scaler["SOFA"]["mean"]
    sofa_std  = scaler["SOFA"]["std"]
    sofa_raw  = df["s_SOFA"].values * sofa_std + sofa_mean
    sev_labels = sofa_severity(sofa_raw)

    logging.info("SOFA severity distribution in test set:")
    for sev in ["low", "medium", "high"]:
        n = (sev_labels == sev).sum()
        logging.info("  %-8s %6d timesteps (%.1f%%)", sev, n, 100 * n / len(df))

    # ── Load DDQN recommended actions ────────────────────────────────
    actions_path = os.path.join(args.model_dir, "dqn_actions_test.pkl")
    logging.info("Loading DDQN actions: %s", actions_path)
    with open(actions_path, "rb") as f:
        ddqn_actions = pickle.load(f)

    if len(ddqn_actions) != len(df):
        raise ValueError(
            f"DDQN actions length {len(ddqn_actions)} does not match "
            f"test set length {len(df)}. Make sure step 11 full run is complete."
        )

    # ── Decode actions to 5-bit binary ───────────────────────────────
    phys_bin = decode_actions(df["a"].values)       # physician actions from data
    ddqn_bin = decode_actions(ddqn_actions)          # DDQN recommended actions

    # Readmission label -- patient-level, propagated to all rows
    readmit = df["readmit_30d"].values.astype(float)

    logging.info("Physician action stats: mean drugs active per timestep = %.2f",
                 phys_bin.sum(axis=1).mean())
    logging.info("DDQN action stats:      mean drugs active per timestep = %.2f",
                 ddqn_bin.sum(axis=1).mean())

    # ── Agreement summary ─────────────────────────────────────────────
    agreement = compute_agreement_summary(phys_bin, ddqn_bin, sev_labels)

    # ── Figure 1 ──────────────────────────────────────────────────────
    logging.info("--- Generating Fig 1: action distribution ---")
    fig1_stats = plot_action_distribution(phys_bin, ddqn_bin, sev_labels, args.report_dir)

    # ── Figure 2 ──────────────────────────────────────────────────────
    logging.info("--- Generating Fig 2: readmission vs disagreement ---")
    fig2_stats = plot_readmission_vs_disagreement(
        phys_bin, ddqn_bin, readmit, sev_labels, args.report_dir,
    )

    # ── Save stats ────────────────────────────────────────────────────
    all_stats = {
        "agreement_pct": agreement,
        "fig1_action_distribution": fig1_stats,
        "fig2_readmission_vs_disagreement": fig2_stats,
    }
    stats_path = os.path.join(args.report_dir, "policy_analysis_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    logging.info("Stats saved: %s", stats_path)

    logging.info("Step 12b complete.")
    logging.info("  Fig 1: %s/fig1_action_distribution.png", args.report_dir)
    logging.info("  Fig 2: %s/fig2_readmission_vs_disagreement.png", args.report_dir)


if __name__ == "__main__":
    main()
