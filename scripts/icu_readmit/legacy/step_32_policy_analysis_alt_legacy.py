"""
Step 12b -- Policy analysis: Fig 1 + Fig 2 (+ Fig 3 for discharge model).

Produces Raghu-style evaluation figures for:
  A) Tier 2 drug policy (step 11b): models/icu_readmit/tier2/
  B) Tier 2 discharge-aware policy (step 11c): models/icu_readmit/tier2_discharge/

No training required -- loads pre-saved action pkl files and parquet data.

Output dirs:
  reports/icu_readmit/tier2/
  reports/icu_readmit/tier2_discharge/
"""

import json
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR    = str(PROJECT_ROOT)
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed", "icu_readmit", "legacy", "step_20_24")
MODEL_DIR   = os.path.join(BASE_DIR, "models", "icu_readmit", "legacy", "step_25_33")
REPORT_DIR  = os.path.join(BASE_DIR, "reports", "icu_readmit", "legacy", "step_30_33")

PARQUET_TIER2     = os.path.join(DATA_DIR, "rl_dataset_tier2.parquet")
PARQUET_DISCHARGE = os.path.join(DATA_DIR, "rl_dataset_tier2_discharge.parquet")
SCALER_PARAMS     = os.path.join(DATA_DIR, "scaler_params_tier2.json")

DRUG_NAMES = ["vasopressor", "ivfluid", "antibiotic", "diuretic"]

# Shock_Index severity thresholds (raw, de-normalized values)
SEV_BINS   = ["low", "medium", "high"]
SEV_THRESHOLDS = (0.6, 1.0)   # low < 0.6, medium 0.6-1.0, high > 1.0
SEV_COLORS = {"low": "#2196F3", "medium": "#FF9800", "high": "#F44336"}

DISCHARGE_LABELS = {0: "Home", 1: "Home+Services", 2: "Institutional"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def decode_4bit(actions_int):
    """Integer action -> (n, 4) binary drug array. vasopressor=bit0, ivfluid=bit1, antibiotic=bit2, diuretic=bit3."""
    arr = np.asarray(actions_int, dtype=np.int32)
    return ((arr[:, None] >> np.arange(4)) & 1).astype(np.float32)


def shock_severity(shock_raw):
    """Assign severity label per timestep from de-normalized Shock_Index values."""
    labels = np.where(shock_raw < SEV_THRESHOLDS[0], "low",
             np.where(shock_raw <= SEV_THRESHOLDS[1], "medium", "high"))
    return labels


def denormalize_shock(s_shock, scaler):
    return s_shock * scaler["std"] + scaler["mean"]


# ---------------------------------------------------------------------------
# Figure 1: Drug distribution by severity
# ---------------------------------------------------------------------------

def plot_fig1(phys_bin, ddqn_bin, sarsa_bin, sev_labels, title, save_path):
    """
    Bar chart: physician vs DDQN vs SARSA drug usage rates per severity tier.
    3 panels (low / medium / high), 4 drug bars each.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    x     = np.arange(len(DRUG_NAMES))
    width = 0.25

    fig1_stats = {}

    for ax, sev in zip(axes, SEV_BINS):
        mask = sev_labels == sev
        n    = mask.sum()

        phys_rates  = phys_bin[mask].mean(axis=0)  * 100
        ddqn_rates  = ddqn_bin[mask].mean(axis=0)  * 100
        sarsa_rates = sarsa_bin[mask].mean(axis=0) * 100

        ax.bar(x - width,     phys_rates,  width, label="Physician", color="#455A64", alpha=0.85)
        ax.bar(x,             ddqn_rates,  width, label="DDQN",      color=SEV_COLORS[sev], alpha=0.85)
        ax.bar(x + width,     sarsa_rates, width, label="SARSA",     color=SEV_COLORS[sev], alpha=0.5,
               hatch="//", edgecolor="black", linewidth=0.5)

        ax.set_title(f"Shock_Index: {sev.upper()}  (n={n:,})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(DRUG_NAMES, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("% timesteps drug active" if sev == "low" else "")
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        fig1_stats[sev] = {
            "n_timesteps": int(n),
            "physician": {d: float(r) for d, r in zip(DRUG_NAMES, phys_rates)},
            "ddqn":      {d: float(r) for d, r in zip(DRUG_NAMES, ddqn_rates)},
            "sarsa":     {d: float(r) for d, r in zip(DRUG_NAMES, sarsa_rates)},
        }

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig 1 saved: {save_path}")
    return fig1_stats


# ---------------------------------------------------------------------------
# Figure 2: Readmission rate vs Hamming distance
# ---------------------------------------------------------------------------

def plot_fig2(phys_bin, ddqn_bin, readmit, sev_labels, title, save_path):
    """
    Raghu Fig 2 equivalent.
    X-axis: number of drugs where DDQN and physician disagree (Hamming distance, 0-4).
    Y-axis: 30-day readmission rate (%).
    Hypothesis: disagreement=0 (physician agrees with DDQN) -> lowest readmission.
    """
    hamming = (phys_bin != ddqn_bin).sum(axis=1)   # 0-4 per timestep

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    fig2_stats = {}

    for ax, sev in zip(axes, SEV_BINS):
        mask   = sev_labels == sev
        h_sev  = hamming[mask]
        r_sev  = readmit[mask]

        bin_rates = []
        bin_ns    = []
        for d in range(5):
            bm    = h_sev == d
            n_bin = bm.sum()
            rate  = r_sev[bm].mean() * 100 if n_bin > 0 else np.nan
            bin_rates.append(rate)
            bin_ns.append(int(n_bin))

        bin_rates = np.array(bin_rates)
        valid = ~np.isnan(bin_rates)

        ax.bar(np.arange(5)[valid], bin_rates[valid],
               color=SEV_COLORS[sev], alpha=0.8, edgecolor="black", linewidth=0.5)

        for d in range(5):
            if valid[d] and bin_ns[d] > 0:
                ax.text(d, bin_rates[d] + 0.3, f"n={bin_ns[d]:,}",
                        ha="center", va="bottom", fontsize=7, rotation=45)

        ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2, alpha=0.5)
        ax.set_title(f"Shock_Index: {sev.upper()}", fontsize=10)
        ax.set_xlabel("Drugs where DDQN and physician disagree (Hamming)")
        ax.set_ylabel("30-day readmission rate (%)" if sev == "low" else "")
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"{d} drug{'s' if d != 1 else ''}" for d in range(5)],
                           rotation=15, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        fig2_stats[sev] = {
            "hamming_bins": {
                str(d): {"readmission_pct": float(bin_rates[d]) if valid[d] else None,
                         "n_timesteps": bin_ns[d]}
                for d in range(5)
            }
        }

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig 2 saved: {save_path}")
    return fig2_stats


# ---------------------------------------------------------------------------
# Figure 3: Discharge destination distribution (tier2_discharge only)
# ---------------------------------------------------------------------------

def plot_fig3(phys_discharge, ddqn_discharge, sarsa_discharge, save_path):
    """
    Bar chart comparing physician vs DDQN vs SARSA discharge destination.
    Categories: 0=Home, 1=Home+Services, 2=Institutional.
    """
    cats   = [0, 1, 2]
    labels = [DISCHARGE_LABELS[c] for c in cats]
    x      = np.arange(len(cats))
    width  = 0.25

    def pcts(arr):
        total = len(arr)
        return [100 * (arr == c).sum() / total for c in cats]

    phys_p  = pcts(phys_discharge)
    ddqn_p  = pcts(ddqn_discharge)
    sarsa_p = pcts(sarsa_discharge)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width,    phys_p,  width, label="Physician (observed)", color="#455A64", alpha=0.85)
    ax.bar(x,            ddqn_p,  width, label="DDQN policy",          color="#E53935", alpha=0.85)
    ax.bar(x + width,    sarsa_p, width, label="SARSA policy",          color="#FB8C00", alpha=0.85)

    ax.set_title("Fig 3 -- Discharge Destination: Physician vs DDQN vs SARSA\n"
                 "(Step 11c: discharge-aware two-phase MDP)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("% of discharge decisions")
    ax.set_ylim(0, 70)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for i, (pp, dp, sp) in enumerate(zip(phys_p, ddqn_p, sarsa_p)):
        ax.text(i - width, pp + 0.5, f"{pp:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.text(i,         dp + 0.5, f"{dp:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.text(i + width, sp + 0.5, f"{sp:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Fig 3 saved: {save_path}")

    return {
        "physician": {DISCHARGE_LABELS[c]: float(p) for c, p in zip(cats, phys_p)},
        "ddqn":      {DISCHARGE_LABELS[c]: float(p) for c, p in zip(cats, ddqn_p)},
        "sarsa":     {DISCHARGE_LABELS[c]: float(p) for c, p in zip(cats, sarsa_p)},
    }


# ---------------------------------------------------------------------------
# Run analysis for one run
# ---------------------------------------------------------------------------

def run_analysis(label, parquet_path, ddqn_action_pkl, sarsa_action_pkl,
                 scaler, report_dir,
                 # discharge-specific (optional)
                 ddqn_discharge_pkl=None, sarsa_discharge_pkl=None,
                 discharge_parquet_path=None):

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    os.makedirs(report_dir, exist_ok=True)

    # Load parquet, test split
    df_test = pd.read_parquet(parquet_path, engine="pyarrow")
    df_test = df_test[df_test["split"] == "test"].reset_index(drop=True)
    print(f"  Test rows: {len(df_test):,}")

    # Severity from Shock_Index
    shock_raw  = denormalize_shock(df_test["s_Shock_Index"].values, scaler["Shock_Index"])
    sev_labels = shock_severity(shock_raw)
    for sev in SEV_BINS:
        print(f"    {sev}: {(sev_labels == sev).sum():,} timesteps "
              f"({100*(sev_labels==sev).mean():.1f}%)")

    # Physician actions (from individual binary columns -- always correct)
    phys_bin = np.stack([
        df_test["vasopressor_b"].values,
        df_test["ivfluid_b"].values,
        df_test["antibiotic_b"].values,
        df_test["diuretic_b"].values,
    ], axis=1).astype(np.float32)

    # Load model actions
    # For tier2_discharge: pkl covers all phases in test split; slice to phase=0 only
    # For tier2: pkl already covers only the drug-phase rows (no phase column)
    ddqn_actions_raw  = np.asarray(load_pkl(ddqn_action_pkl))
    sarsa_actions_raw = np.asarray(load_pkl(sarsa_action_pkl))

    if len(ddqn_actions_raw) != len(df_test):
        # Discharge parquet case: pkl has phase=0 + phase=1, take first len(df_test) rows
        assert len(ddqn_actions_raw) > len(df_test), (
            f"DDQN action count {len(ddqn_actions_raw)} < test rows {len(df_test)} -- unexpected"
        )
        ddqn_actions  = ddqn_actions_raw[:len(df_test)]
        sarsa_actions = sarsa_actions_raw[:len(df_test)]
        print(f"  (sliced drug pkl from {len(ddqn_actions_raw)} to {len(ddqn_actions)} -- phase=0 only)")
    else:
        ddqn_actions  = ddqn_actions_raw
        sarsa_actions = sarsa_actions_raw

    ddqn_bin  = decode_4bit(ddqn_actions)
    sarsa_bin = decode_4bit(sarsa_actions)

    readmit = df_test["readmit_30d"].values.astype(float)
    print(f"  Readmission rate (test): {readmit.mean():.1%}")

    # Action agreement
    ddqn_agree  = (ddqn_actions  == df_test["a"].values).mean()
    sarsa_agree = (sarsa_actions == df_test["a"].values).mean()
    print(f"  Action agreement -- DDQN: {ddqn_agree:.1%}  SARSA: {sarsa_agree:.1%}")

    # Fig 1
    fig1_stats = plot_fig1(
        phys_bin, ddqn_bin, sarsa_bin, sev_labels,
        title=f"Fig 1 -- Drug Usage: Physician vs DDQN vs SARSA by Shock_Index Severity\n({label})",
        save_path=os.path.join(report_dir, "fig1_drug_distribution.png"),
    )

    # Fig 2
    fig2_stats = plot_fig2(
        phys_bin, ddqn_bin, readmit, sev_labels,
        title=(f"Fig 2 -- 30-Day Readmission vs DDQN-Physician Disagreement\n"
               f"(Raghu Fig 2 equivalent -- {label})"),
        save_path=os.path.join(report_dir, "fig2_readmission_vs_disagreement.png"),
    )

    results = {
        "label": label,
        "n_test": int(len(df_test)),
        "readmission_rate": float(readmit.mean()),
        "ddqn_agreement":   float(ddqn_agree),
        "sarsa_agreement":  float(sarsa_agree),
        "severity_counts":  {sev: int((sev_labels == sev).sum()) for sev in SEV_BINS},
        "fig1": fig1_stats,
        "fig2": fig2_stats,
    }

    # Fig 3: discharge distribution (only for tier2_discharge)
    if ddqn_discharge_pkl and sarsa_discharge_pkl and discharge_parquet_path:
        # The discharge parquet test split has both phase=0 and phase=1 rows.
        # The pkl files were saved over the full test split (both phases combined).
        # We need to slice out the phase=1 portion.
        df_disc_full = pd.read_parquet(discharge_parquet_path, engine="pyarrow")
        df_disc_full = df_disc_full[df_disc_full["split"] == "test"].reset_index(drop=True)

        phase0_mask = df_disc_full["phase"].values == 0
        phase1_mask = df_disc_full["phase"].values == 1
        n_phase0 = phase0_mask.sum()
        n_phase1 = phase1_mask.sum()
        print(f"  Discharge parquet test: phase=0 {n_phase0:,} rows, phase=1 {n_phase1:,} rows")

        ddqn_disc_all  = load_pkl(ddqn_discharge_pkl)
        sarsa_disc_all = load_pkl(sarsa_discharge_pkl)
        total_test = n_phase0 + n_phase1
        assert len(ddqn_disc_all)  == total_test, f"Discharge DDQN count {len(ddqn_disc_all)} != {total_test}"
        assert len(sarsa_disc_all) == total_test, f"Discharge SARSA count {len(sarsa_disc_all)} != {total_test}"

        # Slice to phase=1 rows only
        ddqn_disc  = np.asarray(ddqn_disc_all)[phase1_mask]
        sarsa_disc = np.asarray(sarsa_disc_all)[phase1_mask]
        phys_disc  = df_disc_full.loc[phase1_mask, "a"].values  # physician's discharge action
        print(f"  Discharge rows (phase=1): {len(phys_disc):,}")

        fig3_stats = plot_fig3(
            phys_disc, ddqn_disc, sarsa_disc,
            save_path=os.path.join(report_dir, "fig3_discharge_distribution.png"),
        )
        results["fig3_discharge"] = fig3_stats

    # Save results JSON
    results_path = os.path.join(report_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_path}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scaler = json.load(open(SCALER_PARAMS))

    # ── A: Tier 2 drug policy (step 11b) ────────────────────────────────────
    run_analysis(
        label="Tier 2 -- FCI drug policy (step 11b)",
        parquet_path=PARQUET_TIER2,
        ddqn_action_pkl  = os.path.join(MODEL_DIR, "tier2", "ddqn",       "dqn_actions_test.pkl"),
        sarsa_action_pkl = os.path.join(MODEL_DIR, "tier2", "sarsa_phys", "phys_actions_test.pkl"),
        scaler=scaler,
        report_dir=os.path.join(REPORT_DIR, "tier2"),
    )

    # ── B: Tier 2 discharge-aware policy (step 11c) ─────────────────────────
    run_analysis(
        label="Tier 2 Discharge -- two-phase MDP (step 11c)",
        parquet_path=PARQUET_TIER2,   # drug policy evaluated on same in-stay parquet
        ddqn_action_pkl  = os.path.join(MODEL_DIR, "tier2_discharge", "ddqn",       "drug_actions_test.pkl"),
        sarsa_action_pkl = os.path.join(MODEL_DIR, "tier2_discharge", "sarsa_phys", "phys_drug_actions_test.pkl"),
        scaler=scaler,
        report_dir=os.path.join(REPORT_DIR, "tier2_discharge"),
        # discharge analysis
        ddqn_discharge_pkl  = os.path.join(MODEL_DIR, "tier2_discharge", "ddqn",       "discharge_actions_test.pkl"),
        sarsa_discharge_pkl = os.path.join(MODEL_DIR, "tier2_discharge", "sarsa_phys", "phys_discharge_actions_test.pkl"),
        discharge_parquet_path=PARQUET_DISCHARGE,
    )

    print("\nDone. All figures saved to reports/icu_readmit/")
