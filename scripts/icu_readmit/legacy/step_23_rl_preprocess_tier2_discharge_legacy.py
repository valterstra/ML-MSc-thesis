"""
Step 10d -- RL preprocessing: Tier 2 + discharge terminal action.

PURPOSE
-------
Extends rl_dataset_tier2.parquet with a discharge terminal step for each
includable stay, enabling the joint DDQN (step 11c) to learn both in-ICU
drug decisions AND discharge destination jointly via cross-phase Bellman backup.

TWO-PHASE MDP
-------------
  Phase 0 (in-stay blocs):
      state  = (Hb, BUN, Creatinine, HR, Shock_Index)   -- FCI Tier 2
      action = drug combo integer 0-15 (2^4)
      reward = SOFA delta (dense)

  Phase 1 (discharge terminal, one row per includable stay):
      state  = final in-stay state (same 5 features)
      action = discharge category integer 0-2
      reward = +/-TERMINAL_REWARD based on readmit_30d

DISCHARGE CATEGORIES
--------------------
  0 -- Home independent   HOME, ASSISTED LIVING                 24.2%
  1 -- Home with services HOME HEALTH CARE                      31.3%
  2 -- Institutional      SNF, REHAB, LTAC, PSYCH, HEALTHCARE   40.7%

  Excluded (~3.7%): ACUTE HOSPITAL, HOSPICE, AGAINST ADVICE,
                    OTHER FACILITY, NaN, DIED
  Excluded stays still contribute to drug-policy training; they keep
  their original done=1 / r=+-15 terminal transition.

MODIFICATIONS TO LAST IN-STAY BLOC (includable stays only)
-----------------------------------------------------------
  done          : 1  -->  0          (not terminal; transitions to discharge)
  r             : +-15 --> 0         (readmission signal moved to discharge row)
  s_next_*      : 0.0 --> s_*        (next state = current state; same features)
  next_is_discharge: False --> True

NEW COLUMNS IN OUTPUT
---------------------
  phase              int   0=in-stay, 1=discharge terminal
  next_is_discharge  int   1 for last bloc of includable stays, else 0
  next_a_physician   int   observed next action (precomputed for SARSA baseline)
                           = next drug action for regular blocs
                           = discharge category for last in-stay bloc
                           = 0 for discharge rows / excluded last blocs

OUTPUTS
-------
  data/processed/icu_readmit/rl_dataset_tier2_discharge.parquet
  data/processed/icu_readmit/discharge_stats.json

Usage:
    python scripts/icu_readmit/step_10d_rl_preprocess_discharge.py --smoke
    python scripts/icu_readmit/step_10d_rl_preprocess_discharge.py
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.columns import C_ICUSTAYID, C_BLOC, C_READMIT_30D

# ---------------------------------------------------------------------------
# Discharge category mapping
# ---------------------------------------------------------------------------

DISCHARGE_MAP = {
    "HOME":                         0,
    "ASSISTED LIVING":              0,
    "HOME HEALTH CARE":             1,
    "SKILLED NURSING FACILITY":     2,
    "REHAB":                        2,
    "CHRONIC/LONG TERM ACUTE CARE": 2,
    "PSYCH FACILITY":               2,
    "HEALTHCARE FACILITY":          2,
}

DISCHARGE_LABELS = {0: "Home", 1: "Home+Services", 2: "Institutional"}
N_DISCHARGE      = 3
TERMINAL_REWARD  = 15.0

STATE_COLS = ["s_Hb", "s_BUN", "s_Creatinine", "s_HR", "s_Shock_Index"]
NEXT_COLS  = ["s_next_Hb", "s_next_BUN", "s_next_Creatinine", "s_next_HR", "s_next_Shock_Index"]
DRUG_COLS  = ["vasopressor_b", "ivfluid_b", "antibiotic_b", "diuretic_b"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_discharge_map(icu_csv: str, stay_ids: np.ndarray) -> pd.Series:
    """Load discharge_disposition from ICUdataset.csv, return category Series."""
    logging.info("Loading discharge_disposition from %s ...", icu_csv)
    raw = pd.read_csv(icu_csv, usecols=[C_ICUSTAYID, "discharge_disposition"])
    raw = raw[raw[C_ICUSTAYID].isin(stay_ids)].drop_duplicates(C_ICUSTAYID)
    raw["disch_cat"] = raw["discharge_disposition"].map(DISCHARGE_MAP)   # NaN for excluded
    cat_series = raw.set_index(C_ICUSTAYID)["disch_cat"]
    logging.info("  %d stays loaded, %d includable (mapped), %d excluded",
                 len(raw),
                 raw["disch_cat"].notna().sum(),
                 raw["disch_cat"].isna().sum())
    return cat_series


def log_category_distribution(icu_csv: str, stay_ids: np.ndarray) -> None:
    raw = pd.read_csv(icu_csv, usecols=[C_ICUSTAYID, "discharge_disposition"])
    raw = raw[raw[C_ICUSTAYID].isin(stay_ids)].drop_duplicates(C_ICUSTAYID)
    total = len(raw)
    vc    = raw["discharge_disposition"].value_counts(dropna=False)
    logging.info("Discharge location distribution (stay-level):")
    for val, cnt in vc.items():
        cat = DISCHARGE_MAP.get(str(val), "EXCLUDED")
        logging.info("  %-45s %5d (%4.1f%%)  cat=%s",
                     str(val), cnt, 100 * cnt / total, cat)


# ---------------------------------------------------------------------------
# Core build
# ---------------------------------------------------------------------------

def build_discharge_dataset(df: pd.DataFrame, cat_series: pd.Series) -> pd.DataFrame:
    """
    Modify last in-stay blocs and append discharge rows for includable stays.

    Args:
        df:         rl_dataset_tier2 DataFrame (all phases, pre-sorted)
        cat_series: Series indexed by icustayid, values in {0,1,2} or NaN

    Returns:
        Combined DataFrame with phase, next_is_discharge, next_a_physician columns.
    """
    df = df.copy()

    # Map discharge category to each row
    df["disch_cat"] = df[C_ICUSTAYID].map(cat_series)
    includable      = df["disch_cat"].notna()

    # ---- Identify last in-stay bloc per stay --------------------------------
    last_bloc_idx = df.groupby(C_ICUSTAYID)[C_BLOC].idxmax()
    is_last       = df.index.isin(last_bloc_idx.values)

    # ---- Initialise new columns on all rows ---------------------------------
    df["phase"]             = 0
    df["next_is_discharge"] = 0

    # ---- Modify last in-stay blocs for includable stays --------------------
    mod_mask = is_last & includable

    # Demote: no longer terminal -- transitions to discharge step
    df.loc[mod_mask, "done"] = 0

    # Remove readmission terminal reward; will appear in discharge row instead
    df.loc[mod_mask, "r"] = 0.0

    # Set s_next = s (same state; discharge step operates on the same snapshot)
    for s_col, sn_col in zip(STATE_COLS, NEXT_COLS):
        df.loc[mod_mask, sn_col] = df.loc[mod_mask, s_col].values

    df.loc[mod_mask, "next_is_discharge"] = 1

    # ---- Build discharge rows (one per includable stay) --------------------
    last_rows = df.loc[mod_mask].copy()

    # Discharge row state = same as last in-stay state (already set above)
    disch_rows = last_rows.copy()

    # Action = discharge category
    disch_rows["a"]          = last_rows["disch_cat"].astype(int)
    disch_rows[C_BLOC]       = last_rows[C_BLOC] + 1    # sort after last bloc
    disch_rows["phase"]      = 1
    disch_rows["next_is_discharge"] = 0

    # Drug flags zeroed (not a drug decision)
    for col in DRUG_COLS:
        if col in disch_rows.columns:
            disch_rows[col] = 0

    # Terminal reward at discharge
    disch_rows["r"] = np.where(
        disch_rows[C_READMIT_30D] == 0,
         TERMINAL_REWARD,
        -TERMINAL_REWARD,
    )
    disch_rows["done"] = 1

    # s_next = zeros (terminal)
    for sn_col in NEXT_COLS:
        disch_rows[sn_col] = 0.0

    # ---- Concatenate -------------------------------------------------------
    df_all = pd.concat([df, disch_rows], ignore_index=True)
    df_all = df_all.sort_values([C_ICUSTAYID, "phase", C_BLOC]).reset_index(drop=True)
    df_all.drop(columns=["disch_cat"], inplace=True)

    # ---- Precompute next_a_physician ----------------------------------------
    # Within each stay (sorted: in-stay blocs first, discharge last):
    #   - Regular bloc: next row's drug action
    #   - Last in-stay (next_is_discharge=1): next row = discharge row, so a = discharge cat
    #   - Discharge row: 0 (terminal, irrelevant)
    #   - Excluded last bloc (done=1, no discharge row): 0
    df_all["next_a_physician"] = (
        df_all.groupby(C_ICUSTAYID)["a"]
              .shift(-1)
              .fillna(0)
              .astype(int)
    )

    return df_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tier2-parquet", default=str(
        PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "legacy" / "step_20_24" / "rl_dataset_tier2.parquet"))
    parser.add_argument("--icu-csv", default=str(
        PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "ICUdataset.csv"))
    parser.add_argument("--out-dir", default=str(
        PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "legacy" / "step_20_24"))
    parser.add_argument("--smoke", action="store_true",
                        help="Run on first 2000 stays only")
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    log_file = args.log or str(PROJECT_ROOT / "logs" / "legacy" / "icu_readmit" / "step_23_rl_preprocess_tier2_discharge_legacy.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 10d started. smoke=%s", args.smoke)

    # -- Load Tier 2 parquet --------------------------------------------------
    logging.info("Loading %s ...", args.tier2_parquet)
    df = pd.read_parquet(args.tier2_parquet)
    logging.info("  %d rows, %d cols, %d stays",
                 len(df), len(df.columns), df[C_ICUSTAYID].nunique())

    if args.smoke:
        smoke_stays = np.sort(df[C_ICUSTAYID].unique())[:2000]
        df = df[df[C_ICUSTAYID].isin(smoke_stays)].copy()
        logging.info("Smoke mode: %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    stay_ids = df[C_ICUSTAYID].unique()

    # -- Log discharge distribution (full stays, not smoke-filtered) ----------
    log_category_distribution(args.icu_csv, stay_ids)

    # -- Load discharge categories -------------------------------------------
    cat_series = load_discharge_map(args.icu_csv, stay_ids)

    # Validation: check all required columns exist
    for col in STATE_COLS + NEXT_COLS + ["done", "r", "a", C_READMIT_30D, "split"]:
        if col not in df.columns:
            raise ValueError(f"Required column missing from tier2 parquet: {col}")

    # -- Build dataset --------------------------------------------------------
    logging.info("Building discharge-augmented dataset...")
    df_out = build_discharge_dataset(df, cat_series)

    # -- Validation -----------------------------------------------------------
    n_stays    = df_out[C_ICUSTAYID].nunique()
    n_in_stay  = (df_out["phase"] == 0).sum()
    n_discharge= (df_out["phase"] == 1).sum()

    # Every includable stay should have exactly one discharge row
    includable_stays = cat_series[cat_series.notna()].index
    includable_stays = set(includable_stays) & set(stay_ids)
    n_expected_disch = len(includable_stays)
    if n_discharge != n_expected_disch:
        logging.warning("Discharge row count mismatch: got %d, expected %d",
                        n_discharge, n_expected_disch)

    # next_is_discharge should equal n_discharge
    n_nid = (df_out["next_is_discharge"] == 1).sum()
    if n_nid != n_discharge:
        logging.warning("next_is_discharge count %d != discharge rows %d", n_nid, n_discharge)

    # No NaN in key columns
    for col in STATE_COLS + NEXT_COLS + ["a", "r", "done", "phase", "next_a_physician"]:
        if df_out[col].isna().any():
            logging.warning("NaN found in column %s", col)

    logging.info("Dataset summary:")
    logging.info("  Total rows:           %d", len(df_out))
    logging.info("  Phase 0 (in-stay):    %d", n_in_stay)
    logging.info("  Phase 1 (discharge):  %d  (%d includable stays)", n_discharge, n_expected_disch)
    logging.info("  Excluded stays:       %d  (keep done=1 terminal in-stay row)",
                 n_stays - n_expected_disch)

    for spl in ("train", "val", "test"):
        sub = df_out[df_out["split"] == spl]
        n_d = (sub["phase"] == 1).sum()
        logging.info("  Split %-5s  %d rows  %d stays  %d discharge rows",
                     spl, len(sub), sub[C_ICUSTAYID].nunique(), n_d)

    # Discharge category distribution
    disch_sub = df_out[df_out["phase"] == 1]
    logging.info("Discharge action distribution (agent's action space):")
    for cat_id in range(N_DISCHARGE):
        cnt = (disch_sub["a"] == cat_id).sum()
        logging.info("  %d (%s): %d (%.1f%%)",
                     cat_id, DISCHARGE_LABELS[cat_id], cnt, 100 * cnt / len(disch_sub))

    # Reward check
    phase0_r = df_out[df_out["phase"] == 0]["r"]
    phase1_r = df_out[df_out["phase"] == 1]["r"]
    logging.info("Reward phase=0: mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
                 phase0_r.mean(), phase0_r.std(), phase0_r.min(), phase0_r.max())
    logging.info("Reward phase=1: +15 count=%d  -15 count=%d",
                 (phase1_r > 0).sum(), (phase1_r < 0).sum())

    # -- Save -----------------------------------------------------------------
    suffix    = "_tier2_discharge_smoke" if args.smoke else "_tier2_discharge"
    out_path  = os.path.join(args.out_dir, f"rl_dataset{suffix}.parquet")
    stat_path = os.path.join(args.out_dir, f"discharge_stats{'' if not args.smoke else '_smoke'}.json")

    df_out.to_parquet(out_path, index=False)
    logging.info("Saved %s  (%d rows, %d cols)", out_path, len(df_out), len(df_out.columns))

    stats = {
        "n_rows_total":    int(len(df_out)),
        "n_rows_phase0":   int(n_in_stay),
        "n_rows_phase1":   int(n_discharge),
        "n_stays_total":   int(n_stays),
        "n_stays_included": int(n_expected_disch),
        "n_stays_excluded": int(n_stays - n_expected_disch),
        "discharge_counts": {
            DISCHARGE_LABELS[i]: int((disch_sub["a"] == i).sum())
            for i in range(N_DISCHARGE)
        },
    }
    with open(stat_path, "w") as f:
        json.dump(stats, f, indent=2)
    logging.info("Saved stats -> %s", stat_path)
    logging.info("Step 10d complete.")
