"""
Preprocessing for sepsis RL pipeline.
Ported from sepsisrl/preprocessing/ notebooks.

Handles: action discretization, normalization, train/val/test splitting,
reward construction.
"""
import logging
import numpy as np
import pandas as pd

# ── 48 state features used for clustering and RL ──────────────────────
# Matches sepsisrl/data/state_features.txt exactly (alphabetical)
STATE_FEATURES = [
    "Albumin", "Arterial_BE", "Arterial_lactate", "Arterial_pH",
    "BUN", "CO2_mEqL", "Calcium", "Chloride", "Creatinine",
    "DiaBP", "FiO2_1", "GCS", "Glucose", "HCO3", "HR", "Hb",
    "INR", "Ionised_Ca", "Magnesium", "MeanBP", "PT", "PTT",
    "PaO2_FiO2", "Platelets_count", "Potassium", "RR",
    "SGOT", "SGPT", "SIRS", "SOFA", "Shock_Index", "Sodium",
    "SpO2", "SysBP", "Temp_C", "Total_bili", "WBC_count",
    "Weight_kg", "age", "elixhauser", "gender", "mechvent",
    "output_4hourly", "output_total", "paCO2", "paO2",
    "re_admission", "bloc",
]

# 42 predictive features (exclude mortality, age, etc. for some models)
STATE_FEATURES_PRED = [
    "Albumin", "Arterial_BE", "Arterial_lactate", "Arterial_pH",
    "BUN", "CO2_mEqL", "Calcium", "Chloride", "Creatinine",
    "DiaBP", "FiO2_1", "GCS", "Glucose", "HCO3", "HR", "Hb",
    "INR", "Ionised_Ca", "Magnesium", "MeanBP", "PT", "PTT",
    "PaO2_FiO2", "Platelets_count", "Potassium", "RR",
    "SGOT", "SGPT", "SIRS", "SOFA", "Shock_Index", "Sodium",
    "SpO2", "SysBP", "Temp_C", "Total_bili", "WBC_count",
    "Weight_kg", "mechvent", "output_4hourly", "output_total",
    "paCO2",
]

# ── Normalization groups ──────────────────────────────────────────────
# Binary fields: subtract 0.5
BINARY_FIELDS = ["gender", "mechvent", "re_admission"]

# Gaussian fields: z-standardize (mean/std from train set)
NORM_FIELDS = [
    "age", "Weight_kg", "GCS", "HR", "SysBP", "MeanBP", "DiaBP",
    "RR", "Temp_C", "FiO2_1", "Potassium", "Sodium", "Chloride",
    "Glucose", "Magnesium", "Calcium", "Hb", "WBC_count",
    "Platelets_count", "PTT", "PT", "Arterial_pH", "paO2", "paCO2",
    "Arterial_BE", "HCO3", "Arterial_lactate", "SOFA", "SIRS",
    "Shock_Index", "PaO2_FiO2", "cumulated_balance", "elixhauser",
    "Albumin", "CO2_mEqL", "Ionised_Ca",
]

# Log-normal fields: log(0.1 + x) then z-standardize
LOG_FIELDS = [
    "max_dose_vaso", "SpO2", "BUN", "Creatinine", "SGOT", "SGPT",
    "Total_bili", "INR", "input_total", "input_4hourly_tev",
    "output_total", "output_4hourly", "bloc",
]

# ── Columns to keep in output ────────────────────────────────────────
# Metadata + state features + actions + outcome
OUTPUT_COLS_META = ["bloc", "icustayid"]
OUTPUT_COLS_OUTCOME = ["died_in_hosp", "morta_90", "readmit_30d"]
OUTPUT_COLS_ACTION = [
    "median_dose_vaso", "max_dose_vaso",
    "input_total", "input_4hourly_tev",
    "output_total", "output_4hourly", "cumulated_balance",
    "vaso_input", "iv_input",
]

N_ACTION_BINS = 5  # 0 + 4 quartile bins


def discretize_actions(df):
    """Bin vasopressor and IV fluid doses into 5 discrete levels each.

    Level 0 = no treatment, levels 1-4 = quartiles of nonzero doses.
    Returns df with new columns: vaso_input, iv_input.
    """
    df = df.copy()

    # Vasopressor: quartiles of nonzero doses
    nonzero_vaso = df["max_dose_vaso"][df["max_dose_vaso"] > 0]
    vaso_q = np.percentile(nonzero_vaso.dropna(), [25, 50, 75])
    logging.info("Vaso quartiles: %s", vaso_q)

    df["vaso_input"] = 0
    mask_nz = df["max_dose_vaso"] > 0
    df.loc[mask_nz, "vaso_input"] = np.searchsorted(vaso_q, df.loc[mask_nz, "max_dose_vaso"]) + 1

    # IV fluid: quartiles of nonzero doses
    nonzero_iv = df["input_4hourly_tev"][df["input_4hourly_tev"] > 0]
    iv_q = np.percentile(nonzero_iv.dropna(), [25, 50, 75])
    logging.info("IV quartiles: %s", iv_q)

    df["iv_input"] = 0
    mask_nz = df["input_4hourly_tev"] > 0
    df.loc[mask_nz, "iv_input"] = np.searchsorted(iv_q, df.loc[mask_nz, "input_4hourly_tev"]) + 1

    # Action ID: 5 * iv_level + vaso_level (0-24)
    df["action_id"] = 5 * df["iv_input"] + df["vaso_input"]

    # Distribution summary
    for name, col in [("vaso", "vaso_input"), ("IV", "iv_input")]:
        counts = df[col].value_counts().sort_index()
        logging.info("%s bin distribution:\n%s", name, counts.to_string())

    return df, vaso_q, iv_q


def split_by_patient(df, train_frac=0.70, val_frac=0.10, seed=42):
    """Split into train/val/test by icustayid (patient-level).

    Returns (train_df, val_df, test_df).
    """
    ids = df["icustayid"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * (train_frac + val_frac))

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_val])
    test_ids = set(ids[n_val:])

    train_df = df[df["icustayid"].isin(train_ids)].copy()
    val_df = df[df["icustayid"].isin(val_ids)].copy()
    test_df = df[df["icustayid"].isin(test_ids)].copy()

    logging.info(
        "Split: train=%d rows (%d stays), val=%d rows (%d stays), test=%d rows (%d stays)",
        len(train_df), len(train_ids),
        len(val_df), len(val_ids),
        len(test_df), len(test_ids),
    )
    return train_df, val_df, test_df


def normalize(train_df, val_df, test_df):
    """Normalize features following sepsisrl conventions.

    Binary: subtract 0.5
    Gaussian: z-standardize using train statistics
    Log-normal: log(0.1 + x) then z-standardize using train statistics

    Returns (train, val, test, norm_stats) where norm_stats is a dict
    with mean/std for reproducibility.
    """
    norm_stats = {}

    # Binary fields
    for col in BINARY_FIELDS:
        if col in train_df.columns:
            for df in [train_df, val_df, test_df]:
                df[col] = df[col].astype(float) - 0.5

    # Gaussian fields
    for col in NORM_FIELDS:
        if col not in train_df.columns:
            logging.warning("Norm field %s not found, skipping", col)
            continue
        mean = train_df[col].astype(float).mean()
        std = train_df[col].astype(float).std()
        if std == 0:
            std = 1.0
        norm_stats[col] = {"mean": mean, "std": std, "transform": "zscore"}
        for df in [train_df, val_df, test_df]:
            df[col] = (df[col].astype(float) - mean) / std

    # Log-normal fields
    for col in LOG_FIELDS:
        if col not in train_df.columns:
            logging.warning("Log field %s not found, skipping", col)
            continue
        # Apply log transform first
        for df in [train_df, val_df, test_df]:
            df[col] = np.log(0.1 + df[col].astype(float))
        # Then z-standardize
        mean = train_df[col].mean()
        std = train_df[col].std()
        if std == 0:
            std = 1.0
        norm_stats[col] = {"mean": mean, "std": std, "transform": "log_zscore"}
        for df in [train_df, val_df, test_df]:
            df[col] = (df[col] - mean) / std

    return train_df, val_df, test_df, norm_stats


def scale_01(train_df, val_df, test_df, feature_cols):
    """Scale features to [0, 1] using train set min/max.

    Returns (train, val, test, scale_stats).
    """
    scale_stats = {}
    for col in feature_cols:
        if col not in train_df.columns:
            continue
        col_min = train_df[col].min()
        col_max = train_df[col].max()
        col_range = col_max - col_min
        if col_range == 0:
            col_range = 1.0
        scale_stats[col] = {"min": col_min, "max": col_max}
        for df in [train_df, val_df, test_df]:
            df[col] = (df[col] - col_min) / col_range

    return train_df, val_df, test_df, scale_stats


def add_sparse_reward(df, outcome="mortality"):
    """Add sparse terminal reward to the last row of each ICU stay.

    outcome="mortality" (default, original pipeline):
      died_in_hosp=1  -> -100
      died_in_hosp=0  -> +100

    outcome="readmit" (readmission variant):
      died_in_hosp=1                          -> -100  (death is still a failure)
      died_in_hosp=0 AND readmit_30d=1        -> -100  (survived but readmitted)
      died_in_hosp=0 AND readmit_30d=0        -> +100  (survived and not readmitted)

    All non-terminal rows get reward = 0.
    """
    df = df.copy()
    df["reward"] = 0.0

    stay_ends = df.groupby("icustayid").tail(1).index

    if outcome == "mortality":
        died_mask = df.loc[stay_ends, "died_in_hosp"] == 1
        survived_mask = df.loc[stay_ends, "died_in_hosp"] == 0
        df.loc[stay_ends[died_mask], "reward"] = -100.0
        df.loc[stay_ends[survived_mask], "reward"] = 100.0
        logging.info(
            "Terminal rewards (mortality): %d died (-100), %d survived (+100)",
            died_mask.sum(), survived_mask.sum(),
        )

    elif outcome == "readmit":
        ends = df.loc[stay_ends]
        died_mask      = ends["died_in_hosp"] == 1
        readmit_mask   = (ends["died_in_hosp"] == 0) & (ends["readmit_30d"] == 1)
        good_mask      = (ends["died_in_hosp"] == 0) & (ends["readmit_30d"] == 0)
        df.loc[stay_ends[died_mask],    "reward"] = -100.0
        df.loc[stay_ends[readmit_mask], "reward"] = -100.0
        df.loc[stay_ends[good_mask],    "reward"] = 100.0
        logging.info(
            "Terminal rewards (readmit): %d died (-100), %d readmitted (-100), %d good (+100)",
            died_mask.sum(), readmit_mask.sum(), good_mask.sum(),
        )

    else:
        raise ValueError("outcome must be 'mortality' or 'readmit', got: %s" % outcome)

    return df


def add_shaped_reward(df, orig_df, include_terminal=True, outcome="mortality"):
    """Add SOFA/lactate shaped reward (from new_rewards.ipynb).

    Reward shaping (same for all outcome variants):
      c0 = -0.025 penalty for SOFA stasis (nonzero SOFA unchanged)
      c1 = -0.125 * (SOFA_next - SOFA_prev)  (penalize SOFA increase)
      c2 = -2.0 * tanh(lactate_next - lactate_prev)  (penalize lactate increase)

    Terminal (if include_terminal=True):
      outcome="mortality": +100 survive, -100 die
      outcome="readmit":   -100 die, -100 survive+readmitted, +100 survive+not readmitted
    Final reward clamped to [-15, +15].

    Args:
        df: normalized DataFrame (with scaled values)
        orig_df: original unscaled DataFrame (for raw SOFA/lactate)
        include_terminal: whether to include terminal rewards
        outcome: "mortality" (default) or "readmit"
    """
    c0 = -0.1 / 4   # -0.025
    c1 = -0.5 / 4   # -0.125
    c2 = -2.0

    df = df.copy()
    df["reward"] = 0.0

    sofa = orig_df["SOFA"].values
    lactate = orig_df["Arterial_lactate"].values
    icuids = orig_df["icustayid"].values
    died = orig_df["died_in_hosp"].values
    readmit = orig_df["readmit_30d"].values if outcome == "readmit" else None

    rewards = np.zeros(len(df), dtype=np.float64)

    def terminal_reward(idx):
        if not include_terminal:
            return 0.0
        if died[idx] == 1:
            return -100.0
        if outcome == "readmit" and readmit[idx] == 1:
            return -100.0
        return 100.0

    for i in range(1, len(df)):
        if icuids[i] != icuids[i - 1]:
            rewards[i - 1] = terminal_reward(i - 1)
            continue

        r = 0.0
        sofa_prev, sofa_cur = sofa[i - 1], sofa[i]
        lact_prev, lact_cur = lactate[i - 1], lactate[i]

        if sofa_cur == sofa_prev and sofa_cur != 0:
            r += c0
        r += c1 * (sofa_cur - sofa_prev)
        if not (np.isnan(lact_cur) or np.isnan(lact_prev)):
            r += c2 * np.tanh(lact_cur - lact_prev)

        rewards[i - 1] = r

    rewards[len(df) - 1] = terminal_reward(len(df) - 1)
    rewards = np.clip(rewards, -15.0, 15.0)

    df["reward"] = rewards
    return df


def select_output_columns(df):
    """Select the columns needed for RL pipeline output.

    Returns df with: metadata + 48 state features + actions + reward + outcome.
    """
    keep = []
    for col in OUTPUT_COLS_META + STATE_FEATURES + OUTPUT_COLS_ACTION + OUTPUT_COLS_OUTCOME + ["reward", "action_id"]:
        if col in df.columns:
            keep.append(col)
    # Deduplicate preserving order (bloc appears in both META and STATE_FEATURES)
    seen = set()
    unique_keep = []
    for c in keep:
        if c not in seen:
            seen.add(c)
            unique_keep.append(c)
    return df[unique_keep]
