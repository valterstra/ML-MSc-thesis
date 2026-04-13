"""Preprocessing for V3 hospital RL pipeline (model-free, offline DDQN).

Converts hosp_daily_v3_triplets.csv into normalized (s, a, r, s', done)
transitions for offline DDQN training.

State features (19):
  LOG_LABS  (4): creatinine, bun, wbc, platelets          -- log+zscore
  ZSCORE    (13): remaining 9 core labs + day_of_stay,
                  charlson_score, age_at_admit,
                  positive_culture_cumulative              -- zscore
  BINARY    (2): is_icu, gender                           -- subtract 0.5

Actions: 8 binary drug flags -> action_id in 0..255 (bitmask)

Reward:
  Dense    : weighted lab delta toward clinical normal ranges (every row)
  Terminal : +100 (readmit_30d=0) / -100 (readmit_30d=1) on last row
             per admission
  Combined : dense_reward + terminal_reward
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

CORE_LABS: list[str] = [
    "creatinine", "bun", "sodium", "potassium", "bicarbonate",
    "anion_gap", "calcium", "glucose", "hemoglobin", "wbc",
    "platelets", "phosphate", "magnesium",
]

LOG_LABS: list[str] = ["creatinine", "bun", "wbc", "platelets"]

ZSCORE_CONT: list[str] = (
    [lab for lab in CORE_LABS if lab not in LOG_LABS]
    + ["age_at_admit", "charlson_score", "day_of_stay", "positive_culture_cumulative"]
)

BINARY_FEATURES: list[str] = ["is_icu", "gender"]

# Ordered list of 19 state features
STATE_FEATURES: list[str] = LOG_LABS + ZSCORE_CONT + BINARY_FEATURES

# Mapping: feature name -> source column for next-state
# Static features reuse the current column (they do not change within an admission)
NEXT_COL_MAP: dict[str, str] = {}
for _f in CORE_LABS:
    NEXT_COL_MAP[_f] = "next_" + _f
NEXT_COL_MAP["is_icu"] = "next_is_icu"
NEXT_COL_MAP["positive_culture_cumulative"] = "next_positive_culture_cumulative"
NEXT_COL_MAP["day_of_stay"] = "_next_day_of_stay"      # computed below
NEXT_COL_MAP["age_at_admit"] = "age_at_admit"          # static
NEXT_COL_MAP["charlson_score"] = "charlson_score"      # static
NEXT_COL_MAP["gender"] = "gender"                      # static

DRUG_COLS: list[str] = [
    "antibiotic_active", "anticoagulant_active", "diuretic_active",
    "steroid_active", "insulin_active", "opioid_active",
    "electrolyte_active", "cardiovascular_active",
]

N_ACTIONS: int = 256   # 2 ** 8

# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "creatinine":  (0.6,   1.2),
    "bun":         (7.0,   20.0),
    "sodium":      (136.0, 145.0),
    "potassium":   (3.5,   5.0),
    "bicarbonate": (22.0,  28.0),
    "anion_gap":   (8.0,   16.0),
    "calcium":     (8.4,   10.2),
    "glucose":     (70.0,  180.0),
    "hemoglobin":  (12.0,  17.5),
    "wbc":         (4.5,   11.0),
    "platelets":   (150.0, 400.0),
    "phosphate":   (2.5,   4.5),
    "magnesium":   (1.7,   2.2),
}

LAB_WEIGHTS: dict[str, float] = {
    "potassium":   2.0,
    "sodium":      2.0,
    "anion_gap":   1.5,
    "bicarbonate": 1.5,
    "creatinine":  1.5,
    "glucose":     1.5,
    "platelets":   1.5,
    "wbc":         1.5,
    "bun":         1.0,
    "calcium":     1.0,
    "hemoglobin":  1.0,
    "phosphate":   1.0,
    "magnesium":   0.5,
}

TERMINAL_VALUE: float = 100.0


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------

def encode_actions(df: pd.DataFrame) -> pd.Series:
    """Encode 8 binary drug flags as a single integer 0-255 (bitmask)."""
    action_id = np.zeros(len(df), dtype=np.int32)
    for i, col in enumerate(DRUG_COLS):
        action_id += df[col].fillna(0).astype(int).values * (1 << i)
    return pd.Series(action_id, index=df.index, name="action_id")


# ---------------------------------------------------------------------------
# Dense reward (computed on RAW lab values, before normalization)
# ---------------------------------------------------------------------------

def _lab_distance(vals: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Normalized distance from clinical normal range."""
    width = hi - lo
    below = np.maximum(0.0, lo - vals) / width
    above = np.maximum(0.0, vals - hi) / width
    dist = below + above
    return np.where(np.isnan(vals), 0.0, dist)


def compute_dense_reward(df: pd.DataFrame) -> np.ndarray:
    """Weighted lab delta reward for each row (raw lab values).

    reward = sum over 13 labs: weight * clip(d_prev - d_next, -1, 1)

    Positive when labs move toward normal range, negative when away.
    Computed before any normalization so raw clinical values are used.
    """
    reward = np.zeros(len(df))
    for lab, (lo, hi) in NORMAL_RANGES.items():
        w = LAB_WEIGHTS.get(lab, 1.0)
        prev_col = lab
        next_col = "next_" + lab
        if prev_col not in df.columns or next_col not in df.columns:
            continue
        prev_vals = df[prev_col].values.astype(float)
        next_vals = df[next_col].values.astype(float)
        d_prev = _lab_distance(prev_vals, lo, hi)
        d_next = _lab_distance(next_vals, lo, hi)
        delta = d_prev - d_next
        reward += w * np.clip(delta, -1.0, 1.0)
    return reward


# ---------------------------------------------------------------------------
# Terminal identification and terminal reward
# ---------------------------------------------------------------------------

def mark_terminals(df: pd.DataFrame) -> pd.DataFrame:
    """Mark the last row per hadm_id as terminal (done=1).

    Uses max day_of_stay per admission as the terminal row.
    Adds terminal_reward: +100 if readmit_30d=0, -100 if readmit_30d=1.
    """
    df = df.copy()
    df["done"] = 0
    df["terminal_reward"] = 0.0

    last_idx = df.groupby("hadm_id")["day_of_stay"].idxmax()
    terminal_rows = df.loc[last_idx.values]   # rows for last day per admission

    good_idx = terminal_rows.index[terminal_rows["readmit_30d"].fillna(0) == 0]
    bad_idx  = terminal_rows.index[terminal_rows["readmit_30d"].fillna(0) == 1]

    df.loc[good_idx,        "terminal_reward"] =  TERMINAL_VALUE
    df.loc[bad_idx,         "terminal_reward"] = -TERMINAL_VALUE
    df.loc[last_idx.values, "done"] = 1

    n_good = len(good_idx)
    n_bad  = len(bad_idx)
    log.info(
        "Terminal rows: %d good (+%.0f), %d readmit (-%.0f)",
        n_good, TERMINAL_VALUE, n_bad, TERMINAL_VALUE,
    )
    return df


# ---------------------------------------------------------------------------
# Winsorization
# ---------------------------------------------------------------------------

def winsorize_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Clip each column at train-set 1st / 99th percentile.

    Applied to both current and next-state columns using the same clip bounds
    derived from the current-state column in the training set.
    """
    win_stats: dict = {}
    for col in cols:
        if col not in train_df.columns:
            continue
        lo = float(train_df[col].quantile(0.01))
        hi = float(train_df[col].quantile(0.99))
        win_stats[col] = {"lo": lo, "hi": hi}
        next_col = "next_" + col if col in CORE_LABS else None
        for df in [train_df, val_df, test_df]:
            df[col] = df[col].clip(lo, hi)
            if next_col and next_col in df.columns:
                df[next_col] = df[next_col].clip(lo, hi)
    return train_df, val_df, test_df, win_stats


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Normalize state features using train-set statistics.

    LOG_LABS    : log(0.1 + x), then z-score
    ZSCORE_CONT : z-score
    BINARY      : encode then subtract 0.5

    The same transform is applied to both current and next-state columns
    using identical statistics (derived from current-state train data).
    """
    norm_stats: dict = {}

    # Gender: M -> 1, F -> 0, other -> 0
    for df in [train_df, val_df, test_df]:
        df["gender"] = (df["gender"].astype(str).str.upper() == "M").astype(float)

    # LOG_LABS: apply to current col + corresponding next_ col
    for col in LOG_LABS:
        next_col = "next_" + col
        for df in [train_df, val_df, test_df]:
            df[col] = np.log(0.1 + df[col].astype(float))
            if next_col in df.columns:
                df[next_col] = np.log(0.1 + df[next_col].astype(float))
        mean = float(train_df[col].mean())
        std  = float(train_df[col].std()) or 1.0
        norm_stats[col] = {"transform": "log_zscore", "mean": mean, "std": std}
        for df in [train_df, val_df, test_df]:
            df[col] = (df[col] - mean) / std
            if next_col in df.columns:
                df[next_col] = (df[next_col] - mean) / std

    # ZSCORE_CONT: apply to current col + corresponding next_ col (where applicable)
    for col in ZSCORE_CONT:
        if col not in train_df.columns:
            log.warning("Zscore col %s not found, skipping", col)
            continue
        mean = float(train_df[col].astype(float).mean())
        std  = float(train_df[col].astype(float).std()) or 1.0
        norm_stats[col] = {"transform": "zscore", "mean": mean, "std": std}
        next_col = NEXT_COL_MAP.get(col)
        for df in [train_df, val_df, test_df]:
            df[col] = (df[col].astype(float) - mean) / std
            if next_col and next_col in df.columns:
                df[next_col] = (df[next_col].astype(float) - mean) / std

    # BINARY: subtract 0.5 (gender already encoded above)
    for col in BINARY_FEATURES:
        norm_stats[col] = {"transform": "binary_minus_half"}
        for df in [train_df, val_df, test_df]:
            df[col] = df[col].astype(float) - 0.5

    return train_df, val_df, test_df, norm_stats


# ---------------------------------------------------------------------------
# Build RL dataset (assemble output rows)
# ---------------------------------------------------------------------------

def build_rl_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble final (s, a, r, s', done) dataframe.

    Output columns:
      Metadata : hadm_id, day_of_stay, split, readmit_30d
      Action   : action_id
      Rewards  : reward, dense_reward, terminal_reward
      Terminal : done
      Current s: s_{feature} for each of 19 STATE_FEATURES
      Next s'  : sn_{feature} for each of 19 STATE_FEATURES
    """
    rows: dict[str, pd.Series | np.ndarray] = {}

    # Metadata
    for col in ["hadm_id", "day_of_stay", "split", "readmit_30d",
                "action_id", "reward", "dense_reward", "terminal_reward", "done"]:
        if col in df.columns:
            rows[col] = df[col].values

    # Current state (s_ prefix)
    for feat in STATE_FEATURES:
        if feat in df.columns:
            rows["s_" + feat] = df[feat].values
        else:
            rows["s_" + feat] = np.full(len(df), np.nan)

    # Next state (sn_ prefix)
    for feat in STATE_FEATURES:
        src = NEXT_COL_MAP.get(feat, "next_" + feat)
        if src == "_next_day_of_stay":
            rows["sn_" + feat] = df["day_of_stay"].values + 1.0
        elif src in df.columns:
            rows["sn_" + feat] = df[src].values
        else:
            # Static feature not found as next_col: carry forward current
            rows["sn_" + feat] = df[feat].values if feat in df.columns else np.full(len(df), np.nan)

    return pd.DataFrame(rows, index=df.index)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_preprocessing(
    csv_path: str | Path,
    out_dir: str | Path,
    smoke: bool = False,
) -> None:
    """End-to-end preprocessing: load -> reward -> winsorize -> normalize -> save."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading V3 triplets from %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    if smoke:
        df = df.head(2000)
        log.info("Smoke mode: using %d rows", len(df))
    log.info("Loaded %d rows", len(df))

    # Fill sparse columns that may be NaN
    df["positive_culture_cumulative"] = df["positive_culture_cumulative"].fillna(0)
    df["charlson_score"] = df["charlson_score"].fillna(0)
    df["next_positive_culture_cumulative"] = (
        df["next_positive_culture_cumulative"].fillna(0)
    )

    # Encode actions (before any normalization)
    log.info("Encoding actions...")
    df["action_id"] = encode_actions(df)
    action_dist = df["action_id"].value_counts()
    log.info(
        "Action distribution: %d unique combos observed (of 256 possible). "
        "Top-3: %s",
        len(action_dist),
        action_dist.head(3).to_dict(),
    )

    # Compute dense reward on raw lab values
    log.info("Computing dense reward...")
    df["dense_reward"] = compute_dense_reward(df)
    log.info(
        "Dense reward: mean=%.4f std=%.4f min=%.4f max=%.4f",
        df["dense_reward"].mean(), df["dense_reward"].std(),
        df["dense_reward"].min(), df["dense_reward"].max(),
    )

    # Mark terminal rows and add terminal reward
    log.info("Marking terminal rows...")
    df = mark_terminals(df)

    # Combined reward
    df["reward"] = df["dense_reward"] + df["terminal_reward"]

    # Split
    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "valid"].copy()
    test_df  = df[df["split"] == "test"].copy()
    log.info(
        "Split sizes: train=%d val=%d test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    # Winsorize continuous features (train stats applied to all splits)
    log.info("Winsorizing...")
    win_cols = CORE_LABS + [
        c for c in ZSCORE_CONT if c not in CORE_LABS
    ]
    train_df, val_df, test_df, win_stats = winsorize_features(
        train_df, val_df, test_df, win_cols
    )

    # Normalize
    log.info("Normalizing...")
    train_df, val_df, test_df, norm_stats = normalize_features(
        train_df, val_df, test_df
    )

    # Save norm stats
    stats = {"winsorize": win_stats, "normalize": norm_stats}
    stats_path = out_dir / "norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Saved norm stats to %s", stats_path)

    # Build and save RL datasets
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rl_df = build_rl_dataset(split_df)
        out_path = out_dir / ("rl_%s.csv" % name)
        rl_df.to_csv(out_path, index=False)
        log.info(
            "Saved %s: %d rows, %d cols -> %s",
            name, len(rl_df), len(rl_df.columns), out_path,
        )

    log.info("Preprocessing complete. Output dir: %s", out_dir)
