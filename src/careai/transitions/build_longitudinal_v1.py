"""Build longitudinal long-table and padded tensor artifacts from transition v2 rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LongitudinalArtifacts:
    long_df: pd.DataFrame
    tensors: dict[str, np.ndarray]
    episode_index_df: pd.DataFrame
    metadata: dict[str, Any]


def build_longitudinal_from_transitions_v2(df: pd.DataFrame, cfg: dict[str, Any]) -> LongitudinalArtifacts:
    features = list(cfg["features"]["state_numeric"])
    action_map = {str(k): int(v) for k, v in cfg["actions"]["mapping"].items()}
    split_map = {str(k): int(v) for k, v in cfg["split"]["mapping"].items()}
    padding_action_id = int(cfg["actions"]["padding_id"])

    required = {
        "episode_id",
        "patient_id",
        "t",
        "split",
        "index_admittime",
        "index_dischtime",
        "a_t",
        "y_t1",
        "done",
    }.union(features)
    missing = sorted(c for c in required if c not in df.columns)
    if missing:
        raise ValueError(f"Missing required columns for longitudinal_v1 build: {missing}")

    work = df.copy()
    work["t"] = pd.to_numeric(work["t"], errors="coerce")
    work["y_t1"] = pd.to_numeric(work["y_t1"], errors="coerce")
    work["done"] = pd.to_numeric(work["done"], errors="coerce")
    work = work.dropna(subset=["episode_id", "patient_id", "t", "split", "a_t", "y_t1", "done"]).copy()
    work["t"] = work["t"].astype(int)
    work["y_t1"] = work["y_t1"].astype(int)
    work["done"] = work["done"].astype(int)

    for col in features:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    long_cols = [
        "episode_id",
        "patient_id",
        "t",
        "split",
        "index_admittime",
        "index_dischtime",
        "a_t",
        "y_t1",
        "done",
        *features,
    ]
    long_df = (
        work[long_cols]
        .sort_values(["episode_id", "t", "patient_id"], kind="stable")
        .reset_index(drop=True)
    )

    grouped = list(long_df.groupby("episode_id", sort=False))
    n_episodes = len(grouped)
    t_max = max((len(g) for _, g in grouped), default=0)
    n_features = len(features)

    x_state = np.full((n_episodes, t_max, n_features), np.nan, dtype=np.float32)
    m_valid = np.zeros((n_episodes, t_max), dtype=np.uint8)
    a_action = np.full((n_episodes, t_max), padding_action_id, dtype=np.int16)
    y_next = np.zeros((n_episodes, t_max), dtype=np.uint8)
    d_done = np.zeros((n_episodes, t_max), dtype=np.uint8)
    split_arr = np.full((n_episodes,), -1, dtype=np.int8)

    episode_rows: list[dict[str, Any]] = []
    for idx, (episode_id, g) in enumerate(grouped):
        g = g.sort_values("t", kind="stable").reset_index(drop=True)
        expected_t = list(range(len(g)))
        if g["t"].tolist() != expected_t:
            raise ValueError(f"Episode {episode_id} has non-contiguous t values: {g['t'].tolist()}")

        patient_values = g["patient_id"].dropna().unique().tolist()
        split_values = g["split"].dropna().astype(str).unique().tolist()
        if len(patient_values) != 1:
            raise ValueError(f"Episode {episode_id} maps to multiple patient_id values: {patient_values}")
        if len(split_values) != 1:
            raise ValueError(f"Episode {episode_id} maps to multiple split values: {split_values}")
        split_name = split_values[0]
        if split_name not in split_map:
            raise ValueError(f"Unknown split value '{split_name}' in episode {episode_id}")

        length = len(g)
        m_valid[idx, :length] = 1
        x_state[idx, :length, :] = g[features].to_numpy(dtype=np.float32)
        a_action[idx, :length] = g["a_t"].map(lambda a: action_map.get(str(a), action_map.get("A_UNKNOWN", 3))).to_numpy(
            dtype=np.int16
        )
        y_next[idx, :length] = g["y_t1"].to_numpy(dtype=np.uint8)
        d_done[idx, :length] = g["done"].to_numpy(dtype=np.uint8)
        split_arr[idx] = np.int8(split_map[split_name])

        episode_rows.append(
            {
                "tensor_row": idx,
                "episode_id": str(episode_id),
                "patient_id": int(patient_values[0]),
                "length": int(length),
                "split": split_name,
            }
        )

    episode_index_df = pd.DataFrame(episode_rows)
    tensors = {
        "X_state": x_state,
        "M_valid": m_valid,
        "A_action": a_action,
        "Y_next": y_next,
        "D_done": d_done,
        "split": split_arr,
    }
    metadata = {
        "feature_names": features,
        "action_to_id": action_map,
        "id_to_action": {str(v): k for k, v in action_map.items()},
        "split_to_id": split_map,
        "padding": {"action_id": padding_action_id},
        "shapes": {k: list(v.shape) for k, v in tensors.items()},
        "dtypes": {k: str(v.dtype) for k, v in tensors.items()},
        "n_rows_long": int(len(long_df)),
        "n_episodes": int(n_episodes),
        "t_max": int(t_max),
    }
    return LongitudinalArtifacts(
        long_df=long_df,
        tensors=tensors,
        episode_index_df=episode_index_df,
        metadata=metadata,
    )
