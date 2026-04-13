"""
ICUSequenceDataset: loads an ICU replay parquet and returns padded sequences of
(state, action, next_state, reward, terminal, time_step) for transformer training.

The original CARE-Sim path used a fixed Tier-2 schema. This module now keeps that
schema as the default, while also supporting schema inference from a parquet so
parallel tracks can change state/action space without rewriting the core data loader.
"""
from __future__ import annotations

import torch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from torch.utils.data import Dataset

# Column name constants (Tier-2, 8-state version)
STATE_COLS = [
    "s_Hb",
    "s_BUN",
    "s_Creatinine",
    "s_HR",
    "s_Shock_Index",
    "s_age",
    "s_charlson_score",
    "s_prior_ed_visits_6m",
]
ACTION_COLS = ["vasopressor_b", "ivfluid_b", "antibiotic_b", "diuretic_b"]
NEXT_STATE_COLS = [
    "s_next_Hb",
    "s_next_BUN",
    "s_next_Creatinine",
    "s_next_HR",
    "s_next_Shock_Index",
    "s_next_age",
    "s_next_charlson_score",
    "s_next_prior_ed_visits_6m",
]
REWARD_COL = "r"
DONE_COL = "done"
STAY_COL = "icustayid"
BLOC_COL = "bloc"
SPLIT_COL = "split"


def infer_schema_from_df(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Infer (state_cols, action_cols, next_state_cols) from a replay DataFrame.

    Rules:
      - state columns    : `s_*` but not `s_next_*`
      - next-state cols  : matching `s_next_*`
      - action columns   : binary action flags ending in `_b`, preserving parquet order
    """
    state_cols = [c for c in df.columns if c.startswith("s_") and not c.startswith("s_next_")]
    next_state_cols = [f"s_next_{c.removeprefix('s_')}" for c in state_cols]
    missing_next = [c for c in next_state_cols if c not in df.columns]
    if missing_next:
        raise ValueError(f"Missing inferred next-state columns: {missing_next}")
    action_cols = [c for c in df.columns if c.endswith("_b")]
    if not state_cols:
        raise ValueError("No state columns inferred from dataframe")
    if not action_cols:
        raise ValueError("No action columns inferred from dataframe")
    return state_cols, action_cols, next_state_cols


def infer_schema_from_path(path: str) -> tuple[list[str], list[str], list[str]]:
    """Infer replay schema from a parquet path."""
    cols = pq.ParquetFile(path).schema.names
    return infer_schema_from_df(pd.DataFrame(columns=cols))


class ICUSequenceDataset(Dataset):
    """Dataset of ICU stay sequences for transformer world-model training.

    Args:
        df    : DataFrame with the rl_dataset_tier2 structure (or synthetic equivalent)
        split : "train", "val", "test", or None (use all rows)
        max_seq_len : maximum sequence/window length exposed to the model
        window_mode : "last" keeps the last max_seq_len blocs for long stays;
                      "random" samples a contiguous max_seq_len window from anywhere
                      in the full stay on each __getitem__ call.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split: str | None = "train",
        max_seq_len: int = 80,
        window_mode: str = "last",
        state_cols: list[str] | None = None,
        action_cols: list[str] | None = None,
        next_state_cols: list[str] | None = None,
    ):
        if window_mode not in {"last", "random"}:
            raise ValueError(f"window_mode must be 'last' or 'random', got {window_mode!r}")
        self.max_seq_len = max_seq_len
        self.window_mode = window_mode
        if state_cols is None or action_cols is None or next_state_cols is None:
            inferred_state_cols, inferred_action_cols, inferred_next_state_cols = infer_schema_from_df(df)
            state_cols = state_cols or inferred_state_cols
            action_cols = action_cols or inferred_action_cols
            next_state_cols = next_state_cols or inferred_next_state_cols
        self.state_cols = list(state_cols)
        self.action_cols = list(action_cols)
        self.next_state_cols = list(next_state_cols)

        if split is not None and SPLIT_COL in df.columns:
            df = df[df[SPLIT_COL] == split].copy()

        self.sequences = self._build_sequences(df)

    def _build_sequences(self, df: pd.DataFrame) -> list[dict]:
        """Group rows by stay_id, sort by bloc, return list of sequence dicts."""
        sequences = []
        for stay_id, stay_df in df.groupby(STAY_COL, sort=False):
            stay_df = stay_df.sort_values(BLOC_COL).reset_index(drop=True)

            if len(stay_df) == 0:
                continue

            states = stay_df[self.state_cols].to_numpy(dtype=np.float32, copy=True)
            actions = stay_df[self.action_cols].to_numpy(dtype=np.float32, copy=True)
            next_states = stay_df[self.next_state_cols].to_numpy(dtype=np.float32, copy=True)
            rewards = stay_df[REWARD_COL].to_numpy(dtype=np.float32, copy=True)
            terminals = stay_df[DONE_COL].to_numpy(dtype=np.float32, copy=True)
            time_steps = stay_df[BLOC_COL].to_numpy(dtype=np.float32, copy=True)

            sequences.append({
                "states": states,         # (T, state_dim)
                "actions": actions,       # (T, action_dim)
                "next_states": next_states,  # (T, state_dim)
                "rewards": rewards,       # (T,)
                "terminals": terminals,   # (T,)
                "time_steps": time_steps, # (T,)
                "full_length": len(stay_df),
            })

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        seq = self.sequences[idx]
        states = seq["states"]
        actions = seq["actions"]
        next_states = seq["next_states"]
        rewards = seq["rewards"]
        terminals = seq["terminals"]
        time_steps = seq["time_steps"]
        full_length = int(seq["full_length"])

        if full_length > self.max_seq_len:
            if self.window_mode == "last":
                start = full_length - self.max_seq_len
            else:
                start = int(np.random.randint(0, full_length - self.max_seq_len + 1))
            stop = start + self.max_seq_len
            states = states[start:stop]
            actions = actions[start:stop]
            next_states = next_states[start:stop]
            rewards = rewards[start:stop]
            terminals = terminals[start:stop]
            time_steps = time_steps[start:stop]

        length = len(states)
        return {
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "next_states": torch.from_numpy(next_states),
            "rewards": torch.from_numpy(rewards),
            "terminals": torch.from_numpy(terminals),
            "time_steps": torch.from_numpy(time_steps),
            "length": length,
        }

    @classmethod
    def from_parquet(
        cls,
        path: str,
        split: str | None = "train",
        max_seq_len: int = 80,
        window_mode: str = "last",
        state_cols: list[str] | None = None,
        action_cols: list[str] | None = None,
        next_state_cols: list[str] | None = None,
    ) -> "ICUSequenceDataset":
        df = pd.read_parquet(path)
        return cls(
            df,
            split=split,
            max_seq_len=max_seq_len,
            window_mode=window_mode,
            state_cols=state_cols,
            action_cols=action_cols,
            next_state_cols=next_state_cols,
        )


def collate_sequences(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length sequences to the same length.

    Padding is appended at the END of shorter sequences (right-padding).
    The padding mask is True where positions are padding (to be ignored in attention).

    Returns dict with:
        states               : (B, T_max, state_dim)
        actions              : (B, T_max, action_dim)
        next_states          : (B, T_max, state_dim)
        rewards              : (B, T_max)
        terminals            : (B, T_max)
        time_steps           : (B, T_max) float -- bloc index / elapsed step in stay
        src_key_padding_mask : (B, T_max) bool -- True = padding position
        lengths              : (B,) int
    """
    max_len = max(item["length"] for item in batch)
    B = len(batch)

    state_dim = batch[0]["states"].shape[-1]
    action_dim = batch[0]["actions"].shape[-1]

    states_pad = torch.zeros(B, max_len, state_dim)
    actions_pad = torch.zeros(B, max_len, action_dim)
    next_states_pad = torch.zeros(B, max_len, state_dim)
    rewards_pad = torch.zeros(B, max_len)
    terminals_pad = torch.zeros(B, max_len)
    time_steps_pad = torch.zeros(B, max_len)
    padding_mask = torch.ones(B, max_len, dtype=torch.bool)  # True = ignored
    lengths = torch.zeros(B, dtype=torch.long)

    for i, item in enumerate(batch):
        T = item["length"]
        states_pad[i, :T] = item["states"]
        actions_pad[i, :T] = item["actions"]
        next_states_pad[i, :T] = item["next_states"]
        rewards_pad[i, :T] = item["rewards"]
        terminals_pad[i, :T] = item["terminals"]
        time_steps_pad[i, :T] = item["time_steps"]
        padding_mask[i, :T] = False   # real positions are NOT masked
        lengths[i] = T

    return {
        "states": states_pad,
        "actions": actions_pad,
        "next_states": next_states_pad,
        "rewards": rewards_pad,
        "terminals": terminals_pad,
        "time_steps": time_steps_pad,
        "src_key_padding_mask": padding_mask,
        "lengths": lengths,
    }
