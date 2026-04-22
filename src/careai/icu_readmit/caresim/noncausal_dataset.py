from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


STAY_COL = "icustayid"
BLOC_COL = "bloc"
SPLIT_COL = "split"
DONE_COL = "done"
READMIT_COL = "readmit_30d"


class NonCausalICUSequenceDataset(Dataset):
    """Sequence dataset for the broad non-causal CARE-Sim branch."""

    def __init__(
        self,
        df: pd.DataFrame,
        state_cols: list[str],
        action_cols: list[str],
        next_state_cols: list[str],
        split: str | None = "train",
        max_seq_len: int = 10,
        window_mode: str = "last",
    ):
        if window_mode not in {"last", "random"}:
            raise ValueError(f"window_mode must be 'last' or 'random', got {window_mode!r}")
        self.state_cols = list(state_cols)
        self.action_cols = list(action_cols)
        self.next_state_cols = list(next_state_cols)
        self.max_seq_len = int(max_seq_len)
        self.window_mode = window_mode

        if split is not None and SPLIT_COL in df.columns:
            df = df[df[SPLIT_COL] == split].copy()

        self.sequences = self._build_sequences(df)

    def _build_sequences(self, df: pd.DataFrame) -> list[dict]:
        sequences: list[dict] = []
        for stay_id, stay_df in df.groupby(STAY_COL, sort=False):
            stay_df = stay_df.sort_values(BLOC_COL).reset_index(drop=True)
            if stay_df.empty:
                continue

            sequences.append(
                {
                    "states": stay_df[self.state_cols].to_numpy(dtype=np.float32, copy=True),
                    "actions": stay_df[self.action_cols].to_numpy(dtype=np.float32, copy=True),
                    "next_states": stay_df[self.next_state_cols].to_numpy(dtype=np.float32, copy=True),
                    "terminals": stay_df[DONE_COL].to_numpy(dtype=np.float32, copy=True),
                    "time_steps": stay_df[BLOC_COL].to_numpy(dtype=np.float32, copy=True),
                    "readmit": float(stay_df[READMIT_COL].iloc[0]),
                    "full_length": len(stay_df),
                    "stay_id": int(stay_id),
                }
            )
        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | float]:
        seq = self.sequences[idx]
        full_length = int(seq["full_length"])

        start = 0
        stop = full_length
        if full_length > self.max_seq_len:
            if self.window_mode == "last":
                start = full_length - self.max_seq_len
            else:
                start = int(np.random.randint(0, full_length - self.max_seq_len + 1))
            stop = start + self.max_seq_len

        states = seq["states"][start:stop]
        actions = seq["actions"][start:stop]
        next_states = seq["next_states"][start:stop]
        terminals = seq["terminals"][start:stop]
        time_steps = seq["time_steps"][start:stop]

        return {
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "next_states": torch.from_numpy(next_states),
            "terminals": torch.from_numpy(terminals),
            "time_steps": torch.from_numpy(time_steps),
            "readmit": torch.tensor(seq["readmit"], dtype=torch.float32),
            "length": len(states),
            "stay_id": seq["stay_id"],
        }

    @classmethod
    def from_parquet(
        cls,
        path: str,
        state_cols: list[str],
        action_cols: list[str],
        next_state_cols: list[str],
        split: str | None = "train",
        max_seq_len: int = 10,
        window_mode: str = "last",
    ) -> "NonCausalICUSequenceDataset":
        df = pd.read_parquet(path)
        return cls(
            df=df,
            state_cols=state_cols,
            action_cols=action_cols,
            next_state_cols=next_state_cols,
            split=split,
            max_seq_len=max_seq_len,
            window_mode=window_mode,
        )


def collate_noncausal_sequences(batch: list[dict]) -> dict[str, torch.Tensor]:
    max_len = max(int(item["length"]) for item in batch)
    batch_size = len(batch)
    state_dim = batch[0]["states"].shape[-1]
    action_dim = batch[0]["actions"].shape[-1]
    next_state_dim = batch[0]["next_states"].shape[-1]

    states_pad = torch.zeros(batch_size, max_len, state_dim)
    actions_pad = torch.zeros(batch_size, max_len, action_dim)
    next_states_pad = torch.zeros(batch_size, max_len, next_state_dim)
    terminals_pad = torch.zeros(batch_size, max_len)
    time_steps_pad = torch.zeros(batch_size, max_len)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    readmit = torch.zeros(batch_size, dtype=torch.float32)
    stay_ids = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = int(item["length"])
        states_pad[i, :seq_len] = item["states"]
        actions_pad[i, :seq_len] = item["actions"]
        next_states_pad[i, :seq_len] = item["next_states"]
        terminals_pad[i, :seq_len] = item["terminals"]
        time_steps_pad[i, :seq_len] = item["time_steps"]
        padding_mask[i, :seq_len] = False
        lengths[i] = seq_len
        readmit[i] = item["readmit"]
        stay_ids[i] = item["stay_id"]

    return {
        "states": states_pad,
        "actions": actions_pad,
        "next_states": next_states_pad,
        "terminals": terminals_pad,
        "time_steps": time_steps_pad,
        "src_key_padding_mask": padding_mask,
        "lengths": lengths,
        "readmit": readmit,
        "stay_ids": stay_ids,
    }
