"""Episode chaining helpers for transition v2."""

from __future__ import annotations

import pandas as pd


def assign_episode_ids_and_steps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["patient_id", "index_admittime", "index_hadm_id"]).reset_index(drop=True)
    # A new chain starts after a terminal row. Shift done so current row reacts to previous row's terminal flag.
    prev_done = out.groupby("patient_id", sort=False)["done"].shift(1).fillna(0).astype(int)
    out["_chain_idx"] = prev_done.groupby(out["patient_id"], sort=False).cumsum()
    out["episode_step"] = out.groupby(["patient_id", "_chain_idx"], sort=False).cumcount()
    out["episode_id"] = out["patient_id"].astype(int).astype(str) + "_ep" + out["_chain_idx"].astype(int).astype(str)
    out["t"] = out["episode_step"].astype(int)
    out = out.drop(columns=["_chain_idx"])
    return out
