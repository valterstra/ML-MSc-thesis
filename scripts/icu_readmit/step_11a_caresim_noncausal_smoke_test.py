from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
from torch.utils.data import DataLoader

from careai.icu_readmit.caresim.noncausal_dataset import (
    NonCausalICUSequenceDataset,
    collate_noncausal_sequences,
)
from careai.icu_readmit.caresim.noncausal_model import NonCausalCareSimTransformer
from careai.icu_readmit.caresim.noncausal_train import (
    compute_noncausal_loss,
    train_noncausal_model,
)

from step_11a_caresim_train_noncausal import (  # noqa: E402
    _infer_schema_from_parquet,
    _repair_training_data,
    _validate_training_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "step_10_noncausal" / "rl_dataset_noncausal.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test for non-causal CARE-Sim on real replay data")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data
    print("=" * 60)
    print("CARE-Sim Non-Causal Smoke Test")
    print("=" * 60)
    print(f"Data: {data_path}")

    state_cols, next_state_cols, numeric_state_cols, categorical_state_cols, action_cols, cat_cardinalities = (
        _infer_schema_from_parquet(data_path)
    )
    print(f"State dim={len(state_cols)} target dim={len(next_state_cols)} action dim={len(action_cols)}")
    print(f"Categorical cols={categorical_state_cols}")
    assert len(next_state_cols) < len(state_cols), "Expected static context to be excluded from dynamic targets"
    assert len(categorical_state_cols) > 0, "Expected categorical *_code columns to be detected"

    import pandas as pd
    df = pd.read_parquet(data_path)
    df, repaired_cols = _repair_training_data(df, state_cols, next_state_cols, action_cols)
    if repaired_cols:
        print(f"Repaired invalid values in columns: {repaired_cols}")
    _validate_training_data(df, state_cols, next_state_cols, action_cols)
    print("Data validation OK")

    train_ds = NonCausalICUSequenceDataset(
        df=df,
        state_cols=state_cols,
        action_cols=action_cols,
        next_state_cols=next_state_cols,
        split="train",
        max_seq_len=10,
        window_mode="random",
    )
    val_ds = NonCausalICUSequenceDataset(
        df=df,
        state_cols=state_cols,
        action_cols=action_cols,
        next_state_cols=next_state_cols,
        split="val",
        max_seq_len=10,
        window_mode="last",
    )
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_noncausal_sequences, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_noncausal_sequences, num_workers=0)
    batch = next(iter(train_loader))

    numeric_idx = [state_cols.index(col) for col in numeric_state_cols]
    categorical_idx = [state_cols.index(col) for col in categorical_state_cols]
    model = NonCausalCareSimTransformer(
        state_dim=len(state_cols),
        action_dim=len(action_cols),
        dynamic_state_dim=len(next_state_cols),
        numeric_state_idx=numeric_idx,
        categorical_state_idx=categorical_idx,
        categorical_cardinalities=cat_cardinalities,
        d_model=64,
        n_heads=4,
        n_layers=2,
        max_seq_len=10,
    )
    print(f"Parameters: {model.count_parameters():,}")

    states = batch["states"]
    actions = batch["actions"]
    mask = batch["src_key_padding_mask"]
    time_steps = batch["time_steps"]
    pred = model(states, actions, src_key_padding_mask=mask, time_steps=time_steps)
    assert pred["next_state"].shape[-1] == len(next_state_cols)
    assert pred["terminal"].shape == batch["terminals"].shape
    assert pred["readmit"].shape[0] == states.shape[0]
    loss, breakdown = compute_noncausal_loss(pred, batch)
    assert torch.isfinite(loss), f"Expected finite smoke loss, got {loss}"
    print(f"Initial loss: {breakdown}")

    with tempfile.TemporaryDirectory() as tmpdir:
        history, _ = train_noncausal_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=2,
            lr=1e-3,
            device=torch.device("cpu"),
            verbose=True,
            max_train_batches=2,
            max_val_batches=1,
        )
    for entry in history:
        assert all(
            value == value and value != float("inf") and value != float("-inf")
            for value in [entry["train"]["total"], entry["val"]["total"]]
        ), f"Non-finite losses in history: {entry}"
    print("Smoke training OK")
    print("=" * 60)
    print("NON-CAUSAL CARE-Sim smoke test passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
