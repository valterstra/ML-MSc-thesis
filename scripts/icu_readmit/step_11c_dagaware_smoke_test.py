"""
Step 11c -- DAG-aware temporal world model smoke test.

Runs a compact synthetic-data check covering:
  1. forward pass shapes
  2. dataset + collate behavior
  3. loss and backward pass
  4. tiny training loop
  5. ensemble predict/predict_last_step
  6. simulator reset/step/rollout
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from careai.icu_readmit.caresim.dataset import ICUSequenceDataset, collate_sequences
from careai.icu_readmit.dagaware.ensemble import DAGAwareEnsemble
from careai.icu_readmit.dagaware.model import DAGAwareTemporalWorldModel
from careai.icu_readmit.dagaware.simulator import DAGAwareEnvironment
from careai.icu_readmit.dagaware.train import compute_loss, train_model

torch.manual_seed(42)
np.random.seed(42)

STATE_COLS = [
    "s_Hb",
    "s_BUN",
    "s_Creatinine",
    "s_Phosphate",
    "s_HR",
    "s_Chloride",
    "s_age",
    "s_charlson_score",
    "s_prior_ed_visits_6m",
]
ACTION_COLS = ["vasopressor_b", "ivfluid_b", "antibiotic_b", "diuretic_b", "mechvent_b"]
NEXT_STATE_COLS = [f"s_next_{col.removeprefix('s_')}" for col in STATE_COLS]
STATE_DIM = len(STATE_COLS)
ACTION_DIM = len(ACTION_COLS)
DEVICE = torch.device("cpu")


def make_synthetic_df(n_stays: int = 120, min_blocs: int = 4, max_blocs: int = 16) -> pd.DataFrame:
    rows = []
    mask = np.array(
        [
            [1, 1, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 1, 0],
        ],
        dtype=np.float32,
    )
    for stay_id in range(n_stays):
        n_blocs = np.random.randint(min_blocs, max_blocs + 1)
        dynamic_state = np.random.randn(6).astype(np.float32)
        static_state = np.array(
            [
                np.random.uniform(30, 85),
                np.random.uniform(0, 10),
                np.random.uniform(0, 6),
            ],
            dtype=np.float32,
        )
        state = np.concatenate([dynamic_state, static_state], axis=0)
        split = np.random.choice(["train", "val", "test"], p=[0.7, 0.15, 0.15])

        for bloc in range(n_blocs):
            action = (np.random.rand(ACTION_DIM) > 0.72).astype(np.float32)
            direct_effect = mask @ action
            next_dynamic = state[:6] + 0.15 * np.random.randn(6).astype(np.float32) + 0.1 * direct_effect
            next_state = np.concatenate([next_dynamic, static_state], axis=0)
            done = 1.0 if bloc == n_blocs - 1 else 0.0

            row = {"icustayid": stay_id, "bloc": bloc, "split": split, "r": float(np.random.randn() * 0.1), "done": done}
            for idx, col in enumerate(STATE_COLS):
                row[col] = float(state[idx])
            for idx, col in enumerate(ACTION_COLS):
                row[col] = float(action[idx])
            for idx, col in enumerate(NEXT_STATE_COLS):
                row[col] = float(next_state[idx])
            rows.append(row)
            state = next_state
    return pd.DataFrame(rows)


def make_model(**overrides) -> DAGAwareTemporalWorldModel:
    kwargs = {
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "d_model": 48,
        "n_heads": 4,
        "n_layers": 3,
        "dropout": 0.1,
        "max_seq_len": 24,
        "dynamic_state_idx": (0, 1, 2, 3, 4, 5),
        "static_state_idx": (6, 7, 8),
        "use_time_feature": True,
        "predict_reward": False,
    }
    kwargs.update(overrides)
    return DAGAwareTemporalWorldModel(**kwargs)


def test_forward():
    print("\n[Test 1] Forward pass...")
    model = make_model()
    states = torch.randn(3, 8, STATE_DIM)
    actions = torch.randint(0, 2, (3, 8, ACTION_DIM)).float()
    out = model(states, actions)
    assert out["next_state"].shape == (3, 8, STATE_DIM)
    assert out["reward"] is None
    assert out["terminal"].shape == (3, 8)
    pred = model.predict_step(states, actions)
    assert pred["next_state"].shape == (3, STATE_DIM)
    assert pred["terminal"].shape == (3,)
    print("  Shapes OK")


def test_dataset(df: pd.DataFrame):
    print("\n[Test 2] Dataset + collate...")
    ds = ICUSequenceDataset(
        df,
        split="train",
        max_seq_len=24,
        state_cols=STATE_COLS,
        action_cols=ACTION_COLS,
        next_state_cols=NEXT_STATE_COLS,
    )
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_sequences)
    batch = next(iter(loader))
    assert batch["states"].shape[-1] == STATE_DIM
    assert batch["actions"].shape[-1] == ACTION_DIM
    assert batch["src_key_padding_mask"].dtype == torch.bool
    print("  Batch shapes OK")
    return loader


def test_loss(loader: DataLoader):
    print("\n[Test 3] Loss + backward...")
    model = make_model()
    batch = next(iter(loader))
    pred = model(batch["states"], batch["actions"], src_key_padding_mask=batch["src_key_padding_mask"], time_steps=batch["time_steps"])
    loss, breakdown = compute_loss(pred, batch)
    assert loss.requires_grad
    assert breakdown["state"] > 0
    assert breakdown["terminal"] >= 0
    loss.backward()
    print(f"  Total loss: {breakdown['total']:.4f}")


def test_training(df: pd.DataFrame):
    print("\n[Test 4] Tiny training run...")
    model = make_model()
    train_ds = ICUSequenceDataset(
        df,
        split="train",
        max_seq_len=24,
        state_cols=STATE_COLS,
        action_cols=ACTION_COLS,
        next_state_cols=NEXT_STATE_COLS,
    )
    val_ds = ICUSequenceDataset(
        df,
        split="val",
        max_seq_len=24,
        state_cols=STATE_COLS,
        action_cols=ACTION_COLS,
        next_state_cols=NEXT_STATE_COLS,
    )
    train_loader = DataLoader(train_ds, batch_size=12, shuffle=True, collate_fn=collate_sequences)
    val_loader = DataLoader(val_ds, batch_size=12, shuffle=False, collate_fn=collate_sequences)
    history = train_model(model, train_loader, val_loader, n_epochs=4, lr=1e-3, device=DEVICE, verbose=True)
    assert history[-1]["train"]["total"] < history[0]["train"]["total"]
    return model


def test_ensemble_and_sim(model: DAGAwareTemporalWorldModel):
    print("\n[Test 5] Ensemble + simulator...")
    ensemble = DAGAwareEnsemble([model, make_model()], device=DEVICE)
    states = torch.randn(2, 6, STATE_DIM)
    actions = torch.randint(0, 2, (2, 6, ACTION_DIM)).float()
    out = ensemble.predict(states, actions)
    assert out["next_state_mean"].shape == (2, 6, STATE_DIM)
    assert out["next_state_std"].shape == (2, 6, STATE_DIM)
    pred_last = ensemble.predict_last_step(states, actions)
    assert pred_last["next_state_mean"].shape == (2, STATE_DIM)

    env = DAGAwareEnvironment(ensemble, max_steps=5, uncertainty_threshold=2.0)
    current = env.reset(states, actions)
    assert current.shape == (2, STATE_DIM)
    next_state, reward, done, info = env.step(torch.randint(0, 2, (2, ACTION_DIM)).float())
    assert next_state.shape == (2, STATE_DIM)
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert "uncertainty" in info
    rollout = env.rollout(torch.randint(0, 2, (2, 3, ACTION_DIM)).float())
    assert rollout["states"].shape == (2, 3, STATE_DIM)
    print("  Ensemble and simulator OK")


def main():
    print("=" * 60)
    print("DAG-aware temporal world model smoke test")
    print("=" * 60)
    df = make_synthetic_df()
    test_forward()
    loader = test_dataset(df)
    test_loss(loader)
    model = test_training(df)
    test_ensemble_and_sim(model)
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
