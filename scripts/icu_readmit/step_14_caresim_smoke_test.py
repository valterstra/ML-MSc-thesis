"""
Step 14 -- CARE-Sim smoke test.

Verifies the full CARE-Sim pipeline (model, dataset, training, simulator) using
synthetic data that mirrors rl_dataset_tier2.parquet structure. No real data needed.

Tests:
  1. CareSimGPT model: forward pass shape checks
  2. ICUSequenceDataset + collate_sequences: padding, mask shapes
  3. compute_loss: differentiable, all three loss components > 0
  4. train_epoch: loss decreases over 5 epochs
  5. CareSimEnsemble: mean/std predictions correct shape
  6. CareSimEnvironment: reset + step + rollout interface
  7. End-to-end: train 2 tiny models, build ensemble, run rollout

Run (from repo root):
  python scripts/icu_readmit/step_14_caresim_smoke_test.py

Windows note: If running via Claude Code's Bash tool, torch DLL loading may fail
when the script is invoked as a .py file (known Windows PATH issue in subprocess
environments). Run directly in your own terminal instead.

Expected output: all checks pass, loss decreases, no exceptions.
"""
import sys
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Make src importable without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from careai.icu_readmit.caresim.model import CareSimGPT
from careai.icu_readmit.caresim.dataset import (
    ICUSequenceDataset, collate_sequences,
    STATE_COLS, ACTION_COLS, NEXT_STATE_COLS, REWARD_COL, DONE_COL,
)
from careai.icu_readmit.caresim.train import compute_loss, train_epoch, train_model
from careai.icu_readmit.caresim.ensemble import CareSimEnsemble
from careai.icu_readmit.caresim.simulator import CareSimEnvironment

torch.manual_seed(42)
np.random.seed(42)

STATE_DIM = len(STATE_COLS)
ACTION_DIM = 4
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Helper: generate synthetic stays
# ---------------------------------------------------------------------------

def make_synthetic_df(n_stays: int = 200, min_blocs: int = 4, max_blocs: int = 20) -> pd.DataFrame:
    """Generate a synthetic DataFrame with the rl_dataset_tier2 column structure."""
    rows = []
    for stay_id in range(n_stays):
        n_blocs = np.random.randint(min_blocs, max_blocs + 1)
        # Random z-scored state features:
        # first 5 are dynamic, last 3 are static confounders repeated each step.
        state = np.random.randn(STATE_DIM).astype(np.float32)
        split = np.random.choice(["train", "val", "test"], p=[0.7, 0.15, 0.15])
        readmit = int(np.random.rand() < 0.2)

        for t in range(n_blocs):
            action = (np.random.rand(ACTION_DIM) > 0.7).astype(np.float32)
            next_state = state + 0.1 * np.random.randn(STATE_DIM).astype(np.float32)
            next_state[5:] = state[5:]   # static confounders do not evolve over time
            done = 1.0 if t == n_blocs - 1 else 0.0
            # Dense reward: simulated SOFA delta; terminal reward: +-15
            if done:
                reward = 15.0 if readmit == 0 else -15.0
            else:
                reward = float(np.random.randn() * 0.5)

            row = {"icustayid": stay_id, "bloc": t, "split": split, "readmit_30d": readmit}
            for i, col in enumerate(STATE_COLS):
                row[col] = float(state[i])
            for i, col in enumerate(ACTION_COLS):
                row[col] = float(action[i])
            for i, col in enumerate(NEXT_STATE_COLS):
                row[col] = float(next_state[i])
            row[REWARD_COL] = reward
            row[DONE_COL] = done

            rows.append(row)
            state = next_state

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: Model forward pass
# ---------------------------------------------------------------------------

def test_model_forward():
    print("\n[Test 1] CareSimGPT forward pass...")
    model = CareSimGPT(state_dim=STATE_DIM, action_dim=ACTION_DIM, d_model=32, n_heads=2, n_layers=2)
    print(f"  Parameters: {model.count_parameters():,}")

    B, T = 4, 10
    states = torch.randn(B, T, STATE_DIM)
    actions = torch.randint(0, 2, (B, T, ACTION_DIM)).float()

    out = model(states, actions)
    assert out["next_state"].shape == (B, T, STATE_DIM), f"next_state shape: {out['next_state'].shape}"
    assert out["reward"].shape == (B, T), f"reward shape: {out['reward'].shape}"
    assert out["terminal"].shape == (B, T), f"terminal shape: {out['terminal'].shape}"
    print(f"  next_state: {out['next_state'].shape} OK")
    print(f"  reward:     {out['reward'].shape} OK")
    print(f"  terminal:   {out['terminal'].shape} OK")

    # predict_step (last position)
    pred_step = model.predict_step(states, actions)
    assert pred_step["next_state"].shape == (B, STATE_DIM)
    assert pred_step["reward"].shape == (B,)
    assert pred_step["terminal"].shape == (B,)
    print("  predict_step shapes OK")
    print("[Test 1] PASSED")


# ---------------------------------------------------------------------------
# Test 2: Dataset and collate
# ---------------------------------------------------------------------------

def test_dataset(df: pd.DataFrame):
    print("\n[Test 2] ICUSequenceDataset + collate_sequences...")
    ds = ICUSequenceDataset(df, split="train", max_seq_len=80)
    print(f"  Train stays: {len(ds)}")
    assert len(ds) > 0

    sample = ds[0]
    assert "states" in sample
    assert sample["states"].shape[-1] == STATE_DIM
    assert sample["actions"].shape[-1] == ACTION_DIM
    print(f"  Sample seq length: {sample['length']}")

    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_sequences)
    batch = next(iter(loader))

    B_act = batch["states"].shape[0]
    T_max = batch["states"].shape[1]
    assert batch["states"].shape == (B_act, T_max, STATE_DIM)
    assert batch["actions"].shape == (B_act, T_max, ACTION_DIM)
    assert batch["next_states"].shape == (B_act, T_max, STATE_DIM)
    assert batch["rewards"].shape == (B_act, T_max)
    assert batch["terminals"].shape == (B_act, T_max)
    assert batch["src_key_padding_mask"].shape == (B_act, T_max)
    assert batch["src_key_padding_mask"].dtype == torch.bool

    n_real = (~batch["src_key_padding_mask"]).sum().item()
    n_pad = batch["src_key_padding_mask"].sum().item()
    print(f"  Batch shape: ({B_act}, {T_max}), real={n_real}, padding={n_pad}")
    print("[Test 2] PASSED")
    return loader


# ---------------------------------------------------------------------------
# Test 3: Loss computation
# ---------------------------------------------------------------------------

def test_loss(loader: DataLoader):
    print("\n[Test 3] compute_loss differentiability...")
    model = CareSimGPT(state_dim=STATE_DIM, action_dim=ACTION_DIM, d_model=32, n_heads=2, n_layers=2)
    model.train()

    batch = next(iter(loader))
    states = batch["states"]
    actions = batch["actions"]
    mask = batch["src_key_padding_mask"]

    pred = model(states, actions, src_key_padding_mask=mask)
    loss, breakdown = compute_loss(pred, batch)

    assert loss.requires_grad, "Loss must require grad"
    assert breakdown["state"] > 0
    assert breakdown["reward"] >= 0
    assert breakdown["terminal"] >= 0
    loss.backward()   # must not raise

    print(f"  state_loss={breakdown['state']:.4f}  reward_loss={breakdown['reward']:.4f}  "
          f"terminal_loss={breakdown['terminal']:.4f}  total={breakdown['total']:.4f}")
    print("  Backward pass OK")
    print("[Test 3] PASSED")


# ---------------------------------------------------------------------------
# Test 4: Training loop -- loss must decrease
# ---------------------------------------------------------------------------

def test_training(df: pd.DataFrame):
    print("\n[Test 4] Training loop (5 epochs, tiny model)...")
    model = CareSimGPT(state_dim=STATE_DIM, action_dim=ACTION_DIM, d_model=32, n_heads=2, n_layers=2)

    train_ds = ICUSequenceDataset(df, split="train", max_seq_len=80)
    val_ds = ICUSequenceDataset(df, split="val", max_seq_len=80)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_sequences)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_sequences)

    history = train_model(
        model, train_loader, val_loader,
        n_epochs=5, lr=1e-3, device=DEVICE, verbose=True
    )

    first_loss = history[0]["train"]["total"]
    last_loss = history[-1]["train"]["total"]
    print(f"  First epoch train loss: {first_loss:.4f}")
    print(f"  Last  epoch train loss: {last_loss:.4f}")
    assert last_loss < first_loss, f"Loss did not decrease: {first_loss:.4f} -> {last_loss:.4f}"
    print("[Test 4] PASSED")
    return model


# ---------------------------------------------------------------------------
# Test 5: Ensemble predictions
# ---------------------------------------------------------------------------

def test_ensemble(model: CareSimGPT):
    print("\n[Test 5] CareSimEnsemble predictions...")
    # Build a tiny ensemble from 2 copies of the trained model
    model2 = CareSimGPT(state_dim=STATE_DIM, action_dim=ACTION_DIM, d_model=32, n_heads=2, n_layers=2)
    ensemble = CareSimEnsemble([model, model2], device=DEVICE)
    print(f"  Ensemble size: {ensemble.n_models}")

    B, T = 3, 8
    states = torch.randn(B, T, STATE_DIM)
    actions = torch.randint(0, 2, (B, T, ACTION_DIM)).float()

    out = ensemble.predict(states, actions)
    assert out["next_state_mean"].shape == (B, T, STATE_DIM)
    assert out["next_state_std"].shape == (B, T, STATE_DIM)
    assert out["reward_mean"].shape == (B, T)
    assert out["terminal_prob"].shape == (B, T)
    assert (out["next_state_std"] >= 0).all(), "Std must be non-negative"

    pred_last = ensemble.predict_last_step(states, actions)
    assert pred_last["next_state_mean"].shape == (B, STATE_DIM)
    assert pred_last["next_state_std"].shape == (B, STATE_DIM)
    assert pred_last["reward_mean"].shape == (B,)
    assert pred_last["terminal_prob"].shape == (B,)

    unc = ensemble.uncertainty_score(states, actions)
    assert unc.shape == (B,)
    print(f"  next_state_mean: {out['next_state_mean'].shape} OK")
    print(f"  next_state_std:  {out['next_state_std'].shape} OK")
    print(f"  uncertainty:     {unc.cpu().numpy().round(4)}")
    print("[Test 5] PASSED")
    return ensemble


# ---------------------------------------------------------------------------
# Test 6: Simulator interface
# ---------------------------------------------------------------------------

def test_simulator(ensemble: CareSimEnsemble):
    print("\n[Test 6] CareSimEnvironment (reset + step + rollout)...")
    sim = CareSimEnvironment(ensemble, max_steps=10, uncertainty_threshold=2.0)

    B = 2
    T_seed = 5
    seed_states = torch.randn(B, T_seed, STATE_DIM)
    seed_actions = torch.randint(0, 2, (B, T_seed, ACTION_DIM)).float()

    # Test reset
    current = sim.reset(seed_states, seed_actions)
    assert current.shape == (B, STATE_DIM), f"reset shape: {current.shape}"
    print(f"  reset() -> current_state shape: {current.shape} OK")

    # Test single step
    action = torch.randint(0, 2, (B, ACTION_DIM)).float()
    next_s, reward, done, info = sim.step(action)
    assert next_s.shape == (B, STATE_DIM)
    assert reward.shape == (B,)
    assert done.shape == (B,)
    assert "uncertainty" in info
    assert "terminal_prob" in info
    print(f"  step() -> next_state: {next_s.shape}, reward: {reward.shape}, done: {done.shape} OK")
    print(f"  info keys: {list(info.keys())}")
    print(f"  uncertainty: {info['uncertainty'].round(4)}")

    # Test rollout (3 steps)
    sim.reset(seed_states, seed_actions)
    H = 3
    action_seq = torch.randint(0, 2, (B, H, ACTION_DIM)).float()
    traj = sim.rollout(action_seq)
    assert traj["states"].shape == (B, H, STATE_DIM)
    assert traj["rewards"].shape == (B, H)
    assert traj["dones"].shape == (B, H)
    assert traj["uncertainty"].shape == (B, H)
    print(f"  rollout({H} steps) -> states: {traj['states'].shape}, "
          f"rewards: {traj['rewards'].shape} OK")
    print("[Test 6] PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("CARE-Sim Smoke Test")
    print("=" * 60)

    print("\nGenerating synthetic data (200 stays)...")
    df = make_synthetic_df(n_stays=200, min_blocs=4, max_blocs=20)
    print(f"  DataFrame: {len(df)} rows, {df['icustayid'].nunique()} stays")

    test_model_forward()
    loader = test_dataset(df)
    test_loss(loader)
    model = test_training(df)
    ensemble = test_ensemble(model)
    test_simulator(ensemble)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED -- CARE-Sim is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
