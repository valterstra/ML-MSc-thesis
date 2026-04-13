"""
Step 14 -- Train CARE-Sim: GPT transformer world model for ICU patient simulation.

Trains an ensemble of N CareSimGPT models on the Tier-2 RL trajectory dataset.
Each model learns to predict (next_state, reward, terminal) from the full
history of (state, action) pairs for a patient's ICU stay.

Architecture:
  - Causal GPT-2 transformer (d_model=128, 4 layers, 8 heads)
  - Continuous linear state/action embeddings (no discretization)
  - Ensemble of 5 independent models for uncertainty quantification

Usage (CPU, background with log):
  python scripts/icu_readmit/step_14_caresim_train.py 2>&1 | tee logs/step_14_caresim.log

Usage (GPU):
  python scripts/icu_readmit/step_14_caresim_train.py --device cuda --d-model 256

Outputs:
  models/icu_readmit/caresim/member_0/model.pt  through member_4/model.pt
  models/icu_readmit/caresim/ensemble_config.json
"""
import sys
import os
import argparse
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
from torch.utils.data import DataLoader

from careai.icu_readmit.caresim.model import CareSimGPT
from careai.icu_readmit.caresim.dataset import ICUSequenceDataset, collate_sequences
from careai.icu_readmit.caresim.train import EnsembleTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train CARE-Sim GPT world model (ensemble)")
    p.add_argument("--data",       default="data/processed/icu_readmit/rl_dataset_tier2.parquet")
    p.add_argument("--save-dir",   default="models/icu_readmit/caresim")
    p.add_argument("--n-models",   type=int, default=5)
    p.add_argument("--d-model",    type=int, default=128)
    p.add_argument("--n-heads",    type=int, default=8)
    p.add_argument("--n-layers",   type=int, default=4)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--max-seq-len",type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-epochs",   type=int, default=30)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--w-state",    type=float, default=1.0, help="MSE weight for next_state loss")
    p.add_argument("--w-reward",   type=float, default=0.5, help="MSE weight for reward loss")
    p.add_argument("--w-term",     type=float, default=0.5, help="BCE weight for terminal loss")
    p.add_argument("--grad-clip",  type=float, default=1.0)
    p.add_argument("--device",     default=None, help="cuda or cpu (auto-detected if omitted)")
    p.add_argument("--state-dim",  type=int, default=8,  help="number of state features")
    p.add_argument("--action-dim", type=int, default=4,  help="number of action features")
    p.add_argument("--train-window-mode", default="last", choices=["last", "random"],
                   help="How to expose long stays during training")
    p.add_argument("--val-window-mode", default="last", choices=["last", "random"],
                   help="How to expose long stays during validation")
    p.add_argument("--causal-constraints", action="store_true", default=False,
                   help="Add FCI-masked action residual layer (causal variant). "
                        "When off (default), trains unconstrained baseline.")
    p.add_argument("--freeze-static-context", action="store_true", default=False,
                   help="Do not predict static confounders; copy them through from the current state")
    p.add_argument("--use-time-feature", action="store_true", default=False,
                   help="Add elapsed bloc index as an explicit model input")
    p.add_argument("--predict-reward", action="store_true", default=True,
                   help="Train a reward head alongside next-state and terminal")
    p.add_argument("--no-predict-reward", dest="predict_reward", action="store_false",
                   help="Disable the reward head and train only next-state + terminal")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    causal = args.causal_constraints
    print(f"Device: {device}")
    print(f"Data:   {args.data}")
    print(f"Save:   {args.save_dir}")
    print(f"Ensemble: {args.n_models} models x {args.n_epochs} epochs")
    print(f"Architecture: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"Causal constraints: {'ON (FCI-masked action residual)' if causal else 'OFF (unconstrained baseline)'}")
    print(f"Static context: {'frozen / conditioning-only' if args.freeze_static_context else 'predicted in next-state head'}")
    print(f"Time feature: {'ON (elapsed bloc index)' if args.use_time_feature else 'OFF'}")
    print(f"Reward head: {'ON' if args.predict_reward else 'OFF'}")
    print(f"Windowing: train={args.train_window_mode} val={args.val_window_mode}")

    # --- Load data ---
    t0 = time.time()
    print("\nLoading data...")
    train_ds = ICUSequenceDataset.from_parquet(
        args.data,
        split="train",
        max_seq_len=args.max_seq_len,
        window_mode=args.train_window_mode,
    )
    val_ds = ICUSequenceDataset.from_parquet(
        args.data,
        split="val",
        max_seq_len=args.max_seq_len,
        window_mode=args.val_window_mode,
    )
    print(f"  Train: {len(train_ds)} stays  Val: {len(val_ds)} stays  [{time.time()-t0:.1f}s]")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_sequences, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_sequences, num_workers=0,
    )

    # --- Build and train ensemble ---
    model_kwargs = {
        "state_dim":               args.state_dim,
        "action_dim":              args.action_dim,
        "d_model":                 args.d_model,
        "n_heads":                 args.n_heads,
        "n_layers":                args.n_layers,
        "dropout":                 args.dropout,
        "max_seq_len":             args.max_seq_len,
        "use_causal_constraints":  causal,
        "freeze_static_context":   args.freeze_static_context,
        "use_time_feature":        args.use_time_feature,
        "predict_reward":          args.predict_reward,
    }
    # Count parameters once
    dummy = CareSimGPT(**model_kwargs)
    print(f"  Model parameters: {dummy.count_parameters():,} per member")
    del dummy

    trainer = EnsembleTrainer(n_models=args.n_models, model_kwargs=model_kwargs)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir,
        verbose=True,
        w_state=args.w_state,
        w_reward=args.w_reward,
        w_term=args.w_term,
        grad_clip=args.grad_clip,
    )

    # --- Save ---
    trainer.save(args.save_dir)

    # --- Save run metadata ---
    meta = {
        "data": args.data,
        "n_train_stays": len(train_ds),
        "n_val_stays": len(val_ds),
        "model_kwargs": model_kwargs,
        "n_models": args.n_models,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "w_state": args.w_state,
        "w_reward": args.w_reward,
        "w_term": args.w_term,
        "causal_constraints": causal,
        "train_window_mode": args.train_window_mode,
        "val_window_mode": args.val_window_mode,
        "freeze_static_context": args.freeze_static_context,
        "use_time_feature": args.use_time_feature,
        "predict_reward": args.predict_reward,
        "device": str(device),
        "total_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Total time: {(time.time()-t0)/60:.1f} min")
    print(f"Ensemble saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
