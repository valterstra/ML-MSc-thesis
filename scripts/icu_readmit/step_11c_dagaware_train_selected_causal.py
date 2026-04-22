"""
Step 11c -- Train DAG-aware temporal transformer world model on the selected ICU track.

This is the DAG-aware parallel simulator family:
  - selected 9-state / 5-action schema
  - static confounders frozen as context-only tokens
  - no reward head
  - node-time DAG masking for structural action constraints
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.caresim.dataset import ICUSequenceDataset, collate_sequences
from careai.icu_readmit.dagaware.model import DAGAwareTemporalWorldModel
from careai.icu_readmit.dagaware.train import EnsembleTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Step 11c: train DAG-aware temporal world model")
    parser.add_argument("--data", default="data/processed/icu_readmit/rl_dataset_selected.parquet")
    parser.add_argument("--save-dir", default="models/icu_readmit/dagaware_selected_causal")
    parser.add_argument("--n-models", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=80)
    parser.add_argument(
        "--action-mask-placement",
        default="final_only",
        choices=["final_only", "first_only", "all_layers"],
        help="Where to inject action-causal attention while keeping the existing history mask unchanged.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w-state", type=float, default=1.0)
    parser.add_argument("--w-term", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detect if omitted)")
    parser.add_argument("--train-window-mode", default="random", choices=["last", "random"])
    parser.add_argument("--val-window-mode", default="last", choices=["last", "random"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    t0 = time.time()
    print(f"Device: {device}")
    print(f"Data:   {args.data}")
    print(f"Save:   {args.save_dir}")
    print(f"Ensemble: {args.n_models} models x {args.n_epochs} epochs")
    print(f"Architecture: d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"Action mask placement: {args.action_mask_placement}")
    print(f"Windowing: train={args.train_window_mode} val={args.val_window_mode}")

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
    print(f"  Train: {len(train_ds)} stays  Val: {len(val_ds)} stays  [{time.time() - t0:.1f}s]")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=0,
    )

    model_kwargs = {
        "state_dim": 9,
        "action_dim": 5,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "max_seq_len": args.max_seq_len,
        "action_mask_placement": args.action_mask_placement,
        "dynamic_state_idx": (0, 1, 2, 3, 4, 5),
        "static_state_idx": (6, 7, 8),
        "use_time_feature": True,
        "predict_reward": False,
    }
    dummy = DAGAwareTemporalWorldModel(**model_kwargs)
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
        w_reward=0.0,
        w_term=args.w_term,
        grad_clip=args.grad_clip,
    )
    trainer.save(args.save_dir)

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
        "w_term": args.w_term,
        "train_window_mode": args.train_window_mode,
        "val_window_mode": args.val_window_mode,
        "device": str(device),
        "total_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Total time: {(time.time() - t0) / 60:.1f} min")
    print(f"Ensemble saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
