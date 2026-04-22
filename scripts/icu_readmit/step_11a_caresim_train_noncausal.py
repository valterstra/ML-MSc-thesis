from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from careai.icu_readmit.caresim.noncausal_dataset import (
    NonCausalICUSequenceDataset,
    collate_noncausal_sequences,
)
from careai.icu_readmit.caresim.noncausal_model import NonCausalCareSimTransformer
from careai.icu_readmit.caresim.noncausal_train import train_noncausal_model


STATIC_CONTINUOUS_BASE = {
    "age",
    "Weight_kg",
    "charlson_score",
    "prior_ed_visits_6m",
}
STATIC_BINARY_BASE = {
    "gender",
    "re_admission",
}
SPLIT_COL = "split"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train non-causal CARE-Sim transformer")
    p.add_argument("--data", default="data/processed/icu_readmit/step_10_noncausal/rl_dataset_noncausal.parquet")
    p.add_argument("--save-dir", default="models/icu_readmit/caresim_noncausal")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-seq-len", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w-state", type=float, default=1.0)
    p.add_argument("--w-term", type=float, default=0.2)
    p.add_argument("--w-readmit", type=float, default=0.2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default=None)
    p.add_argument("--train-window-mode", default="random", choices=["last", "random"])
    p.add_argument("--val-window-mode", default="last", choices=["last", "random"])
    p.add_argument("--categorical-embed-dim", type=int, default=8)
    p.add_argument("--use-time-feature", action="store_true", default=True)
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)
    p.add_argument("--fail-on-invalid-data", action="store_true", default=True)
    p.add_argument("--repair-invalid-data", action="store_true", default=True)
    return p.parse_args()


def _infer_schema_from_parquet(path: str) -> tuple[list[str], list[str], list[str], list[str], list[str], list[int]]:
    cols = list(pd.read_parquet(path).head(0).columns)

    state_cols = [c for c in cols if c.startswith("s_") and not c.startswith("s_next_")]
    static_state_cols = [
        c for c in state_cols
        if c.endswith("_code")
        or c.removeprefix("s_") in STATIC_CONTINUOUS_BASE
        or c.removeprefix("s_") in STATIC_BINARY_BASE
    ]
    dynamic_state_cols = [c for c in state_cols if c not in static_state_cols]
    next_state_cols = [f"s_next_{c.removeprefix('s_')}" for c in dynamic_state_cols]
    categorical_state_cols = [c for c in static_state_cols if c.endswith("_code")]
    numeric_state_cols = [c for c in state_cols if c not in categorical_state_cols]
    action_cols = [c for c in cols if c.endswith("_active")]

    if not state_cols:
        raise ValueError("No s_* state columns found in replay parquet")
    if not action_cols:
        raise ValueError("No *_active action columns found in replay parquet")
    missing_next = [c for c in next_state_cols if c not in cols]
    if missing_next:
        raise ValueError(f"Missing expected dynamic s_next_* targets: {missing_next[:10]}")

    sample = pd.read_parquet(path, columns=categorical_state_cols) if categorical_state_cols else pd.DataFrame()
    cat_cardinalities = [int(sample[col].max()) + 1 for col in categorical_state_cols]
    return state_cols, next_state_cols, numeric_state_cols, categorical_state_cols, action_cols, cat_cardinalities


def _validate_training_data(
    df: pd.DataFrame,
    state_cols: list[str],
    next_state_cols: list[str],
    action_cols: list[str],
) -> None:
    cols = [*state_cols, *next_state_cols, *action_cols, "done", "readmit_30d"]
    problems = []
    for col in cols:
        series = df[col]
        missing = int(series.isna().sum())
        infs = 0
        if pd.api.types.is_numeric_dtype(series):
            infs = int(np.isinf(series.to_numpy(dtype=np.float64, copy=False)).sum())
        if missing or infs:
            problems.append((col, missing, infs))
    if problems:
        preview = ", ".join(f"{c}(na={na},inf={inf})" for c, na, inf in problems[:10])
        raise ValueError(f"Training parquet contains invalid values: {preview}")


def _repair_training_data(
    df: pd.DataFrame,
    state_cols: list[str],
    next_state_cols: list[str],
    action_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    repaired_cols: list[str] = []
    out = df.copy()

    mean_fill_cols = [*state_cols, *next_state_cols]
    zero_fill_cols = [*action_cols, "done", "readmit_30d"]

    train_mask = out[SPLIT_COL] == "train" if SPLIT_COL in out.columns else pd.Series(True, index=out.index)

    for col in mean_fill_cols:
        if col not in out.columns:
            continue
        series = out[col].astype(float)
        missing = int(series.isna().sum())
        inf_mask = np.isinf(series.to_numpy(dtype=np.float64, copy=False))
        inf_count = int(inf_mask.sum())
        if missing or inf_count:
            finite_series = series.where(~pd.Series(inf_mask, index=series.index), np.nan)
            fill_value = finite_series[train_mask].mean()
            if not np.isfinite(fill_value):
                fill_value = finite_series.mean()
            if not np.isfinite(fill_value):
                fill_value = 0.0
            out[col] = finite_series.fillna(float(fill_value)).astype(np.float32)
            repaired_cols.append(col)

    for col in zero_fill_cols:
        if col not in out.columns:
            continue
        series = out[col]
        missing = int(series.isna().sum())
        inf_count = int(np.isinf(series.to_numpy(dtype=np.float64, copy=False)).sum()) if pd.api.types.is_numeric_dtype(series) else 0
        if missing or inf_count:
            numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
            out[col] = numeric.astype(series.dtype if pd.api.types.is_numeric_dtype(series) else np.int8)
            repaired_cols.append(col)

    return out, repaired_cols


def main() -> None:
    args = parse_args()
    t0 = time.time()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    os.makedirs(args.save_dir, exist_ok=True)

    (
        state_cols,
        next_state_cols,
        numeric_state_cols,
        categorical_state_cols,
        action_cols,
        cat_cardinalities,
    ) = _infer_schema_from_parquet(args.data)

    print(f"Device: {device}")
    print(f"Data:   {args.data}")
    print(f"Save:   {args.save_dir}")

    df = pd.read_parquet(args.data)
    print(f"Loaded replay parquet rows: {len(df)}")
    if args.repair_invalid_data:
        df, repaired_cols = _repair_training_data(df, state_cols, next_state_cols, action_cols)
        if repaired_cols:
            preview = repaired_cols[:10]
            print(f"Repaired invalid values in {len(repaired_cols)} columns: {preview}")
        else:
            print("No replay repairs needed.")
    if args.fail_on_invalid_data:
        _validate_training_data(df, state_cols, next_state_cols, action_cols)
        print("Replay data validation passed.")

    train_ds = NonCausalICUSequenceDataset(
        df=df,
        state_cols=state_cols,
        action_cols=action_cols,
        next_state_cols=next_state_cols,
        split="train",
        max_seq_len=args.max_seq_len,
        window_mode=args.train_window_mode,
    )
    val_ds = NonCausalICUSequenceDataset(
        df=df,
        state_cols=state_cols,
        action_cols=action_cols,
        next_state_cols=next_state_cols,
        split="val",
        max_seq_len=args.max_seq_len,
        window_mode=args.val_window_mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_noncausal_sequences,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_noncausal_sequences,
        num_workers=0,
    )

    numeric_idx = [state_cols.index(col) for col in numeric_state_cols]
    categorical_idx = [state_cols.index(col) for col in categorical_state_cols]

    model = NonCausalCareSimTransformer(
        state_dim=len(state_cols),
        action_dim=len(action_cols),
        dynamic_state_dim=len(next_state_cols),
        numeric_state_idx=numeric_idx,
        categorical_state_idx=categorical_idx,
        categorical_cardinalities=cat_cardinalities,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        use_time_feature=args.use_time_feature,
        categorical_embed_dim=args.categorical_embed_dim,
    )
    print(f"Train stays: {len(train_ds)}  Val stays: {len(val_ds)}")
    print(f"State dim: {len(state_cols)}  Dynamic target dim: {len(next_state_cols)}  Action dim: {len(action_cols)}")
    print(f"Parameters: {model.count_parameters():,}")

    history, best_state = train_noncausal_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        lr=args.lr,
        device=device,
        w_state=args.w_state,
        w_term=args.w_term,
        w_readmit=args.w_readmit,
        grad_clip=args.grad_clip,
        verbose=True,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )

    final_model_path = Path(args.save_dir) / "model.pt"
    best_model_path = Path(args.save_dir) / "best_model.pt"
    torch.save(model.state_dict(), final_model_path)
    if best_state is not None:
        torch.save(best_state, best_model_path)

    config = {
        "data": args.data,
        "state_cols": state_cols,
        "next_state_cols": next_state_cols,
        "numeric_state_cols": numeric_state_cols,
        "categorical_state_cols": categorical_state_cols,
        "action_cols": action_cols,
        "categorical_cardinalities": cat_cardinalities,
        "model_kwargs": {
            "state_dim": len(state_cols),
            "action_dim": len(action_cols),
            "dynamic_state_dim": len(next_state_cols),
            "numeric_state_idx": numeric_idx,
            "categorical_state_idx": categorical_idx,
            "categorical_cardinalities": cat_cardinalities,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "max_seq_len": args.max_seq_len,
            "use_time_feature": args.use_time_feature,
            "categorical_embed_dim": args.categorical_embed_dim,
        },
        "training": {
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "lr": args.lr,
            "w_state": args.w_state,
            "w_term": args.w_term,
            "w_readmit": args.w_readmit,
            "grad_clip": args.grad_clip,
            "train_window_mode": args.train_window_mode,
            "val_window_mode": args.val_window_mode,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
        },
        "dataset": {
            "n_train_stays": len(train_ds),
            "n_val_stays": len(val_ds),
        },
        "device": str(device),
        "total_seconds": round(time.time() - t0, 1),
    }

    with open(Path(args.save_dir) / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(Path(args.save_dir) / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved final model to {final_model_path}")
    if best_state is not None:
        print(f"Saved best model to  {best_model_path}")
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
