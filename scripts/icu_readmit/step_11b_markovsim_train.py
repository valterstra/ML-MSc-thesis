"""Step 11b -- Train the selected causal Markov simulator baseline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.markovsim.model import MarkovSimConfig, SELECTED_CAUSAL_ACTION_MASK
from careai.icu_readmit.markovsim.train import fit_markovsim_from_dataframe


def resolve_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def parse_args():
    p = argparse.ArgumentParser(description="Step 11b: train selected causal Markov ICU simulator baseline")
    p.add_argument("--data", default="data/processed/icu_readmit/rl_dataset_selected.parquet")
    p.add_argument("--save-dir", default="models/icu_readmit/markovsim_selected_causal")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--terminal-c", type=float, default=1.0)
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.data = resolve_repo_path(args.data)
    args.save_dir = resolve_repo_path(args.save_dir)
    t0 = time.time()

    df = pd.read_parquet(args.data)
    if args.smoke:
        train_df = df[df["split"] == "train"].copy()
        keep_stays = train_df["icustayid"].drop_duplicates().iloc[:500]
        df = df[df["icustayid"].isin(keep_stays)].copy()

    config = MarkovSimConfig(
        ridge_alpha=args.ridge_alpha,
        terminal_c=args.terminal_c,
        max_iter=args.max_iter,
    )
    ensemble, metrics = fit_markovsim_from_dataframe(df, split="train", config=config)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(str(save_dir))

    meta = {
        "data": args.data,
        "save_dir": args.save_dir,
        "smoke": args.smoke,
        "causal_action_mask": SELECTED_CAUSAL_ACTION_MASK.tolist(),
        "config": metrics["config"],
        "train_metrics": {k: v for k, v in metrics.items() if k != "config"},
        "total_seconds": round(time.time() - t0, 1),
    }
    with open(save_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
