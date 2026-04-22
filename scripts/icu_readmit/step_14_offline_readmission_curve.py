"""Step 14 -- Offline DDQN checkpoint readmission curve.

Evaluate saved offline DDQN checkpoints in a chosen simulator and plot the
estimated fixed-horizon readmission risk over training progress.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_step14_module():
    path = PROJECT_ROOT / "scripts" / "icu_readmit" / "step_14_policy_readmission_estimate.py"
    spec = importlib.util.spec_from_file_location("step_14_policy_readmission_estimate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 14: offline DDQN readmission curve")
    parser.add_argument("--data", default="data/processed/icu_readmit/rl_dataset_selected.parquet")
    parser.add_argument("--offline-ddqn-dir", default="models/icu_readmit/offline_selected/ddqn")
    parser.add_argument("--terminal-model-dir", default="models/icu_readmit/terminal_readmit_selected")
    parser.add_argument("--caresim-model-dir", default="models/icu_readmit/caresim_selected_causal")
    parser.add_argument("--report-dir", default="reports/icu_readmit/offline_selected")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--history-len", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def resolve_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def checkpoint_step(path: Path) -> int:
    stem = path.stem
    return int(stem.split("_")[-1])


def main() -> None:
    args = build_parser().parse_args()
    args.data = resolve_repo_path(args.data)
    args.offline_ddqn_dir = resolve_repo_path(args.offline_ddqn_dir)
    args.terminal_model_dir = resolve_repo_path(args.terminal_model_dir)
    args.caresim_model_dir = resolve_repo_path(args.caresim_model_dir)
    args.report_dir = resolve_repo_path(args.report_dir)

    helper = load_step14_module()
    device = helper.resolve_device(args.device)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    episodes = helper.load_seed_episodes(
        data_path=args.data,
        split=args.split,
        history_len=args.history_len,
        max_episodes=args.max_episodes,
        seed=args.seed,
    )
    terminal_model = helper.LightGBMReadmitModel.from_dir(
        args.terminal_model_dir,
        state_feature_names=helper.STATE_COLS,
        device=device,
    )
    evaluator = helper.EvaluatorSpec(
        name="caresim",
        ensemble=helper.CareSimEnsemble.from_dir(args.caresim_model_dir, device=device),
        env_cls=helper.CareSimEnvironment,
    )

    ckpt_dir = Path(args.offline_ddqn_dir)
    checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pt"), key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_*.pt files found in {ckpt_dir}")

    curve_rows = []
    for ckpt in checkpoints:
        model = helper.load_ddqn(str(ckpt), device=device)
        result, _detail = helper.evaluate_policy_in_env(
            evaluator,
            episodes,
            policy_name=f"offline_ddqn_{checkpoint_step(ckpt)}",
            model=model,
            terminal_model=terminal_model,
            horizon=args.horizon,
            device=device,
        )
        curve_rows.append(
            {
                "checkpoint_step": checkpoint_step(ckpt),
                "mean_predicted_readmit": result["mean_predicted_readmit"],
                "mean_predicted_readmit_pct": result["mean_predicted_readmit_pct"],
                "std_predicted_readmit": result["std_predicted_readmit"],
            }
        )

    curve_df = pd.DataFrame(curve_rows).sort_values("checkpoint_step").reset_index(drop=True)
    stay_df = pd.read_parquet(args.data, columns=["icustayid", "split", "readmit_30d"])[["icustayid", "split", "readmit_30d"]]
    observed_rate = float(stay_df.drop_duplicates("icustayid").query("split == @args.split")["readmit_30d"].mean())

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=args.dpi)
    ax.plot(curve_df["checkpoint_step"], curve_df["mean_predicted_readmit_pct"], marker="o", linewidth=2, color="#1f77b4")
    ax.axhline(100.0 * observed_rate, color="#d62728", linestyle="--", linewidth=1.8, label=f"Observed {args.split} readmission")
    ax.set_title("Offline DDQN fixed-horizon estimated readmission")
    ax.set_xlabel("Training step checkpoint")
    ax.set_ylabel("Estimated readmission (%)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    png_path = report_dir / f"step_14_offline_readmission_curve_{args.split}.png"
    csv_path = report_dir / f"step_14_offline_readmission_curve_{args.split}.csv"
    json_path = report_dir / f"step_14_offline_readmission_curve_{args.split}.json"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    curve_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "history_len": args.history_len,
                "horizon": args.horizon,
                "max_episodes": args.max_episodes,
                "evaluator": "caresim",
                "observed_readmit_rate": observed_rate,
                "observed_readmit_rate_pct": 100.0 * observed_rate,
                "curve": curve_rows,
            },
            f,
            indent=2,
        )

    print(f"Saved plot: {png_path}")
    print(f"Saved csv:  {csv_path}")
    print(f"Saved json: {json_path}")


if __name__ == "__main__":
    main()
