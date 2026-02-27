from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.causal.data_prep import load_and_prepare
from careai.causal.diagnostics import compute_balance_smd, diagnostics_summary, summarize_propensity, summarize_weights
from careai.causal.effect_estimators import bootstrap_effects, estimate_effects, percentile_ci
from careai.causal.reporting import summary_markdown
from careai.io.load_inputs import load_yaml, resolve_from_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate causal effect baseline (IPW/AIPW) for high vs low support.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "causal_effect_v1.yaml"))
    parser.add_argument("--input", default=None, help="Optional override for transitions CSV path.")
    parser.add_argument("--bootstrap-resamples", type=int, default=None, help="Optional override for bootstrap resamples.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print bootstrap progress every N resamples (set 0 to disable).",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg: dict[str, Any] = load_yaml(config_path)

    in_path = resolve_from_config(config_path, cfg["input"]["transitions_path"])
    if args.input:
        in_path = Path(args.input).resolve()

    out_dir = resolve_from_config(config_path, cfg["output"]["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(cfg["output"]["prefix"])

    df = pd.read_csv(in_path)
    prep = load_and_prepare(df, cfg)
    est, diag = estimate_effects(prep.X, prep.t, prep.y, cfg)

    balance_df = compute_balance_smd(prep.X, prep.t, diag["sw"])
    ps_summary_df = summarize_propensity(diag["ps"], prep.t)
    w_summary_df = summarize_weights(diag["sw_raw"], diag["sw"], prep.t)
    diag_summary = diagnostics_summary(balance_df, diag["ps"])

    n_boot = int(args.bootstrap_resamples) if args.bootstrap_resamples is not None else int(cfg["bootstrap"]["n_resamples"])
    progress_every = int(args.progress_every)
    if progress_every <= 0:
        progress_every = None
    boot_df = bootstrap_effects(
        prep.X,
        prep.t,
        prep.y,
        cfg,
        n_resamples=n_boot,
        seed=int(cfg["bootstrap"]["seed"]),
        progress_every=progress_every,
    )

    ci = {
        "ipw_rd": percentile_ci(boot_df["ipw_rd"]),
        "ipw_rr": percentile_ci(boot_df["ipw_rr"]),
        "aipw_rd": percentile_ci(boot_df["aipw_rd"]),
        "aipw_rr": percentile_ci(boot_df["aipw_rr"]),
    }

    summary = {
        "config_path": str(config_path),
        "input_path": str(in_path),
        "n_rows": int(len(prep.df)),
        "analysis": cfg["analysis"],
        "estimates": asdict(est),
        "bootstrap_n_resamples": n_boot,
        "bootstrap_ci": ci,
        "diagnostics": diag_summary,
    }

    summary_json = out_dir / f"{prefix}_summary.json"
    summary_md = out_dir / f"{prefix}_summary.md"
    balance_csv = out_dir / f"{prefix}_balance_smd.csv"
    ps_csv = out_dir / f"{prefix}_propensity_summary.csv"
    w_csv = out_dir / f"{prefix}_weight_summary.csv"
    boot_csv = out_dir / f"{prefix}_bootstrap_estimates.csv"

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(summary_markdown(summary), encoding="utf-8")
    balance_df.to_csv(balance_csv, index=False)
    ps_summary_df.to_csv(ps_csv, index=False)
    w_summary_df.to_csv(w_csv, index=False)
    if bool(cfg["bootstrap"].get("store_replicates_csv", False)):
        boot_df.to_csv(boot_csv, index=False)

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {summary_md}")
    print(f"Wrote: {balance_csv}")
    print(f"Wrote: {ps_csv}")
    print(f"Wrote: {w_csv}")
    if bool(cfg["bootstrap"].get("store_replicates_csv", False)):
        print(f"Wrote: {boot_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

