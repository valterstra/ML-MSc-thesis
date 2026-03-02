from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.io.load_inputs import load_yaml, resolve_from_config
from careai.sim_hourly.counterfactual import rollout_fixed_action_paths
from careai.sim_hourly.data import prepare_hourly_data
from careai.sim_hourly.dynamics import fit_dynamics_model
from careai.sim_hourly.readmission import fit_readmission_model


def _plot(paths: pd.DataFrame, risks: pd.DataFrame, state_cols: list[str], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    show_cols = [c for c in ["s_t_sofa", "s_t_mbp", "s_t_creatinine"] if c in state_cols]
    if not show_cols:
        show_cols = state_cols[: min(3, len(state_cols))]

    fig, axes = plt.subplots(len(show_cols) + 1, 1, figsize=(11, 3.2 * (len(show_cols) + 1)), sharex=False)
    if len(show_cols) == 0:
        axes = [axes]

    for i, col in enumerate(show_cols):
        ax = axes[i]
        for pol in paths["policy"].drop_duplicates():
            p = paths[paths["policy"] == pol]
            ax.plot(p["t"], p[col], label=pol)
        ax.set_title(col)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    ax = axes[-1]
    r = risks.sort_values("pred_readmit_risk", ascending=True)
    ax.bar(r["policy"], r["pred_readmit_risk"])
    ax.set_title("Predicted 30d Readmission Risk")
    ax.set_ylabel("Risk")
    ax.set_xlabel("Policy")
    ax.grid(alpha=0.3, axis="y")
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot fixed-action counterfactual trajectories from one initial hourly state.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "hourly_sim.yaml"))
    parser.add_argument("--transitions-input", default=None)
    parser.add_argument("--episode-input", default=None)
    parser.add_argument("--episode-id", default=None, help="Optional episode_id to choose t=0 initial state.")
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg: dict[str, Any] = load_yaml(config_path)

    transitions_path = resolve_from_config(config_path, cfg["input"]["transitions_path"])
    episode_path = resolve_from_config(config_path, cfg["input"]["episode_table_path"])
    if args.transitions_input:
        transitions_path = Path(args.transitions_input).resolve()
    if args.episode_input:
        episode_path = Path(args.episode_input).resolve()

    out_dir = resolve_from_config(config_path, cfg["output"]["report_dir"])
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    transitions = pd.read_csv(transitions_path)
    episodes = pd.read_csv(episode_path)
    hourly = prepare_hourly_data(transitions, cfg)
    dynamics = fit_dynamics_model(hourly.train, cfg, hourly.state_cols, hourly.action_cols)
    readmit = fit_readmission_model(episodes, cfg)

    starts = hourly.eval_starts[hourly.eval_starts["t"] == 0].copy()
    if args.episode_id is not None:
        starts = starts[starts["episode_id"].astype(str) == str(args.episode_id)]
        if starts.empty:
            raise ValueError(f"episode_id={args.episode_id} not found among available t=0 start rows.")
    start_row = starts.sort_values("episode_id").iloc[0]
    initial_state = start_row[hourly.state_cols].to_numpy(dtype=float)

    res = rollout_fixed_action_paths(
        initial_state=initial_state,
        state_cols=hourly.state_cols,
        dynamics=dynamics,
        readmit=readmit,
        max_steps=int(args.max_steps),
    )

    traj_csv = out_dir / "hourly_sim_counterfactual_paths.csv"
    risk_csv = out_dir / "hourly_sim_counterfactual_risks.csv"
    plot_png = out_dir / "hourly_sim_counterfactual_plot.png"
    summary_json = out_dir / "hourly_sim_counterfactual_summary.json"
    summary_md = out_dir / "hourly_sim_counterfactual_summary.md"

    res.trajectories.to_csv(traj_csv, index=False)
    res.risks.to_csv(risk_csv, index=False)

    plot_status = "ok"
    plot_error = None
    try:
        _plot(res.trajectories, res.risks, hourly.state_cols, plot_png)
    except Exception as e:  # pragma: no cover
        plot_status = "failed"
        plot_error = str(e)

    payload = {
        "config_path": str(config_path),
        "transitions_path": str(transitions_path),
        "episode_path": str(episode_path),
        "selected_episode_id": str(start_row["episode_id"]),
        "max_steps": int(args.max_steps),
        "output_paths": {
            "paths_csv": str(traj_csv),
            "risks_csv": str(risk_csv),
            "plot_png": str(plot_png),
        },
        "plot_status": plot_status,
        "plot_error": plot_error,
        "risk_summary": res.risks.sort_values("pred_readmit_risk").to_dict(orient="records"),
    }

    md_lines = [
        "# Hourly Counterfactual (Fixed Action) Summary",
        "",
        f"- selected episode_id: `{payload['selected_episode_id']}`",
        f"- max steps: `{payload['max_steps']}`",
        f"- plot status: `{payload['plot_status']}`",
        "",
        "## Predicted Readmission Risk by Policy",
    ]
    for r in payload["risk_summary"]:
        md_lines.append(f"- `{r['policy']}`: {r['pred_readmit_risk']:.4f}")

    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Wrote: {traj_csv}")
    print(f"Wrote: {risk_csv}")
    if plot_status == "ok":
        print(f"Wrote: {plot_png}")
    else:
        print(f"Plot not written: {plot_error}")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
