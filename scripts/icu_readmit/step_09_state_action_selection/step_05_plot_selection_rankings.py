"""
Step 05 -- Plot step 09 selected-state rankings.

PURPOSE
-------
Create thesis-ready bar charts for the step 09 selection stage:

  1. Top states from the initial LightGBM readmission ranking
  2. Top states from the FCI readmission stability ranking

This script is intentionally specific to the step 09 selection pipeline.
It reads the existing ranking outputs produced by the earlier step 09 scripts
and writes plots into a dedicated figures directory under the step 09 report tree.

DEFAULT INPUTS
--------------
LightGBM:
  reports/icu_readmit/step_09_state_action_selection/state_variable_ranking.csv

FCI:
  Prefer robust Step 03b output if available:
    reports/icu_readmit/step_09_state_action_selection/random_stability_results_robust.csv
  Otherwise fall back to the legacy Step 03 output:
    reports/icu_readmit/step_09_state_action_selection/random_stability_results.csv

OUTPUTS
-------
  reports/icu_readmit/step_09_state_action_selection/figures/
    step_09_lightgbm_top_states.png
    step_09_lightgbm_top_states.pdf
    step_09_fci_top_states.png
    step_09_fci_top_states.pdf
    step_09_selection_overview.png
    step_09_selection_overview.pdf
    step_09_lightgbm_top_states.csv
    step_09_fci_top_states.csv

Usage:
    python scripts/icu_readmit/step_09_state_action_selection/step_05_plot_selection_rankings.py
    python scripts/icu_readmit/step_09_state_action_selection/step_05_plot_selection_rankings.py --top-n 10
    python scripts/icu_readmit/step_09_state_action_selection/step_05_plot_selection_rankings.py --report-dir reports/icu_readmit/step_09_state_action_selection
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


STATE_LABEL_MAP = {
    "Hb": "Hb",
    "BUN": "BUN",
    "Creatinine": "Creatinine",
    "Phosphate": "Phosphate",
    "HR": "HR",
    "Chloride": "Chloride",
    "Shock_Index": "Shock Index",
    "Ht": "Ht",
    "PT": "PT",
}

SELECTED_STATE_VARS = {"Hb", "BUN", "Creatinine", "Phosphate", "HR", "Chloride"}
BORDERLINE_STATE_VARS = {"Shock_Index"}

COLOR_SELECTED = "#2f6f8f"
COLOR_BORDERLINE = "#d08c3b"
COLOR_OTHER = "#b7bcc2"


def pick_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def pretty_state_name(name: str) -> str:
    if name.startswith("last_"):
        name = name.removeprefix("last_")
    if name.startswith("delta_"):
        name = name.removeprefix("delta_")
    return STATE_LABEL_MAP.get(name, name.replace("_", " "))


def state_category(variable: str) -> str:
    raw = variable.removeprefix("last_").removeprefix("delta_")
    if raw in SELECTED_STATE_VARS:
        return "selected"
    if raw in BORDERLINE_STATE_VARS:
        return "borderline"
    return "other"


def category_color(category: str) -> str:
    if category == "selected":
        return COLOR_SELECTED
    if category == "borderline":
        return COLOR_BORDERLINE
    return COLOR_OTHER


def load_state_ranking(report_dir: Path, top_n: int) -> tuple[pd.DataFrame, Path]:
    path = report_dir / "state_variable_ranking.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Could not find the LightGBM state ranking CSV in "
            f"{report_dir}"
        )

    df = pd.read_csv(path)
    required = {"variable", "importance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    state_df = (
        df.sort_values(
            ["importance", "rank", "variable"],
            ascending=[False, True, True],
        )
        .head(top_n)
        .copy()
    )
    state_df = state_df.drop(columns=["rank"], errors="ignore")
    state_df.insert(0, "rank", range(1, len(state_df) + 1))
    state_df["label"] = state_df["variable"].map(pretty_state_name)
    state_df["category"] = state_df["variable"].map(state_category)
    state_df["bar_color"] = state_df["category"].map(category_color)
    return state_df, path


def load_fci_ranking(report_dir: Path, top_n: int) -> tuple[pd.DataFrame, Path]:
    path = pick_existing_path([
        report_dir / "random_stability_results_robust.csv",
        report_dir / "random_stability_results.csv",
    ])
    if path is None:
        raise FileNotFoundError(
            "Could not find an FCI state ranking CSV in "
            f"{report_dir}"
        )

    df = pd.read_csv(path)
    required = {"variable", "freq_definite", "freq_any", "n_runs_included"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    fci_df = (
        df.sort_values(
            ["freq_definite", "freq_any", "n_runs_included", "variable"],
            ascending=[False, False, False, True],
        )
        .head(top_n)
        .copy()
    )
    fci_df.insert(0, "rank", range(1, len(fci_df) + 1))
    fci_df["label"] = fci_df["variable"].map(pretty_state_name)
    fci_df["category"] = fci_df["variable"].map(state_category)
    fci_df["bar_color"] = fci_df["category"].map(category_color)
    return fci_df, path


def style_axes(ax: plt.Axes, xlabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#d7d7d7", linewidth=0.8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#666666")


def plot_ranked_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    xlabel: str,
    value_formatter,
) -> None:
    plot_df = df.sort_values(value_col, ascending=True).copy()
    bars = ax.barh(
        plot_df[label_col],
        plot_df[value_col],
        color=plot_df["bar_color"],
        edgecolor="#2f2f2f",
        linewidth=0.6,
    )
    ax.bar_label(
        bars,
        labels=[value_formatter(value) for value in plot_df[value_col]],
        padding=4,
        fontsize=9,
        color="#333333",
    )
    style_axes(ax, xlabel)
    max_value = float(plot_df[value_col].max()) if len(plot_df) else 0.0
    ax.set_xlim(0, max_value * 1.15 if max_value > 0 else 1.0)


def add_selection_legend(ax: plt.Axes) -> None:
    handles = [
        Patch(facecolor=COLOR_SELECTED, edgecolor="#2f2f2f", label="Selected main set"),
        Patch(facecolor=COLOR_BORDERLINE, edgecolor="#2f2f2f", label="Borderline"),
        Patch(facecolor=COLOR_OTHER, edgecolor="#2f2f2f", label="Ranked but not selected"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")


def main(args: argparse.Namespace) -> None:
    report_dir = Path(args.report_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gbm_df, gbm_source = load_state_ranking(report_dir, args.top_n)
    fci_df, fci_source = load_fci_ranking(report_dir, args.top_n)

    log.info("LightGBM ranking source: %s", gbm_source)
    log.info("FCI ranking source: %s", fci_source)
    log.info("Writing figures to: %s", out_dir)

    gbm_out = out_dir / "step_09_lightgbm_top_states"
    fci_out = out_dir / "step_09_fci_top_states"
    overview_out = out_dir / "step_09_selection_overview"

    gbm_df.to_csv(gbm_out.with_suffix(".csv"), index=False)
    fci_df.to_csv(fci_out.with_suffix(".csv"), index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.titlepad": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    # LightGBM figure.
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.45 * len(gbm_df) + 1.2)))
    plot_ranked_bars(
        ax,
        gbm_df,
        label_col="label",
        value_col="importance",
        xlabel="LightGBM feature importance",
        value_formatter=lambda v: f"{v:.3f}",
    )
    ax.set_ylabel("")
    ax.invert_yaxis()
    add_selection_legend(ax)
    save_figure(fig, gbm_out)
    plt.close(fig)

    # FCI figure.
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.45 * len(fci_df) + 1.2)))
    plot_ranked_bars(
        ax,
        fci_df,
        label_col="label",
        value_col="freq_definite",
        xlabel="FCI definite edge frequency",
        value_formatter=lambda v: f"{v:.1%}",
    )
    ax.set_ylabel("")
    ax.invert_yaxis()
    add_selection_legend(ax)
    save_figure(fig, fci_out)
    plt.close(fig)

    # Combined overview figure for thesis reuse.
    height = max(6.5, 0.45 * max(len(gbm_df), len(fci_df)) + 1.4)
    fig, axes = plt.subplots(1, 2, figsize=(16, height), constrained_layout=True)

    plot_ranked_bars(
        axes[0],
        gbm_df,
        label_col="label",
        value_col="importance",
        xlabel="LightGBM feature importance",
        value_formatter=lambda v: f"{v:.3f}",
    )
    axes[0].set_ylabel("")
    axes[0].invert_yaxis()
    add_selection_legend(axes[0])

    plot_ranked_bars(
        axes[1],
        fci_df,
        label_col="label",
        value_col="freq_definite",
        xlabel="FCI definite edge frequency",
        value_formatter=lambda v: f"{v:.1%}",
    )
    axes[1].set_ylabel("")
    axes[1].invert_yaxis()
    add_selection_legend(axes[1])

    save_figure(fig, overview_out)
    plt.close(fig)

    log.info("Saved: %s", gbm_out.with_suffix(".png"))
    log.info("Saved: %s", fci_out.with_suffix(".png"))
    log.info("Saved: %s", overview_out.with_suffix(".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "icu_readmit" / "step_09_state_action_selection"),
        help="Step 09 report directory containing the ranking outputs.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "reports" / "icu_readmit" / "step_09_state_action_selection" / "figures"),
        help="Directory for the generated thesis figures.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top states/actions to plot.",
    )
    args = parser.parse_args()
    main(args)
