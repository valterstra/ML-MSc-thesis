"""
Step 04b -- Robust multivariate action -> state causal stability analysis (FCI-based).

PURPOSE
-------
Parallel track to step_04_action_state_stability.py.

This version strengthens the action -> state analysis by making each FCI run less
"thin". Instead of testing exactly one drug against one delta-state at a time, each
run samples:

  - 2 drug exposure variables
  - 2 target delta-state variables
  - baseline values for the sampled states

The graph per run is therefore:

  [age, charlson_score, prior_ed_visits_6m, num_blocs,
   first_state_1, first_state_2 | frac_drug_1, frac_drug_2 | delta_state_1, delta_state_2]

This adds robustness in three ways:
  1. drug -> state links must survive in the presence of another drug
  2. drug -> state links must survive in the presence of another changing state
  3. each delta-state is adjusted for its own baseline value

The resulting frequencies are still pairwise:
  freq_definite(drug, delta_state)

but they are estimated from multivariate random graphs rather than isolated
one-drug / one-state runs.

STATE POOL
----------
By default, this robust track keeps the same broad state pool as the original
Step 04:

  step_03 pool minus last_input_total = 28 raw state variables

So the only intended methodological change in Step 04b is the graph structure:
  - 2 drugs per run
  - 2 states per run
  - baseline-state adjustment

It does not narrow the search to the top-ranked Step 03 states by default.

ACTION POOL
-----------
Uses the same 9 clinically meaningful action classes as the original Step 04:
  vasopressor, ivfluid, antibiotic, anticoagulant, diuretic,
  steroid, insulin, sedation, mechvent

Drug representation:
  frac_active = fraction of blocs where the drug was active

State representation:
  first_state = first observed value in the stay
  delta_state = last_value - first_value

OUTPUTS
-------
All outputs are written alongside the original Step 04 outputs, but with `_robust`
suffixes so the old track stays untouched:

  reports/icu_readmit/step_09_state_action_selection/
    action_state_pair_results_robust.csv
    action_state_frequency_matrix_robust.csv
    action_state_summary_robust.json
    action_state_state_pool_robust.json
    action_state_results_robust/
      frac_<drug>_results.csv

Usage:
    python scripts/icu_readmit/step_09_state_action_selection/step_04b_action_state_stability_robust.py
    python scripts/icu_readmit/step_09_state_action_selection/step_04b_action_state_stability_robust.py --smoke
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.columns import (
    C_ICUSTAYID, C_BLOC,
    C_VASOPRESSOR_DOSE, C_IVFLUID_DOSE,
    C_ANTIBIOTIC_ACTIVE, C_ANTICOAGULANT_ACTIVE,
    C_DIURETIC_ACTIVE, C_STEROID_ACTIVE,
    C_INSULIN_ACTIVE, C_SEDATION_ACTIVE,
    C_MECHVENT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


CONFOUNDER_COLS = ["age", "charlson_score", "prior_ed_visits_6m", "num_blocs"]

DRUGS = [
    (C_VASOPRESSOR_DOSE,     "frac_vasopressor",   True),
    (C_IVFLUID_DOSE,         "frac_ivfluid",       True),
    (C_ANTIBIOTIC_ACTIVE,    "frac_antibiotic",    False),
    (C_ANTICOAGULANT_ACTIVE, "frac_anticoagulant", False),
    (C_DIURETIC_ACTIVE,      "frac_diuretic",      False),
    (C_STEROID_ACTIVE,       "frac_steroid",       False),
    (C_INSULIN_ACTIVE,       "frac_insulin",       False),
    (C_SEDATION_ACTIVE,      "frac_sedation",      False),
    (C_MECHVENT,             "frac_mechvent",      False),
]

STATE_POOL_RAW = [
    "BUN", "Hb", "Platelets_count", "WBC_count",
    "cumulated_balance", "Creatinine", "PT", "input_4hourly_tev",
    "PTT", "Glucose", "output_total", "HR", "RR",
    "Alkaline_Phosphatase", "Ht", "Temp_C", "SpO2",
    "Phosphate", "Shock_Index", "Chloride", "CO2_mEqL",
    "Fibrinogen", "SGOT", "Pain_Level", "Lymphs_pct",
    "Sodium", "paCO2", "TidalVolume_Observed",
]


def build_state_entries_from_raw(raw_states: list[str]) -> list[dict[str, str]]:
    return [
        {
            "last": f"last_{raw}",
            "raw": raw,
            "first": f"first_{raw}",
            "delta": f"delta_{raw}",
        }
        for raw in raw_states
    ]


def build_action_state_table(df: pd.DataFrame, state_pool_raw: list[str]) -> pd.DataFrame:
    """
    Collapse bloc-level data to stay-level for the robust action-state track.

    Includes:
      confounders     : age, charlson_score, prior_ed_visits_6m
      num_blocs       : stay length
      frac_drug       : fraction of blocs with drug active
      first_state     : baseline value for selected states
      delta_state     : last - first for selected states
    """
    df = df.sort_values([C_ICUSTAYID, C_BLOC])
    grp = df.groupby(C_ICUSTAYID)

    parts = []

    conf_base = ["age", "charlson_score", "prior_ed_visits_6m"]
    present_conf = [c for c in conf_base if c in df.columns]
    parts.append(grp[present_conf].first())
    parts.append(grp[C_BLOC].count().rename("num_blocs"))

    for raw_col, frac_col, is_continuous in DRUGS:
        if raw_col not in df.columns:
            log.warning("Drug column not found in dataset: %s", raw_col)
            continue
        if is_continuous:
            frac = grp[raw_col].apply(lambda x: (x > 0).mean())
        else:
            frac = grp[raw_col].mean()
        parts.append(frac.rename(frac_col))

    present_states = [s for s in state_pool_raw if s in df.columns]
    missing_states = [s for s in state_pool_raw if s not in df.columns]
    if missing_states:
        log.warning("State columns not found in dataset (skipped): %s", missing_states)

    for s in present_states:
        first = grp[s].first().rename(f"first_{s}")
        delta = (grp[s].last() - grp[s].first()).rename(f"delta_{s}")
        parts.extend([first, delta])

    return pd.concat(parts, axis=1).reset_index()


def build_background_knowledge(nodes, col_names: list[str], tier: dict[str, int]):
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    bk = BackgroundKnowledge()
    node_map = {col_names[i]: nodes[i] for i in range(len(col_names))}
    for ni in col_names:
        for nj in col_names:
            if ni != nj and tier.get(ni, 1) > tier.get(nj, 1):
                bk.add_forbidden_by_node(node_map[ni], node_map[nj])
    return bk


def run_fci_once(data: np.ndarray, col_names: list[str], tier: dict[str, int], alpha: float):
    """Two-pass FCI so background knowledge can be attached to graph nodes."""
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            g_init, _ = fci(data, fisherz, alpha, node_names=col_names, verbose=False, show_progress=False)
            nodes = g_init.get_nodes()
            bk = build_background_knowledge(nodes, col_names, tier)
            g_final, _ = fci(
                data,
                fisherz,
                alpha,
                node_names=col_names,
                background_knowledge=bk,
                verbose=False,
                show_progress=False,
            )
        return g_final
    except Exception as exc:
        log.debug("FCI failed: %s", exc)
        return None


def extract_directed_edge(g, col_names: list[str], source_col: str, target_col: str) -> str | None:
    """
    Return edge type for source_col -> target_col, or None.

    PAG encoding (causal-learn GeneralGraph, confirmed empirically):
      adj[i,j] = -1 -> arrowhead at j end
      adj[i,j] =  1 -> tail at j end
      adj[i,j] =  2 -> circle at j end
    """
    if g is None or source_col not in col_names or target_col not in col_names:
        return None

    try:
        adj = g.graph
    except AttributeError:
        return None

    si = col_names.index(source_col)
    ti = col_names.index(target_col)
    s_t = adj[si, ti]
    t_s = adj[ti, si]

    if s_t == -1 and t_s == 1:
        return "definite"
    if s_t == -1 and t_s == 2:
        return "possible"
    if s_t == -1 and t_s == -1:
        return "bidirected"
    return None


def run_multivariate_stability(
    stay_df: pd.DataFrame,
    drug_cols: list[str],
    state_entries: list[dict[str, str]],
    n_runs: int,
    n_sample: int,
    alpha: float,
    n_drugs_per_run: int,
    n_states_per_run: int,
    seed: int,
    log_every: int = 100,
) -> tuple[dict, int]:
    """
    Run the robust multivariate stability loop.

    Each run samples multiple drugs and states, then updates pairwise counts for
    every sampled (drug, delta_state) combination.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    pair_included: dict[tuple[str, str], int] = defaultdict(int)
    pair_definite: dict[tuple[str, str], int] = defaultdict(int)
    pair_possible: dict[tuple[str, str], int] = defaultdict(int)
    pair_any: dict[tuple[str, str], int] = defaultdict(int)
    n_failed = 0

    scaler = StandardScaler()
    t_start = time.time()

    for run_idx in range(n_runs):
        sampled_drugs = rng.sample(drug_cols, k=min(n_drugs_per_run, len(drug_cols)))
        sampled_states = rng.sample(state_entries, k=min(n_states_per_run, len(state_entries)))

        baseline_cols = [s["first"] for s in sampled_states if s["first"] in stay_df.columns]
        delta_cols = [s["delta"] for s in sampled_states if s["delta"] in stay_df.columns]
        state_delta_pairs = [(s["raw"], s["delta"]) for s in sampled_states if s["delta"] in stay_df.columns]

        present_conf = [c for c in CONFOUNDER_COLS if c in stay_df.columns]
        col_names = present_conf + baseline_cols + sampled_drugs + delta_cols

        tier = {c: 0 for c in present_conf + baseline_cols}
        tier.update({d: 1 for d in sampled_drugs})
        tier.update({d: 2 for d in delta_cols})

        n_draw = min(n_sample, len(stay_df))
        subset = stay_df[col_names].sample(n=n_draw, random_state=seed + run_idx).dropna()

        if len(subset) < 100:
            n_failed += 1
            continue

        if any(subset[d].std() < 1e-6 for d in sampled_drugs):
            n_failed += 1
            continue
        if any(subset[d].std() < 1e-6 for d in delta_cols):
            n_failed += 1
            continue

        for drug in sampled_drugs:
            for _, delta_col in state_delta_pairs:
                pair_included[(drug, delta_col)] += 1

        data = scaler.fit_transform(subset.values.astype(np.float64))
        g = run_fci_once(data, col_names, tier, alpha)
        if g is None:
            n_failed += 1
            continue

        for drug in sampled_drugs:
            for _, delta_col in state_delta_pairs:
                edge = extract_directed_edge(g, col_names, drug, delta_col)
                if edge == "definite":
                    pair_definite[(drug, delta_col)] += 1
                    pair_any[(drug, delta_col)] += 1
                elif edge == "possible":
                    pair_possible[(drug, delta_col)] += 1
                    pair_any[(drug, delta_col)] += 1

        if (run_idx + 1) % log_every == 0:
            elapsed = time.time() - t_start
            est_remaining = elapsed / (run_idx + 1) * (n_runs - run_idx - 1)
            log.info(
                "Run %4d / %d  |  elapsed %.0fs  |  est. remaining %.0fs  |  failed %d",
                run_idx + 1,
                n_runs,
                elapsed,
                est_remaining,
                n_failed,
            )
            live_counters = {
                "pair_included": dict(pair_included),
                "pair_definite": dict(pair_definite),
                "pair_possible": dict(pair_possible),
                "pair_any": dict(pair_any),
            }
            live_top = build_live_top_pairs(live_counters, state_entries, top_n=5, min_included=3)
            if live_top:
                log.info("Current top robust pairs (min n=3):")
                for row in live_top:
                    log.info(
                        "  %-22s %-26s freq=%.3f any=%.3f n=%d",
                        row["drug"],
                        row["state_delta"],
                        row["freq_definite"],
                        row["freq_any"],
                        row["n_runs_included"],
                    )

    counters = {
        "pair_included": dict(pair_included),
        "pair_definite": dict(pair_definite),
        "pair_possible": dict(pair_possible),
        "pair_any": dict(pair_any),
    }
    return counters, n_failed


def build_pair_results_df(
    counters: dict,
    drug_cols: list[str],
    state_entries: list[dict[str, str]],
) -> pd.DataFrame:
    rows = []
    for drug in drug_cols:
        for state_entry in state_entries:
            delta_col = state_entry["delta"]
            pair = (drug, delta_col)
            ni = counters["pair_included"].get(pair, 0)
            nd = counters["pair_definite"].get(pair, 0)
            np_ = counters["pair_possible"].get(pair, 0)
            na = counters["pair_any"].get(pair, 0)
            rows.append({
                "drug": drug,
                "state_raw": state_entry["raw"],
                "state_last": state_entry["last"],
                "baseline_state": state_entry["first"],
                "state_delta": delta_col,
                "n_runs_included": ni,
                "n_definite": nd,
                "n_possible": np_,
                "n_any": na,
                "freq_definite": round(nd / ni, 4) if ni > 0 else 0.0,
                "freq_any": round(na / ni, 4) if ni > 0 else 0.0,
            })

    df = pd.DataFrame(rows).sort_values(
        ["freq_definite", "freq_any", "drug", "state_delta"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def build_live_top_pairs(
    counters: dict,
    state_entries: list[dict[str, str]],
    top_n: int = 5,
    min_included: int = 3,
) -> list[dict[str, object]]:
    state_by_delta = {s["delta"]: s for s in state_entries}
    rows = []
    for (drug, delta_col), ni in counters["pair_included"].items():
        if ni < min_included:
            continue
        nd = counters["pair_definite"].get((drug, delta_col), 0)
        na = counters["pair_any"].get((drug, delta_col), 0)
        state_entry = state_by_delta.get(delta_col)
        rows.append({
            "drug": drug,
            "state_delta": delta_col,
            "state_raw": state_entry["raw"] if state_entry else delta_col,
            "n_runs_included": ni,
            "freq_definite": nd / ni if ni > 0 else 0.0,
            "freq_any": na / ni if ni > 0 else 0.0,
        })

    rows.sort(key=lambda r: (-r["freq_definite"], -r["freq_any"], -r["n_runs_included"], r["drug"], r["state_delta"]))
    return rows[:top_n]


def build_per_drug_rankings(pair_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for drug, sub in pair_df.groupby("drug", sort=False):
        sub = sub.sort_values(["freq_definite", "freq_any", "state_delta"], ascending=[False, False, True]).reset_index(drop=True)
        sub = sub.copy()
        sub["rank"] = np.arange(1, len(sub) + 1)
        cols = [
            "rank", "drug", "state_raw", "state_last", "baseline_state", "state_delta",
            "n_runs_included", "n_definite", "n_possible", "n_any", "freq_definite", "freq_any",
        ]
        out[drug] = sub[cols]
    return out


def main(args: argparse.Namespace) -> None:
    t0 = time.time()

    input_path = Path(args.input)
    report_dir = Path(args.report_dir)
    results_dir = report_dir / "action_state_results_robust"
    results_dir.mkdir(parents=True, exist_ok=True)

    state_entries = build_state_entries_from_raw(STATE_POOL_RAW)
    state_pool_raw = [s["raw"] for s in state_entries]
    log.info("Selected original Step 04 broad state pool (%d states): %s", len(state_entries), [s["last"] for s in state_entries])

    need_cols = (
        [C_ICUSTAYID, C_BLOC]
        + ["age", "charlson_score", "prior_ed_visits_6m"]
        + [raw for raw, _, _ in DRUGS]
        + state_pool_raw
    )
    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    use_cols = [c for c in need_cols if c in header]
    log.info("Loading %s (selected columns only) ...", input_path)
    df_raw = pd.read_csv(input_path, usecols=use_cols, low_memory=False)
    log.info("Loaded: %d rows, %d columns (of %d total)", len(df_raw), len(df_raw.columns), len(header))

    if args.smoke:
        keep_stays = df_raw[C_ICUSTAYID].unique()[:3000]
        df_raw = df_raw[df_raw[C_ICUSTAYID].isin(keep_stays)].copy()
        log.info("Smoke: restricted to %d stays (%d rows)", len(keep_stays), len(df_raw))

    log.info("Building stay-level robust action+state table...")
    stay_df = build_action_state_table(df_raw, state_pool_raw=state_pool_raw)
    del df_raw
    log.info("Stay-level table: %d stays, %d columns", len(stay_df), len(stay_df.columns))

    state_entries = [s for s in state_entries if s["first"] in stay_df.columns and s["delta"] in stay_df.columns]
    drug_cols = [frac for _, frac, _ in DRUGS if frac in stay_df.columns]
    if len(drug_cols) < args.n_drugs_per_run:
        raise ValueError(f"Only {len(drug_cols)} drug exposures available, need {args.n_drugs_per_run}")
    if len(state_entries) < args.n_states_per_run:
        raise ValueError(f"Only {len(state_entries)} target states available, need {args.n_states_per_run}")

    log.info("Drug exposures: %d   Target states: %d", len(drug_cols), len(state_entries))
    log.info(
        "Config: n_runs=%d  n_sample=%d  alpha=%.3f  n_drugs_per_run=%d  n_states_per_run=%d  seed=%d",
        args.n_runs,
        args.n_sample,
        args.alpha,
        args.n_drugs_per_run,
        args.n_states_per_run,
        args.seed,
    )

    counters, n_failed = run_multivariate_stability(
        stay_df=stay_df,
        drug_cols=drug_cols,
        state_entries=state_entries,
        n_runs=args.n_runs,
        n_sample=args.n_sample,
        alpha=args.alpha,
        n_drugs_per_run=args.n_drugs_per_run,
        n_states_per_run=args.n_states_per_run,
        seed=args.seed,
    )

    pair_df = build_pair_results_df(counters, drug_cols=drug_cols, state_entries=state_entries)
    pair_path = report_dir / "action_state_pair_results_robust.csv"
    pair_df.to_csv(pair_path, index=False)
    log.info("Pair results saved: %s", pair_path)

    per_drug = build_per_drug_rankings(pair_df)
    for drug, df in per_drug.items():
        out_path = results_dir / f"{drug}_results.csv"
        df.to_csv(out_path, index=False)

    delta_cols = [s["delta"] for s in state_entries]
    freq_matrix = (
        pair_df.pivot(index="drug", columns="state_delta", values="freq_definite")
        .reindex(index=drug_cols, columns=delta_cols)
    )
    matrix_path = report_dir / "action_state_frequency_matrix_robust.csv"
    freq_matrix.to_csv(matrix_path)
    log.info("Frequency matrix saved: %s", matrix_path)

    expected_pair_appearances = args.n_runs * args.n_drugs_per_run * args.n_states_per_run / max(len(drug_cols) * len(state_entries), 1)
    top_pairs = (
        pair_df[pair_df["freq_definite"] > 0]
        .sort_values(["freq_definite", "freq_any"], ascending=[False, False])
        .head(30)[["drug", "state_delta", "state_raw", "freq_definite", "n_runs_included"]]
        .to_dict("records")
    )

    pool_payload = {
        "state_pool_source": "original_step_04_broad_pool",
        "selected_states": state_entries,
    }
    pool_path = report_dir / "action_state_state_pool_robust.json"
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(pool_payload, f, indent=2)

    summary = {
        "n_stays": len(stay_df),
        "n_drugs": len(drug_cols),
        "n_states": len(state_entries),
        "n_runs": args.n_runs,
        "n_sample": args.n_sample,
        "alpha": args.alpha,
        "n_drugs_per_run": args.n_drugs_per_run,
        "n_states_per_run": args.n_states_per_run,
        "expected_pair_appearances": round(expected_pair_appearances, 1),
        "n_failed": n_failed,
        "runtime_s": round(time.time() - t0, 1),
        "top_pairs": top_pairs,
    }
    summary_path = report_dir / "action_state_summary_robust.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved: %s", summary_path)
    log.info("State pool manifest saved: %s", pool_path)

    log.info("")
    log.info("=" * 84)
    log.info("ROBUST ACTION -> STATE STABILITY  (n_runs=%d  alpha=%.2f)", args.n_runs, args.alpha)
    log.info("Top robust (drug, delta_state) pairs:")
    log.info("%-25s %-28s %12s %8s", "Drug", "State", "Freq_Definite", "N_runs")
    log.info("-" * 80)
    for row in top_pairs[:20]:
        log.info(
            "%-25s %-28s %12.3f %8d",
            row["drug"],
            row["state_delta"],
            row["freq_definite"],
            row["n_runs_included"],
        )
    log.info("=" * 84)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "ICUdataset.csv"),
        help="Path to ICUdataset.csv",
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "icu_readmit" / "step_09_state_action_selection"),
        help="Output directory (parallel robust outputs will be written here)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=4000,
        help="Number of multivariate FCI runs",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=5000,
        help="Stays subsampled per run",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FCI significance threshold",
    )
    parser.add_argument(
        "--n-drugs-per-run",
        type=int,
        default=2,
        help="Number of sampled drug exposures per run",
    )
    parser.add_argument(
        "--n-states-per-run",
        type=int,
        default=2,
        help="Number of sampled target states per run",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 25 runs, 500-sample subset",
    )
    args = parser.parse_args()

    if args.smoke:
        args.n_runs = 25
        args.n_sample = 500

    main(args)
