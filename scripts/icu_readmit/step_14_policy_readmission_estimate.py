"""Step 14 -- Fixed-horizon simulator-based readmission risk comparison.

This supplements the existing Step 14 OPE comparison with a more interpretable
policy outcome estimate:

  - load the four DDQN policies:
      * offline DDQN
      * CARE-Sim DDQN
      * MarkovSim DDQN
      * DAG-aware DDQN
  - start from held-out real seed histories
  - roll each policy forward for a fixed horizon in one or more simulators
  - score the final simulated state with the terminal readmission model

This is intentionally separate from the main step_14_offline_selected.py flow so
it can be run locally on CPU without retraining OPE support models.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.caresim.control.actions import ACTION_GRID, action_tensor, build_action_grid  # noqa: E402
from careai.icu_readmit.caresim.control.evaluation import load_seed_episodes  # noqa: E402
from careai.icu_readmit.caresim.control.observation import ObservationBuilder  # noqa: E402
from careai.icu_readmit.caresim.ensemble import CareSimEnsemble  # noqa: E402
from careai.icu_readmit.caresim.readmit import LightGBMReadmitModel  # noqa: E402
from careai.icu_readmit.caresim.simulator import CareSimEnvironment  # noqa: E402
from careai.icu_readmit.dagaware.ensemble import DAGAwareEnsemble  # noqa: E402
from careai.icu_readmit.dagaware.simulator import DAGAwareEnvironment  # noqa: E402
from careai.icu_readmit.rl.networks import DuelingDQN  # noqa: E402


DEFAULT_DATA = "data/processed/icu_readmit/rl_dataset_selected.parquet"
DEFAULT_REPORT_DIR = "reports/icu_readmit/offline_selected"
DEFAULT_TERMINAL_MODEL_DIR = "models/icu_readmit/terminal_readmit_selected"
DEFAULT_CARESIM_MODEL_DIR = "models/icu_readmit/caresim_selected_causal"
DEFAULT_DAGAWARE_MODEL_DIR = "models/icu_readmit/dagaware_selected_causal"
DEFAULT_OFFLINE_DDQN = "models/icu_readmit/offline_selected/ddqn/dqn_model.pt"
DEFAULT_CARESIM_DDQN = "models/icu_readmit/caresim_control_selected_causal/ddqn_model.pt"
DEFAULT_MARKOVSIM_DDQN = "models/icu_readmit/markovsim_control_selected_causal/ddqn_model.pt"
DEFAULT_DAGAWARE_DDQN = "models/icu_readmit/dagaware_control_selected_causal/ddqn_model.pt"

STATE_FEATURE_NAMES = [
    "Hb",
    "BUN",
    "Creatinine",
    "Phosphate",
    "HR",
    "Chloride",
    "age",
    "charlson_score",
    "prior_ed_visits_6m",
]
STATE_COLS = [f"s_{name}" for name in STATE_FEATURE_NAMES]
ACTION_COLS = ["vasopressor_b", "ivfluid_b", "antibiotic_b", "diuretic_b", "mechvent_b"]


@dataclass
class EvaluatorSpec:
    name: str
    ensemble: object
    env_cls: type


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 14: fixed-horizon policy readmission estimate")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--terminal-model-dir", default=DEFAULT_TERMINAL_MODEL_DIR)
    parser.add_argument("--caresim-model-dir", default=DEFAULT_CARESIM_MODEL_DIR)
    parser.add_argument("--dagaware-model-dir", default=DEFAULT_DAGAWARE_MODEL_DIR)
    parser.add_argument("--offline-ddqn-path", default=DEFAULT_OFFLINE_DDQN)
    parser.add_argument("--caresim-ddqn-path", default=DEFAULT_CARESIM_DDQN)
    parser.add_argument("--markovsim-ddqn-path", default=DEFAULT_MARKOVSIM_DDQN)
    parser.add_argument("--dagaware-ddqn-path", default=DEFAULT_DAGAWARE_DDQN)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--history-len", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--evaluators", nargs="+", choices=["caresim", "dagaware"], default=["caresim", "dagaware"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", default="logs/step_14_policy_readmission_estimate.log")
    return parser


def setup_logging(log_path: str) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(path, mode="w", encoding="utf-8"), logging.StreamHandler()],
    )


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def load_ddqn(model_path: str, device: torch.device, obs_dim: int = 70, n_actions: int = 32) -> DuelingDQN:
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        obs_dim = int(ckpt.get("obs_dim", obs_dim))
        n_actions = int(ckpt.get("n_actions", n_actions))
    else:
        state_dict = ckpt
    model = DuelingDQN(n_input=obs_dim, n_actions=n_actions, hidden=128).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def greedy_action_id(model: DuelingDQN, obs: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model(obs_t)
    return int(q_values.argmax(dim=1).item())


def load_evaluators(args, device: torch.device) -> list[EvaluatorSpec]:
    specs: list[EvaluatorSpec] = []
    if "caresim" in args.evaluators:
        specs.append(
            EvaluatorSpec(
                name="caresim",
                ensemble=CareSimEnsemble.from_dir(args.caresim_model_dir, device=device),
                env_cls=CareSimEnvironment,
            )
        )
    if "dagaware" in args.evaluators:
        specs.append(
            EvaluatorSpec(
                name="dagaware",
                ensemble=DAGAwareEnsemble.from_dir(args.dagaware_model_dir, device=device),
                env_cls=DAGAwareEnvironment,
            )
        )
    return specs


def policy_models(args, device: torch.device) -> dict[str, DuelingDQN]:
    return {
        "offline_ddqn": load_ddqn(args.offline_ddqn_path, device=device),
        "caresim_ddqn": load_ddqn(args.caresim_ddqn_path, device=device),
        "markovsim_ddqn": load_ddqn(args.markovsim_ddqn_path, device=device),
        "dagaware_ddqn": load_ddqn(args.dagaware_ddqn_path, device=device),
    }


def evaluate_policy_in_env(
    env_spec: EvaluatorSpec,
    episodes,
    policy_name: str,
    model: DuelingDQN,
    terminal_model: LightGBMReadmitModel,
    horizon: int,
    device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    rows: list[dict] = []
    action_dim = int(episodes[0].seed_actions.shape[1])
    action_grid = ACTION_GRID if action_dim == ACTION_GRID.shape[1] else build_action_grid(action_dim)

    for episode in episodes:
        seed_states_t = torch.tensor(episode.seed_states, dtype=torch.float32, device=device).unsqueeze(0)
        seed_actions_t = torch.tensor(episode.seed_actions, dtype=torch.float32, device=device).unsqueeze(0)
        seed_time_steps_t = torch.tensor(episode.seed_time_steps, dtype=torch.float32, device=device).unsqueeze(0)

        env = env_spec.env_cls(
            env_spec.ensemble,
            max_steps=horizon,
            uncertainty_threshold=1.0,
            device=device,
            severity_model=None,
            terminal_outcome_model=None,
            terminal_reward_scale=15.0,
        )
        env.reset(seed_states_t, seed_actions_t, seed_time_steps=seed_time_steps_t)
        obs_builder = ObservationBuilder(
            window_len=5,
            state_dim=episode.seed_states.shape[1],
            action_dim=episode.seed_actions.shape[1],
        )
        obs = obs_builder.reset(episode.seed_states, episode.seed_actions)

        early_done = False
        final_terminal_prob = float("nan")
        final_uncertainty = float("nan")
        action_trace: list[int] = []

        for step_idx in range(horizon):
            action_id = greedy_action_id(model, obs, device=device)
            action = action_tensor(action_id, device=device, action_dim=action_dim)
            next_state, _reward, done_tensor, info = env.step(action)
            obs = obs_builder.append(
                next_state[0].detach().cpu().numpy(),
                np.asarray(action_grid[action_id], dtype=np.float32),
            )
            action_trace.append(action_id)
            final_terminal_prob = float(info["terminal_prob"][0])
            final_uncertainty = float(info["uncertainty"][0])
            if bool(done_tensor[0].item()):
                early_done = True

        final_state = env.current_state()
        final_p_readmit = float(terminal_model.predict_proba(final_state)[0].item())
        rows.append(
            {
                "evaluator": env_spec.name,
                "policy": policy_name,
                "stay_id": int(episode.stay_id),
                "split": episode.split,
                "horizon": horizon,
                "predicted_readmit": final_p_readmit,
                "final_terminal_prob": final_terminal_prob,
                "final_uncertainty": final_uncertainty,
                "early_done": early_done,
                "action_trace": ",".join(str(a) for a in action_trace),
            }
        )

    df = pd.DataFrame(rows)
    result = {
        "episodes": int(len(df)),
        "mean_predicted_readmit": float(df["predicted_readmit"].mean()),
        "std_predicted_readmit": float(df["predicted_readmit"].std(ddof=0)),
        "mean_predicted_readmit_pct": float(100.0 * df["predicted_readmit"].mean()),
        "mean_final_terminal_prob": float(df["final_terminal_prob"].mean()),
        "mean_final_uncertainty": float(df["final_uncertainty"].mean()),
        "early_done_rate": float(df["early_done"].mean()),
    }
    return result, df


def main() -> None:
    args = build_parser().parse_args()
    args.data = resolve_repo_path(args.data)
    args.report_dir = resolve_repo_path(args.report_dir)
    args.terminal_model_dir = resolve_repo_path(args.terminal_model_dir)
    args.caresim_model_dir = resolve_repo_path(args.caresim_model_dir)
    args.dagaware_model_dir = resolve_repo_path(args.dagaware_model_dir)
    args.offline_ddqn_path = resolve_repo_path(args.offline_ddqn_path)
    args.caresim_ddqn_path = resolve_repo_path(args.caresim_ddqn_path)
    args.markovsim_ddqn_path = resolve_repo_path(args.markovsim_ddqn_path)
    args.dagaware_ddqn_path = resolve_repo_path(args.dagaware_ddqn_path)
    args.log = resolve_repo_path(args.log)

    setup_logging(args.log)
    device = resolve_device(args.device)
    t0 = time.time()
    logging.info("Step 14 fixed-horizon readmission estimate started. device=%s", device)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_seed_episodes(
        data_path=args.data,
        split=args.split,
        history_len=args.history_len,
        max_episodes=args.max_episodes,
        seed=args.seed,
    )
    if not episodes:
        raise ValueError("No eligible seed episodes found")
    logging.info("Loaded %d seed episodes from split=%s", len(episodes), args.split)

    terminal_model = LightGBMReadmitModel.from_dir(
        args.terminal_model_dir,
        state_feature_names=STATE_COLS,
        device=device,
    )
    policies = policy_models(args, device)
    evaluators = load_evaluators(args, device)

    all_rows: list[pd.DataFrame] = []
    summary = {
        "meta": {
            "data": args.data,
            "split": args.split,
            "history_len": args.history_len,
            "horizon": args.horizon,
            "max_episodes": args.max_episodes,
            "evaluators": args.evaluators,
            "terminal_model_dir": args.terminal_model_dir,
            "device": str(device),
        }
    }

    for evaluator in evaluators:
        summary[evaluator.name] = {}
        logging.info("Evaluating in %s simulator", evaluator.name)
        for policy_name, model in policies.items():
            result, detail_df = evaluate_policy_in_env(
                evaluator,
                episodes,
                policy_name,
                model,
                terminal_model,
                horizon=args.horizon,
                device=device,
            )
            summary[evaluator.name][policy_name] = result
            all_rows.append(detail_df)
            logging.info(
                "  %s | mean readmit=%.3f (%.1f%%) | early_done=%.2f",
                policy_name,
                result["mean_predicted_readmit"],
                result["mean_predicted_readmit_pct"],
                result["early_done_rate"],
            )

    summary["meta"]["total_seconds"] = round(time.time() - t0, 1)
    summary_path = report_dir / "step_14_policy_readmission_estimate.json"
    detail_path = report_dir / "step_14_policy_readmission_detail.csv"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.concat(all_rows, ignore_index=True).to_csv(detail_path, index=False)

    logging.info("Saved summary: %s", summary_path)
    logging.info("Saved detail:  %s", detail_path)
    logging.info("Step 14 fixed-horizon readmission estimate complete in %.1f sec", time.time() - t0)


if __name__ == "__main__":
    main()
