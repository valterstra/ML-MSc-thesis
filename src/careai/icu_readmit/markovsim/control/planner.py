"""Short-horizon planner over the causal Markov simulator."""

from __future__ import annotations

from dataclasses import dataclass

from ..model import MarkovSimEnsemble
from ..simulator import MarkovSimEnvironment
from .actions import ACTION_GRID, action_tensor, build_action_grid


@dataclass
class PlannerConfig:
    horizon: int = 3
    gamma: float = 0.99
    uncertainty_penalty: float = 0.25
    max_steps: int = 5
    uncertainty_threshold: float = 1.0


def _score_transition(reward: float, uncertainty: float, penalty: float) -> float:
    return float(reward - penalty * uncertainty)


def planner_action(
    ensemble: MarkovSimEnsemble,
    seed_states,
    seed_actions,
    seed_time_steps,
    config: PlannerConfig,
    device,
    action_dim: int | None = None,
) -> dict:
    candidates: list[dict] = []
    action_dim = int(action_dim or seed_actions.shape[-1])
    action_grid = ACTION_GRID if action_dim == ACTION_GRID.shape[1] else build_action_grid(action_dim)

    for action_id, _ in enumerate(action_grid):
        env = MarkovSimEnvironment(
            ensemble,
            max_steps=max(config.max_steps, config.horizon),
            uncertainty_threshold=config.uncertainty_threshold,
            device=device,
        )
        env.reset(seed_states, seed_actions, seed_time_steps=seed_time_steps)

        cumulative = 0.0
        first_reward = None
        first_uncertainty = None
        first_terminal_prob = None
        first_done = None
        next_state_snapshot = None

        for step_idx in range(config.horizon):
            action = action_tensor(action_id, device=device, action_dim=action_dim)
            next_state, reward, done_tensor, info = env.step(action)
            reward_value = float(reward[0].item())
            uncertainty_value = float(info["uncertainty"][0])
            terminal_prob = float(info["terminal_prob"][0])
            done = bool(done_tensor[0].item())

            cumulative += (config.gamma ** step_idx) * _score_transition(
                reward_value,
                uncertainty_value,
                config.uncertainty_penalty,
            )

            if step_idx == 0:
                first_reward = reward_value
                first_uncertainty = uncertainty_value
                first_terminal_prob = terminal_prob
                first_done = done
                next_state_snapshot = next_state[0].detach().cpu().numpy()

            if done:
                break

        candidates.append({
            "action_id": int(action_id),
            "score": float(cumulative),
            "reward_pred": float(first_reward),
            "uncertainty": float(first_uncertainty),
            "terminal_prob": float(first_terminal_prob),
            "done_pred": bool(first_done),
            "next_state": next_state_snapshot,
        })

    candidates.sort(key=lambda row: (-row["score"], row["uncertainty"], row["action_id"]))
    best = candidates[0]
    return {
        "best_action_id": int(best["action_id"]),
        "best_score": float(best["score"]),
        "next_state": best["next_state"],
        "reward_pred": float(best["reward_pred"]),
        "uncertainty": float(best["uncertainty"]),
        "terminal_prob": float(best["terminal_prob"]),
        "done_pred": bool(best["done_pred"]),
        "candidates": candidates,
    }
