"""
Step 12c -- DAG-aware last-layer attention visualization for adjacent history and same-time actions.

This is a post-hoc analysis utility:
  - loads an already-trained DAG-aware model member
  - selects one short held-out patient sequence
  - extracts last-layer attention weights
  - keeps only:
      * current dynamic states at time t
      * previous dynamic states at time t-1
      * current-time actions at time t
  - saves a compact heatmap and raw attention artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.caresim.dataset import ICUSequenceDataset, infer_schema_from_path
from careai.icu_readmit.dagaware.train import EnsembleTrainer


STATIC_STATE_NAMES = ["age", "charlson_score", "prior_ed_visits_6m"]


def parse_args():
    parser = argparse.ArgumentParser(description="Step 12c: visualize DAG-aware last-layer attention")
    parser.add_argument("--data", default="data/processed/icu_readmit/rl_dataset_selected.parquet")
    parser.add_argument("--model-dir", default="models/icu_readmit/dagaware_selected_causal")
    parser.add_argument("--report-dir", default="reports/icu_readmit/dagaware_selected_causal")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--member-index", type=int, default=0)
    parser.add_argument("--sequence-index", type=int, default=0, help="Index among eligible sequences in the chosen split.")
    parser.add_argument("--seq-len", type=int, default=5, help="Trim selected patient window to this many final steps.")
    parser.add_argument("--head", type=int, default=None, help="Attention head to plot. Omit to average over heads.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dynamic_state_names(state_names: list[str]) -> list[str]:
    return [name for name in state_names if name not in STATIC_STATE_NAMES]


def choose_sequence(dataset: ICUSequenceDataset, seq_len: int, sequence_index: int) -> tuple[dict, int]:
    eligible = [(i, seq) for i, seq in enumerate(dataset.sequences) if int(seq["full_length"]) >= seq_len]
    if not eligible:
        raise ValueError(f"No sequences in split have length >= {seq_len}")
    if sequence_index < 0 or sequence_index >= len(eligible):
        raise IndexError(f"sequence-index must be in [0, {len(eligible) - 1}]")
    return eligible[sequence_index]


def trim_sequence(seq: dict, seq_len: int) -> dict[str, np.ndarray]:
    start = max(int(seq["full_length"]) - seq_len, 0)
    stop = int(seq["full_length"])
    return {
        "states": seq["states"][start:stop],
        "actions": seq["actions"][start:stop],
        "time_steps": seq["time_steps"][start:stop],
    }


def focused_attention_matrix(
    model,
    attention: np.ndarray,
    seq_len: int,
    state_names: list[str],
    action_names: list[str],
) -> tuple[np.ndarray, list[str], list[str], int, int]:
    dynamic_names = dynamic_state_names(state_names)
    query_t = seq_len - 1
    key_t = seq_len - 2
    query_idx = model.n_static + query_t * model.step_width + np.arange(model.n_dynamic)
    prev_state_idx = model.n_static + key_t * model.step_width + np.arange(model.n_dynamic)
    current_action_idx = model.n_static + query_t * model.step_width + model.n_dynamic + np.arange(model.action_dim)
    key_idx = np.concatenate([prev_state_idx, current_action_idx])
    matrix = attention[np.ix_(query_idx, key_idx)]
    row_labels = [f"{name}_t{query_t}" for name in dynamic_names]
    col_labels = [f"{name}_t{key_t}" for name in dynamic_names] + [f"{name}_t{query_t}" for name in action_names]
    return matrix, row_labels, col_labels, key_t, query_t


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    state_cols, action_cols, _ = infer_schema_from_path(args.data)
    state_names = [c.removeprefix("s_") for c in state_cols]
    action_names = list(action_cols)

    dataset = ICUSequenceDataset.from_parquet(
        args.data,
        split=args.split,
        max_seq_len=80,
        window_mode="last",
        state_cols=state_cols,
        action_cols=action_cols,
    )
    original_index, seq = choose_sequence(dataset, args.seq_len, args.sequence_index)
    trimmed = trim_sequence(seq, args.seq_len)

    trainer = EnsembleTrainer.load(args.model_dir, device=device)
    if args.member_index < 0 or args.member_index >= len(trainer.models):
        raise IndexError(f"member-index must be in [0, {len(trainer.models) - 1}]")
    model = trainer.models[args.member_index].to(device).eval()

    states = torch.tensor(trimmed["states"], dtype=torch.float32, device=device).unsqueeze(0)
    actions = torch.tensor(trimmed["actions"], dtype=torch.float32, device=device).unsqueeze(0)
    time_steps = torch.tensor(trimmed["time_steps"], dtype=torch.float32, device=device).unsqueeze(0)

    attn_out = model.last_layer_attention(states, actions, time_steps=time_steps, average_heads=False)
    attention = attn_out["attention"][0].detach().cpu().numpy()  # (H, N, N)

    if args.head is None:
        matrix = attention.mean(axis=0)
        head_label = "avg"
    else:
        if args.head < 0 or args.head >= attention.shape[0]:
            raise IndexError(f"head must be in [0, {attention.shape[0] - 1}]")
        matrix = attention[args.head]
        head_label = f"head{args.head}"

    matrix, row_labels, col_labels, key_t, query_t = focused_attention_matrix(
        model,
        matrix,
        args.seq_len,
        state_names,
        action_names,
    )

    base_name = f"dagaware_focused_attention_{args.split}_seq{args.sequence_index}_member{args.member_index}_{head_label}"
    png_path = report_dir / f"{base_name}.png"
    npy_path = report_dir / f"{base_name}.npy"
    meta_path = report_dir / f"{base_name}.json"

    fig_w = max(7, len(col_labels) * 0.9)
    fig_h = max(5, len(row_labels) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=args.dpi)
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title(f"DAG-aware focused last-layer attention ({head_label})")
    ax.set_xlabel(f"Previous-step states at t={key_t} and current-time actions at t={query_t}")
    ax.set_ylabel(f"Current-step dynamic states at t={query_t}")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    np.save(npy_path, matrix)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": args.split,
                "member_index": args.member_index,
                "sequence_index": args.sequence_index,
                "original_dataset_index": int(original_index),
                "seq_len": args.seq_len,
                "head": args.head,
                "row_labels": row_labels,
                "col_labels": col_labels,
                "time_steps": [float(x) for x in trimmed["time_steps"]],
                "action_mask_placement": getattr(model, "action_mask_placement", None),
                "focus": "dynamic_state_t_to_prev_state_t-1_and_current_action_t",
                "query_time_index": int(query_t),
                "key_time_index": int(key_t),
            },
            f,
            indent=2,
        )

    print(f"Saved heatmap: {png_path}")
    print(f"Saved matrix:  {npy_path}")
    print(f"Saved meta:    {meta_path}")


if __name__ == "__main__":
    main()
