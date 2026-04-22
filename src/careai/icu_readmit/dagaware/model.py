"""
DAG-aware temporal world model for ICU trajectories.

Core idea:
  - each scalar variable at each time step is a token
  - static confounders are global tokens shared across the whole sequence
  - a fixed temporal DAG mask decides which tokens may attend to which others

Selected-track design:
  - 6 dynamic states
  - 3 static confounders
  - 5 actions
  - predicts next dynamic state values and terminal probability
  - static confounders are copied through unchanged
"""
from __future__ import annotations

import torch
import torch.nn as nn


SELECTED_DYNAMIC_ACTION_MASK = torch.tensor(
    [
        [1, 1, 1, 0, 1],  # Hb
        [0, 1, 0, 1, 0],  # BUN
        [0, 1, 0, 1, 0],  # Creatinine
        [0, 1, 1, 1, 0],  # Phosphate
        [1, 0, 0, 0, 1],  # HR
        [0, 1, 0, 1, 0],  # Chloride
    ],
    dtype=torch.bool,
)

TIER2_DYNAMIC_ACTION_MASK = torch.tensor(
    [
        [0, 0, 0, 1],  # Hb
        [0, 0, 0, 1],  # BUN
        [0, 1, 0, 0],  # Creatinine
        [1, 0, 1, 0],  # HR
        [1, 0, 0, 0],  # Shock_Index
    ],
    dtype=torch.bool,
)


def default_state_partitions(state_dim: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return built-in dynamic/static splits for known ICU tracks."""
    if state_dim == 9:
        return tuple(range(6)), (6, 7, 8)
    if state_dim == 8:
        return tuple(range(5)), (5, 6, 7)
    raise ValueError(
        f"No built-in dynamic/static partition for state_dim={state_dim}. "
        "Pass dynamic_state_idx and static_state_idx explicitly."
    )


def default_action_causal_mask(n_dynamic: int, action_dim: int) -> torch.Tensor | None:
    if (n_dynamic, action_dim) == tuple(SELECTED_DYNAMIC_ACTION_MASK.shape):
        return SELECTED_DYNAMIC_ACTION_MASK.clone()
    if (n_dynamic, action_dim) == tuple(TIER2_DYNAMIC_ACTION_MASK.shape):
        return TIER2_DYNAMIC_ACTION_MASK.clone()
    return None


class DAGAwareTemporalWorldModel(nn.Module):
    """Node-time DAG-aware transformer world model."""

    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 5,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 80,
        dynamic_state_idx: tuple[int, ...] | None = None,
        static_state_idx: tuple[int, ...] | None = None,
        action_causal_mask: torch.Tensor | None = None,
        action_mask_placement: str = "final_only",
        use_time_feature: bool = True,
        predict_reward: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        if dynamic_state_idx is None or static_state_idx is None:
            built_dynamic, built_static = default_state_partitions(state_dim)
            dynamic_state_idx = dynamic_state_idx or built_dynamic
            static_state_idx = static_state_idx or built_static

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_time_feature = use_time_feature
        self.predict_reward = bool(predict_reward)
        self.action_mask_placement = action_mask_placement
        self.dynamic_state_idx = tuple(dynamic_state_idx)
        self.static_state_idx = tuple(static_state_idx)

        if self.action_mask_placement not in {"final_only", "first_only", "all_layers"}:
            raise ValueError(
                "action_mask_placement must be one of "
                "{'final_only', 'first_only', 'all_layers'}"
            )

        if set(self.dynamic_state_idx).intersection(self.static_state_idx):
            raise ValueError("dynamic_state_idx and static_state_idx must be disjoint")
        if len(self.dynamic_state_idx) + len(self.static_state_idx) != state_dim:
            raise ValueError("dynamic_state_idx and static_state_idx must cover all state dimensions")

        self.n_dynamic = len(self.dynamic_state_idx)
        self.n_static = len(self.static_state_idx)
        self.step_width = self.n_dynamic + self.action_dim
        self.n_node_ids = self.state_dim + self.action_dim

        if action_causal_mask is None:
            action_causal_mask = default_action_causal_mask(self.n_dynamic, action_dim)
        if action_causal_mask is None:
            raise ValueError(
                f"No built-in action mask for (n_dynamic={self.n_dynamic}, action_dim={action_dim}). "
                "Pass action_causal_mask explicitly."
            )
        if tuple(action_causal_mask.shape) != (self.n_dynamic, self.action_dim):
            raise ValueError(
                "action_causal_mask must have shape "
                f"({self.n_dynamic}, {self.action_dim}), got {tuple(action_causal_mask.shape)}"
            )
        self.register_buffer("action_causal_mask", action_causal_mask.bool())

        self.value_proj = nn.Linear(1, d_model)
        self.node_embed = nn.Embedding(self.n_node_ids, d_model)
        self.type_embed = nn.Embedding(3, d_model)  # static, state, action
        if self.use_time_feature:
            self.time_proj = nn.Linear(1, d_model)
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=4 * d_model,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.state_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(self.n_dynamic)])
        self.head_terminal = nn.Linear(d_model, 1)

        self._layout_cache: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
        self._mask_cache: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _layout_for_length(self, seq_len: int, device: torch.device) -> dict[str, torch.Tensor]:
        key = (seq_len, str(device))
        cached = self._layout_cache.get(key)
        if cached is not None:
            return cached

        token_count = self.n_static + seq_len * self.step_width
        node_ids = torch.empty(token_count, dtype=torch.long, device=device)
        type_ids = torch.empty(token_count, dtype=torch.long, device=device)
        time_ids = torch.zeros(token_count, dtype=torch.float32, device=device)
        state_indices = torch.empty(seq_len, self.n_dynamic, dtype=torch.long, device=device)
        action_indices = torch.empty(seq_len, self.action_dim, dtype=torch.long, device=device)

        if self.n_static:
            static_state_ids = torch.tensor(self.static_state_idx, dtype=torch.long, device=device)
            node_ids[: self.n_static] = static_state_ids
            type_ids[: self.n_static] = 0

        dynamic_ids = torch.tensor(self.dynamic_state_idx, dtype=torch.long, device=device)
        for t in range(seq_len):
            base = self.n_static + t * self.step_width
            state_pos = torch.arange(base, base + self.n_dynamic, device=device)
            action_pos = torch.arange(base + self.n_dynamic, base + self.step_width, device=device)
            state_indices[t] = state_pos
            action_indices[t] = action_pos

            node_ids[state_pos] = dynamic_ids
            type_ids[state_pos] = 1
            time_ids[state_pos] = float(t)

            node_ids[action_pos] = torch.arange(self.state_dim, self.state_dim + self.action_dim, device=device)
            type_ids[action_pos] = 2
            time_ids[action_pos] = float(t)

        layout = {
            "node_ids": node_ids,
            "type_ids": type_ids,
            "time_ids": time_ids,
            "state_indices": state_indices,
            "action_indices": action_indices,
        }
        self._layout_cache[key] = layout
        return layout

    def _attention_masks(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (seq_len, str(device))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        layout = self._layout_for_length(seq_len, device)
        token_count = self.n_static + seq_len * self.step_width
        history_mask = torch.ones(token_count, token_count, dtype=torch.bool, device=device)
        action_mask = torch.ones(token_count, token_count, dtype=torch.bool, device=device)

        static_range = torch.arange(self.n_static, device=device)
        if self.n_static:
            history_mask[static_range[:, None], static_range[None, :]] = False
            action_mask[static_range[:, None], static_range[None, :]] = False

        for t in range(seq_len):
            state_pos = layout["state_indices"][t]
            action_pos = layout["action_indices"][t]

            if self.n_static:
                history_mask[state_pos[:, None], static_range[None, :]] = False
                action_mask[state_pos[:, None], static_range[None, :]] = False
                history_mask[action_pos[:, None], static_range[None, :]] = False
                action_mask[action_pos[:, None], static_range[None, :]] = False

            for tp in range(t + 1):
                past_state_pos = layout["state_indices"][tp]
                history_mask[state_pos[:, None], past_state_pos[None, :]] = False
                action_mask[state_pos[:, None], past_state_pos[None, :]] = False
                history_mask[action_pos[:, None], past_state_pos[None, :]] = False
                action_mask[action_pos[:, None], past_state_pos[None, :]] = False

            history_mask[action_pos, action_pos] = False
            action_mask[action_pos, action_pos] = False

            for dyn_i in range(self.n_dynamic):
                allowed_actions = torch.nonzero(self.action_causal_mask[dyn_i], as_tuple=False).flatten()
                if len(allowed_actions):
                    action_mask[state_pos[dyn_i], action_pos[allowed_actions]] = False

        self._mask_cache[key] = (history_mask, action_mask)
        return history_mask, action_mask

    def _expand_padding_mask(self, src_key_padding_mask: torch.Tensor | None) -> torch.Tensor | None:
        if src_key_padding_mask is None:
            return None
        if src_key_padding_mask.ndim != 2:
            raise ValueError("src_key_padding_mask must have shape (B, T)")
        batch_size = src_key_padding_mask.shape[0]
        static_pad = torch.zeros(batch_size, self.n_static, dtype=torch.bool, device=src_key_padding_mask.device)
        temporal_pad = src_key_padding_mask.repeat_interleave(self.step_width, dim=1)
        return torch.cat([static_pad, temporal_pad], dim=1)

    def _mask_for_layer(self, layer_index: int, n_layers: int, history_mask: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        if self.action_mask_placement == "all_layers":
            return action_mask
        if self.action_mask_placement == "first_only":
            return action_mask if layer_index == 0 else history_mask
        return action_mask if layer_index == n_layers - 1 else history_mask

    def embed(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        time_steps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, seq_len, _ = states.shape
        layout = self._layout_for_length(seq_len, states.device)

        if self.n_static:
            static_values = states[:, 0, list(self.static_state_idx)]
        else:
            static_values = states.new_zeros(batch_size, 0)
        dynamic_values = states[:, :, list(self.dynamic_state_idx)]
        step_values = torch.cat([dynamic_values, actions], dim=-1).reshape(batch_size, -1)
        token_values = torch.cat([static_values, step_values], dim=1)

        x = self.value_proj(token_values.unsqueeze(-1))
        x = x + self.node_embed(layout["node_ids"]).unsqueeze(0)
        x = x + self.type_embed(layout["type_ids"]).unsqueeze(0)

        if self.use_time_feature:
            if time_steps is None:
                time_steps = torch.arange(seq_len, device=states.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
            time_tokens = time_steps.unsqueeze(-1).repeat(1, 1, self.step_width).reshape(batch_size, -1)
            if self.n_static:
                static_time = torch.zeros(batch_size, self.n_static, device=states.device, dtype=torch.float32)
                time_tokens = torch.cat([static_time, time_tokens], dim=1)
            x = x + self.time_proj(torch.log1p(time_tokens).unsqueeze(-1))

        x = self.embed_drop(self.embed_norm(x))
        return x, layout

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        time_steps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _, seq_len, _ = states.shape
        x, layout = self.embed(states, actions, time_steps=time_steps)
        token_padding_mask = self._expand_padding_mask(src_key_padding_mask)
        history_mask, action_mask = self._attention_masks(seq_len, states.device)

        h = x
        for layer_index, layer in enumerate(self.layers):
            layer_mask = self._mask_for_layer(layer_index, len(self.layers), history_mask, action_mask)
            h = layer(h, src_mask=layer_mask, src_key_padding_mask=token_padding_mask)
        h = self.final_norm(h)

        state_hidden = h[:, layout["state_indices"].reshape(-1), :].view(
            states.shape[0], seq_len, self.n_dynamic, self.d_model
        )
        dynamic_preds = [head(state_hidden[:, :, i, :]).squeeze(-1) for i, head in enumerate(self.state_heads)]
        dynamic_next = torch.stack(dynamic_preds, dim=-1)

        next_state = states.clone()
        next_state[..., list(self.dynamic_state_idx)] = dynamic_next
        if self.n_static:
            next_state[..., list(self.static_state_idx)] = states[..., list(self.static_state_idx)]

        terminal_context = state_hidden.mean(dim=2)
        terminal = self.head_terminal(terminal_context).squeeze(-1)

        state_loss_mask = torch.zeros(self.state_dim, device=states.device, dtype=states.dtype)
        state_loss_mask[list(self.dynamic_state_idx)] = 1.0
        return {
            "next_state": next_state,
            "reward": None,
            "terminal": terminal,
            "state_loss_mask": state_loss_mask,
        }

    def predict_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        time_steps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            out = self.forward(states, actions, time_steps=time_steps)
        return {
            "next_state": out["next_state"][:, -1, :],
            "reward": None,
            "terminal": torch.sigmoid(out["terminal"][:, -1]),
        }

    @torch.no_grad()
    def last_layer_attention(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        time_steps: torch.Tensor | None = None,
        average_heads: bool = False,
    ) -> dict[str, torch.Tensor]:
        _, seq_len, _ = states.shape
        x, layout = self.embed(states, actions, time_steps=time_steps)
        token_padding_mask = self._expand_padding_mask(src_key_padding_mask)
        history_mask, action_mask = self._attention_masks(seq_len, states.device)

        h = x
        for layer_index, layer in enumerate(self.layers[:-1]):
            layer_mask = self._mask_for_layer(layer_index, len(self.layers), history_mask, action_mask)
            h = layer(h, src_mask=layer_mask, src_key_padding_mask=token_padding_mask)

        last_index = len(self.layers) - 1
        last_layer = self.layers[last_index]
        last_mask = self._mask_for_layer(last_index, len(self.layers), history_mask, action_mask)

        attn_input = last_layer.norm1(h) if last_layer.norm_first else h
        _, attn_weights = last_layer.self_attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=last_mask,
            key_padding_mask=token_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        if average_heads:
            attn_weights = attn_weights.mean(dim=1)

        return {
            "attention": attn_weights,
            "history_mask": history_mask,
            "action_mask": action_mask,
            "layout": layout,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
