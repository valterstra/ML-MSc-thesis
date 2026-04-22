from __future__ import annotations

import torch
import torch.nn as nn


class NonCausalCareSimTransformer(nn.Module):
    """Broad CARE-Sim-style transformer with categorical static embeddings."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        dynamic_state_dim: int,
        numeric_state_idx: list[int],
        categorical_state_idx: list[int],
        categorical_cardinalities: list[int],
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 10,
        use_time_feature: bool = True,
        categorical_embed_dim: int = 8,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.dynamic_state_dim = int(dynamic_state_dim)
        self.max_seq_len = int(max_seq_len)
        self.use_time_feature = bool(use_time_feature)
        self.numeric_state_idx = list(numeric_state_idx)
        self.categorical_state_idx = list(categorical_state_idx)

        numeric_dim = len(self.numeric_state_idx)
        categorical_total_dim = len(categorical_cardinalities) * int(categorical_embed_dim)

        self.numeric_state_proj = nn.Linear(numeric_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(int(cardinality) + 1, int(categorical_embed_dim)) for cardinality in categorical_cardinalities]
        )
        self.cat_proj = nn.Linear(categorical_total_dim, d_model) if categorical_total_dim > 0 else None
        self.time_proj = nn.Linear(1, d_model) if self.use_time_feature else None
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        self.head_next_state = nn.Linear(d_model, self.dynamic_state_dim)
        self.head_terminal = nn.Linear(d_model, 1)
        self.head_readmit = nn.Linear(d_model, 1)

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

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def _embed_categorical_static(self, states: torch.Tensor) -> torch.Tensor | None:
        if not self.categorical_state_idx:
            return None
        first_step = states[:, 0, self.categorical_state_idx].round().long()
        parts = []
        for emb_idx, emb in enumerate(self.cat_embeddings):
            codes = first_step[:, emb_idx] + 1
            codes = codes.clamp(min=0, max=emb.num_embeddings - 1)
            parts.append(emb(codes))
        return torch.cat(parts, dim=-1)

    def embed(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        time_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = states.shape
        pos = torch.arange(seq_len, device=states.device)

        numeric_state = states[..., self.numeric_state_idx]
        x = self.numeric_state_proj(numeric_state) + self.action_proj(actions) + self.pos_embed(pos)

        cat_embed = self._embed_categorical_static(states)
        if cat_embed is not None and self.cat_proj is not None:
            x = x + self.cat_proj(cat_embed).unsqueeze(1)

        if self.use_time_feature and self.time_proj is not None:
            if time_steps is None:
                time_steps = pos.unsqueeze(0).expand(batch_size, seq_len).to(states.device)
            x = x + self.time_proj(torch.log1p(time_steps.float()).unsqueeze(-1))

        return self.embed_drop(self.embed_norm(x))

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        time_steps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _, seq_len, _ = states.shape
        x = self.embed(states, actions, time_steps=time_steps)
        causal_mask = self._make_causal_mask(seq_len, states.device)
        h = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,
        )

        if src_key_padding_mask is None:
            last_idx = torch.full((states.shape[0],), seq_len - 1, device=states.device, dtype=torch.long)
        else:
            last_idx = (~src_key_padding_mask).sum(dim=1) - 1
            last_idx = last_idx.clamp(min=0)
        final_hidden = h[torch.arange(h.shape[0], device=h.device), last_idx]

        return {
            "next_state": self.head_next_state(h),
            "terminal": self.head_terminal(h).squeeze(-1),
            "readmit": self.head_readmit(final_hidden).squeeze(-1),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
