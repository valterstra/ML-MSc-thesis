"""
CareSimGPT: causal GPT-2-style transformer world model.

Conceptual foundation: adapts the Trajectory Transformer (Janner et al., NeurIPS 2021,
arXiv:2106.02039) to clinical ICU data, using continuous linear embeddings rather than
the original discretized-token approach. Each 4-hour ICU bloc is one token; the model
attends causally over the patient's full history to predict the next state.

Input at each time step t:
    state_t  : (state_dim,) float -- z-scored clinical features
    action_t : (action_dim,) float -- binary drug flags (0 or 1)

The embedding for step t is:
    x_t = W_s * state_t + W_a * action_t + pos_embed[t]

A causal transformer (each position attends only to past positions) processes
the full sequence and the output h_t is decoded to:
    next_state_{t+1} : (state_dim,)  via MSE loss
    terminal_t       : scalar logit  via BCE loss

Causal masking is enforced via generate_square_subsequent_mask so the model
cannot use future information when predicting next_state.

Optional: FCI causal constraints (use_causal_constraints=True)
    Adds a separate causally-masked action residual layer on top of the transformer
    next-state prediction. Only FCI-confirmed drug->lab edges contribute signal.
    Derived from Step 09 FCI stability analysis (Tier-2 graph).
    State order : Hb=0, BUN=1, Creatinine=2, HR=3, Shock_Index=4
    Action order: vasopressor=0, ivfluid=1, antibiotic=2, diuretic=3

Default hyperparameters target CPU training on the Tier-2 dataset:
    d_model=64, n_layers=4, n_heads=4 (~250k parameters)
For GPU training, scale to d_model=128-256.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FCI Tier-2 causal graph -- original baseline drug -> state edges
# Rows = state features, Cols = drug actions
# State order : Hb=0, BUN=1, Creatinine=2, HR=3, Shock_Index=4,
#               age=5, charlson_score=6, prior_ed_visits_6m=7  (static -- all zeros)
# Action order: vasopressor=0, ivfluid=1, antibiotic=2, diuretic=3
# Source: original Step 09 FCI stability analysis
# ---------------------------------------------------------------------------
FCI_TIER2_CAUSAL_MASK = torch.tensor([
    #  vasopr  ivfluid  antibiotic  diuretic
    [   0,      0,         0,          1   ],   # Hb
    [   0,      0,         0,          1   ],   # BUN
    [   0,      1,         0,          0   ],   # Creatinine
    [   1,      0,         1,          0   ],   # HR
    [   1,      0,         0,          0   ],   # Shock_Index
    [   0,      0,         0,          0   ],   # age          (static -- drugs don't change age)
    [   0,      0,         0,          0   ],   # charlson_score (static)
    [   0,      0,         0,          0   ],   # prior_ed_visits_6m (static)
], dtype=torch.float32)

# ---------------------------------------------------------------------------
# Selected-set causal graph -- robust Step 04b-supported edges
# Rows = state features, Cols = actions
# State order :
#   Hb=0, BUN=1, Creatinine=2, Phosphate=3, HR=4, Chloride=5,
#   age=6, charlson_score=7, prior_ed_visits_6m=8
# Action order:
#   vasopressor=0, ivfluid=1, antibiotic=2, diuretic=3, mechvent=4
#
# This mask encodes the main retained links from the robust Step 04b analysis
# that were also used in the selected-set recommendation:
#   Hb         <- vasopressor, ivfluid, antibiotic, mechvent
#   BUN        <- ivfluid, diuretic
#   Creatinine <- ivfluid, diuretic
#   Phosphate  <- ivfluid, antibiotic, diuretic
#   HR         <- vasopressor, mechvent
#   Chloride   <- ivfluid, diuretic
# Static confounders stay action-invariant.
# ---------------------------------------------------------------------------
FCI_SELECTED_CAUSAL_MASK = torch.tensor([
    #  vasopr  ivfluid  antibiotic  diuretic  mechvent
    [   1,      1,         1,          0,        1   ],   # Hb
    [   0,      1,         0,          1,        0   ],   # BUN
    [   0,      1,         0,          1,        0   ],   # Creatinine
    [   0,      1,         1,          1,        0   ],   # Phosphate
    [   1,      0,         0,          0,        1   ],   # HR
    [   0,      1,         0,          1,        0   ],   # Chloride
    [   0,      0,         0,          0,        0   ],   # age
    [   0,      0,         0,          0,        0   ],   # charlson_score
    [   0,      0,         0,          0,        0   ],   # prior_ed_visits_6m
], dtype=torch.float32)

DYNAMIC_STATE_IDX = (0, 1, 2, 3, 4)
STATIC_STATE_IDX = (5, 6, 7)


def default_causal_mask_for_shape(state_dim: int, action_dim: int) -> torch.Tensor | None:
    """Return the built-in causal mask matching a known CARE-Sim track shape."""
    if (state_dim, action_dim) == tuple(FCI_TIER2_CAUSAL_MASK.shape):
        return FCI_TIER2_CAUSAL_MASK.clone()
    if (state_dim, action_dim) == tuple(FCI_SELECTED_CAUSAL_MASK.shape):
        return FCI_SELECTED_CAUSAL_MASK.clone()
    return None


class CareSimGPT(nn.Module):
    """Causal GPT-style world model for ICU patient trajectories.

    Args:
        state_dim              : number of state features (default 5 for Tier-2)
        action_dim             : number of binary drug action features (default 4 for Tier-2)
        d_model                : transformer hidden dimension
        n_heads                : number of attention heads (must divide d_model)
        n_layers               : number of transformer encoder layers
        dropout                : dropout probability (applied to embeddings and attention)
        max_seq_len            : maximum ICU stay length in blocs
        use_causal_constraints : if True, adds FCI-masked action residual to next-state
                                 prediction. Enforces that only causally confirmed
                                 drug->lab edges contribute to state predictions.
        causal_mask_matrix     : (state_dim, action_dim) float tensor of 0/1 values
                                 defining which action affects which state output.
                                 Defaults to FCI_TIER2_CAUSAL_MASK if None and
                                 use_causal_constraints=True.
        freeze_static_context  : if True, the model predicts only dynamic next-state
                                 dimensions and copies static confounders straight
                                 through from the current state.
        use_time_feature       : if True, adds a learned embedding of the elapsed
                                 bloc index within the ICU stay.
        predict_reward         : if True, include a reward head. When False, the
                                 model is a pure transition + terminal model.
    """

    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 80,
        use_causal_constraints: bool = False,
        causal_mask_matrix: torch.Tensor | None = None,
        freeze_static_context: bool = False,
        use_time_feature: bool = False,
        predict_reward: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_causal_constraints = use_causal_constraints
        self.freeze_static_context = freeze_static_context
        self.use_time_feature = use_time_feature
        self.predict_reward = predict_reward
        self.dynamic_state_idx = tuple(i for i in DYNAMIC_STATE_IDX if i < state_dim)
        self.static_state_idx = tuple(i for i in STATIC_STATE_IDX if i < state_dim)
        self.predicted_state_dim = len(self.dynamic_state_idx) if freeze_static_context else state_dim

        # --- Embedding layers ---
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        if self.use_time_feature:
            self.time_proj = nn.Linear(1, d_model)
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_drop = nn.Dropout(dropout)

        # --- Causal transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # expects (batch, seq, d_model)
            norm_first=True,    # pre-LN (more stable, GPT-2 style)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,  # avoids warning with causal mask
        )

        # --- Output heads ---
        self.head_state = nn.Linear(d_model, self.predicted_state_dim)  # next state prediction
        self.head_reward = nn.Linear(d_model, 1) if predict_reward else None
        self.head_terminal = nn.Linear(d_model, 1)          # terminal (done) logit

        # --- Optional: FCI causal action residual ---
        # A masked linear layer that adds structured drug effects on top of the
        # transformer prediction. Only FCI-confirmed drug->lab edges are active.
        # Weights are multiplied element-wise by the mask after each update so
        # non-causal connections are permanently zeroed.
        if use_causal_constraints:
            if causal_mask_matrix is None:
                causal_mask_matrix = default_causal_mask_for_shape(state_dim, action_dim)
            if causal_mask_matrix is None:
                raise ValueError(
                    "No built-in causal mask matches "
                    f"(state_dim={state_dim}, action_dim={action_dim}). "
                    "Pass causal_mask_matrix explicitly for this track."
                )
            assert causal_mask_matrix.shape == (state_dim, action_dim), (
                f"causal_mask_matrix must be ({state_dim}, {action_dim}), "
                f"got {tuple(causal_mask_matrix.shape)}"
            )
            self.register_buffer("causal_mask_matrix", causal_mask_matrix)
            # bias=False so the constraint layer only models the drug delta
            self.causal_action_layer = nn.Linear(action_dim, state_dim, bias=False)
            # Initialise to small values; mask will be applied after each step
            nn.init.normal_(self.causal_action_layer.weight, mean=0.0, std=0.01)
            self.causal_action_layer.weight.data *= causal_mask_matrix

        self._init_weights()

    def _init_weights(self):
        """GPT-style weight initialization."""
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
        """Upper-triangular mask: True = masked (ignored) positions."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def embed(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        time_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute token embeddings from states and actions.

        Args:
            states  : (B, T, state_dim)
            actions : (B, T, action_dim)
            time_steps : (B, T) float or None -- elapsed bloc indices
        Returns:
            x       : (B, T, d_model)
        """
        B, T, _ = states.shape
        pos = torch.arange(T, device=states.device)          # (T,)
        x = self.state_proj(states) + self.action_proj(actions) + self.pos_embed(pos)
        if self.use_time_feature:
            if time_steps is None:
                time_steps = pos.unsqueeze(0).expand(B, T).to(states.device)
            time_feat = torch.log1p(time_steps.float()).unsqueeze(-1)
            x = x + self.time_proj(time_feat)
        return self.embed_drop(self.embed_norm(x))

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        time_steps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass over a batch of patient sequences.

        Args:
            states               : (B, T, state_dim)   -- current states
            actions              : (B, T, action_dim)  -- actions taken
            src_key_padding_mask : (B, T) bool         -- True = padding, ignored in attention
            time_steps           : (B, T) float or None -- elapsed bloc indices in stay

        Returns dict with:
            next_state : (B, T, state_dim) -- predicted next state at each step
            terminal   : (B, T)            -- predicted terminal logit at each step
        """
        B, T, _ = states.shape

        x = self.embed(states, actions, time_steps=time_steps)  # (B, T, d_model)
        causal_mask = self._make_causal_mask(T, states.device) # (T, T)

        h = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True,
        )                                                       # (B, T, d_model)

        if self.freeze_static_context:
            next_state = states.clone()
            next_state[..., list(self.dynamic_state_idx)] = self.head_state(h)
        else:
            next_state = self.head_state(h)                    # (B, T, state_dim)

        if self.use_causal_constraints:
            # Apply FCI-masked action residual: only causally confirmed drug->lab
            # edges contribute. The mask zeros out non-parent connections so the
            # layer cannot learn spurious drug effects.
            masked_weight = self.causal_action_layer.weight * self.causal_mask_matrix
            action_residual = F.linear(actions, masked_weight) # (B, T, state_dim)
            next_state = next_state + action_residual
            if self.freeze_static_context and self.static_state_idx:
                next_state[..., list(self.static_state_idx)] = states[..., list(self.static_state_idx)]

        if self.freeze_static_context:
            state_loss_mask = torch.zeros(self.state_dim, device=states.device, dtype=states.dtype)
            state_loss_mask[list(self.dynamic_state_idx)] = 1.0
        else:
            state_loss_mask = torch.ones(self.state_dim, device=states.device, dtype=states.dtype)

        reward = self.head_reward(h).squeeze(-1) if self.head_reward is not None else None
        return {
            "next_state": next_state,                          # (B, T, state_dim)
            "reward": reward,                                  # (B, T) or None
            "terminal": self.head_terminal(h).squeeze(-1),     # (B, T)
            "state_loss_mask": state_loss_mask,                # (state_dim,)
        }

    def predict_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        time_steps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict the next state/reward/terminal using the full sequence history.

        This is the key method for the simulator: given a history of (state, action)
        pairs, predict the outcome of the LAST action taken.

        Args:
            states  : (B, T, state_dim)  -- history including current state at position T-1
            actions : (B, T, action_dim) -- history including current action at position T-1
        Returns dict with:
            next_state : (B, state_dim)  -- predicted next state after last action
            terminal   : (B,)            -- terminal probability for last step (sigmoid applied)
        """
        with torch.no_grad():
            out = self.forward(states, actions, time_steps=time_steps)
        return {
            "next_state": out["next_state"][:, -1, :],           # last position
            "reward": None if out["reward"] is None else out["reward"][:, -1],
            "terminal": torch.sigmoid(out["terminal"][:, -1]),   # convert logit to probability
        }

    def enforce_causal_mask(self) -> None:
        """Zero out weights in the causal action layer that violate the FCI graph.

        Call this after each optimizer.step() when use_causal_constraints=True.
        Prevents gradient updates from leaking signal through non-parent edges.

        Example training loop:
            optimizer.step()
            model.enforce_causal_mask()   # <-- add this line
        """
        if self.use_causal_constraints:
            with torch.no_grad():
                self.causal_action_layer.weight *= self.causal_mask_matrix

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
