"""Vendored DYNOTEARS solver -- structure learning from time-series data.

Vendored from causalnex (Apache 2.0 License, QuantumBlack / McKinsey).
Original: https://github.com/mckinsey/causalnex/blob/develop/causalnex/structure/dynotears.py

Reason for vendoring: causalnex requires Python <3.11, we run 3.11.9.
Only numpy and scipy are needed -- no causalnex dependency.

CHANGES FROM ORIGINAL:
  1. Removed StructureModel / networkx dependency -- returns raw (W, A) numpy arrays
  2. Removed DynamicDataTransformer dependency -- caller builds X, Xlags
  3. Added iteration-level logging (log.info) to the outer loop of the
     augmented Lagrangian optimisation for progress monitoring
  4. No algorithmic changes -- optimisation, gradient, convergence criteria
     are identical to the published DYNOTEARS paper (Pamfil et al., 2020)

Reference:
  @inproceedings{pamfil2020dynotears,
      title={DYNOTEARS: Structure Learning from Time-Series Data},
      author={Pamfil, Roxana and Sriwattanaworachai, Nisara and Desai, Shaan
              and Pilgerstorfer, Philip and Georgatzis, Konstantinos
              and Beaumont, Paul and Aragam, Bryon},
      booktitle={AISTATS},
      year={2020},
  }
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def dynotears_solve(
    X: np.ndarray,
    Xlags: np.ndarray,
    lambda_w: float = 0.1,
    lambda_a: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    w_threshold: float = 0.0,
    tabu_edges: Optional[List[Tuple[int, int, int]]] = None,
    tabu_parent_nodes: Optional[List[int]] = None,
    tabu_child_nodes: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run DYNOTEARS and return (W, A) adjacency matrices.

    Args:
        X:  (n, d) current-time observations.
        Xlags: (n, d*p) horizontally stacked lagged observations
               [X_{t-1} | X_{t-2} | ... | X_{t-p}].
        lambda_w: L1 penalty on W (contemporaneous / intra-slice).
        lambda_a: L1 penalty on A (lagged / inter-slice).
        max_iter: Max augmented-Lagrangian iterations.
        h_tol: Acyclicity tolerance (exit when h(W) < h_tol).
        w_threshold: Post-hoc pruning -- zero out edges with |weight| below this.
        tabu_edges: List of (lag, from_idx, to_idx) edges to forbid.
            lag=0 -> intra-slice (W), lag>0 -> inter-slice (A).
        tabu_parent_nodes: Variable indices banned from being a parent.
        tabu_child_nodes: Variable indices banned from being a child.

    Returns:
        W: (d, d) intra-slice adjacency matrix.  W[i,j] != 0  =>  j -> i at time t.
        A: (d*p, d) inter-slice adjacency matrix.  A[i,j] != 0  =>  lag-var i -> j at time t.
    """
    if X.size == 0:
        raise ValueError("Input data X is empty")
    if Xlags.size == 0:
        raise ValueError("Input data Xlags is empty")
    if X.shape[0] != Xlags.shape[0]:
        raise ValueError("X and Xlags must have the same number of rows")
    if Xlags.shape[1] % X.shape[1] != 0:
        raise ValueError("Xlags columns must be a multiple of X columns")

    _, d_vars = X.shape
    p_orders = Xlags.shape[1] // d_vars

    # -- Build box constraints (bounds) ------------------------------------
    bnds_w = 2 * [
        (0, 0)
        if i == j
        else (0, 0)
        if tabu_edges is not None and (0, i, j) in tabu_edges
        else (0, 0)
        if tabu_parent_nodes is not None and i in tabu_parent_nodes
        else (0, 0)
        if tabu_child_nodes is not None and j in tabu_child_nodes
        else (0, None)
        for i in range(d_vars)
        for j in range(d_vars)
    ]

    bnds_a: list = []
    for k in range(1, p_orders + 1):
        bnds_a.extend(
            2 * [
                (0, 0)
                if tabu_edges is not None and (k, i, j) in tabu_edges
                else (0, 0)
                if tabu_parent_nodes is not None and i in tabu_parent_nodes
                else (0, 0)
                if tabu_child_nodes is not None and j in tabu_child_nodes
                else (0, None)
                for i in range(d_vars)
                for j in range(d_vars)
            ]
        )

    bnds = bnds_w + bnds_a

    # -- Run optimisation --------------------------------------------------
    w_est, a_est = _learn_dynamic_structure(
        X, Xlags, bnds, lambda_w, lambda_a, max_iter, h_tol,
    )

    # -- Post-hoc threshold ------------------------------------------------
    w_est[np.abs(w_est) < w_threshold] = 0
    a_est[np.abs(a_est) < w_threshold] = 0

    return w_est, a_est


# ---------------------------------------------------------------------------
# Data preparation helper (replaces DynamicDataTransformer)
# ---------------------------------------------------------------------------

def build_X_Xlags(
    realisations: List[np.ndarray],
    p: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a list of time-series realisations into (X, Xlags) matrices.

    Each element of *realisations* is a 2-D array of shape (T_i, d) where
    T_i is the length of that realisation and d is the number of variables.
    All realisations must have the same number of columns (d).

    For lag order p, the first p rows of each realisation are consumed as
    history -- so a realisation of length T_i contributes (T_i - p) rows.

    Args:
        realisations: List of (T_i, d) arrays, one per continuous segment.
        p: Lag order (typically 1).

    Returns:
        X:     (N, d)   current-time observations (pooled across segments).
        Xlags: (N, d*p) horizontally stacked lagged observations.
    """
    X_parts = []
    Xlags_parts = []
    for arr in realisations:
        T = arr.shape[0]
        if T <= p:
            continue  # segment too short
        X_parts.append(arr[p:])
        lags = np.concatenate(
            [arr[p - i - 1: T - i - 1] for i in range(p)],
            axis=1,
        )
        Xlags_parts.append(lags)

    if not X_parts:
        raise ValueError(
            f"No realisations have length > p={p}. "
            "Need at least p+1 consecutive time points."
        )

    X = np.concatenate(X_parts, axis=0)
    Xlags = np.concatenate(Xlags_parts, axis=0)
    return X, Xlags


# ---------------------------------------------------------------------------
# Internal: reshape helper
# ---------------------------------------------------------------------------

def _reshape_wa(
    wa_vec: np.ndarray, d_vars: int, p_orders: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform flat adjacency vector into (W, A) matrices."""
    w_tilde = wa_vec.reshape([2 * (p_orders + 1) * d_vars, d_vars])
    w_plus = w_tilde[:d_vars, :]
    w_minus = w_tilde[d_vars: 2 * d_vars, :]
    w_mat = w_plus - w_minus
    a_plus = (
        w_tilde[2 * d_vars:]
        .reshape(2 * p_orders, d_vars ** 2)[::2]
        .reshape(d_vars * p_orders, d_vars)
    )
    a_minus = (
        w_tilde[2 * d_vars:]
        .reshape(2 * p_orders, d_vars ** 2)[1::2]
        .reshape(d_vars * p_orders, d_vars)
    )
    a_mat = a_plus - a_minus
    return w_mat, a_mat


# ---------------------------------------------------------------------------
# Internal: core optimisation (augmented Lagrangian + L-BFGS-B)
# ---------------------------------------------------------------------------

def _learn_dynamic_structure(
    X: np.ndarray,
    Xlags: np.ndarray,
    bnds: List[Tuple[float, float]],
    lambda_w: float = 0.1,
    lambda_a: float = 0.1,
    max_iter: int = 100,
    h_tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Augmented-Lagrangian solver for DYNOTEARS.

    Minimises:
        F(W,A) = 0.5/n * ||X - X*W - Xlags*A||^2_F
                 + lambda_w * |W|_1  +  lambda_a * |A|_1
    subject to:
        h(W) = tr(e^{W o W}) - d = 0   (acyclicity of W)

    Algorithm logic is IDENTICAL to the original causalnex implementation.
    Only addition: log.info at each outer iteration for progress monitoring.
    """
    n, d_vars = X.shape
    p_orders = Xlags.shape[1] // d_vars

    def _h(wa_vec: np.ndarray) -> float:
        _w_mat, _ = _reshape_wa(wa_vec, d_vars, p_orders)
        return np.trace(slin.expm(_w_mat * _w_mat)) - d_vars

    def _func(wa_vec: np.ndarray) -> float:
        _w_mat, _a_mat = _reshape_wa(wa_vec, d_vars, p_orders)
        loss = (
            0.5
            / n
            * np.square(
                np.linalg.norm(
                    X.dot(np.eye(d_vars, d_vars) - _w_mat) - Xlags.dot(_a_mat),
                    "fro",
                )
            )
        )
        _h_value = _h(wa_vec)
        l1_penalty = lambda_w * wa_vec[: 2 * d_vars ** 2].sum() + lambda_a * (
            wa_vec[2 * d_vars ** 2:].sum()
        )
        return loss + 0.5 * rho * _h_value * _h_value + alpha * _h_value + l1_penalty

    def _grad(wa_vec: np.ndarray) -> np.ndarray:
        _w_mat, _a_mat = _reshape_wa(wa_vec, d_vars, p_orders)
        e_mat = slin.expm(_w_mat * _w_mat)
        loss_grad_w = (
            -1.0
            / n
            * X.T.dot(X.dot(np.eye(d_vars, d_vars) - _w_mat) - Xlags.dot(_a_mat))
        )
        obj_grad_w = (
            loss_grad_w
            + (rho * (np.trace(e_mat) - d_vars) + alpha) * e_mat.T * _w_mat * 2
        )
        obj_grad_a = (
            -1.0
            / n
            * Xlags.T.dot(
                X.dot(np.eye(d_vars, d_vars) - _w_mat) - Xlags.dot(_a_mat)
            )
        )

        grad_vec_w = np.append(
            obj_grad_w, -obj_grad_w, axis=0,
        ).flatten() + lambda_w * np.ones(2 * d_vars ** 2)
        grad_vec_a = obj_grad_a.reshape(p_orders, d_vars ** 2)
        grad_vec_a = np.hstack(
            (grad_vec_a, -grad_vec_a),
        ).flatten() + lambda_a * np.ones(2 * p_orders * d_vars ** 2)
        return np.append(grad_vec_w, grad_vec_a, axis=0)

    # -- Initialise --------------------------------------------------------
    wa_est = np.zeros(2 * (p_orders + 1) * d_vars ** 2)
    wa_new = np.zeros(2 * (p_orders + 1) * d_vars ** 2)
    rho, alpha, h_value, h_new = 1.0, 0.0, np.inf, np.inf

    log.info("DYNOTEARS optimisation: d=%d, p=%d, n=%d, params=%d",
             d_vars, p_orders, n, len(wa_est))

    for n_iter in range(max_iter):
        inner_count = 0
        while (rho < 1e20) and (h_new > 0.25 * h_value or h_new == np.inf):
            wa_new = sopt.minimize(
                _func, wa_est, method="L-BFGS-B", jac=_grad, bounds=bnds,
            ).x
            h_new = _h(wa_new)
            inner_count += 1
            if h_new > 0.25 * h_value:
                rho *= 10

        wa_est = wa_new
        h_value = h_new
        alpha += rho * h_value

        # --- Progress logging (ADDED -- not in original) ---
        _w_tmp, _a_tmp = _reshape_wa(wa_est, d_vars, p_orders)
        n_w = int(np.count_nonzero(_w_tmp))
        n_a = int(np.count_nonzero(_a_tmp))
        log.info(
            "  iter %2d: h=%.2e, rho=%.1e, alpha=%.2e, "
            "inner_steps=%d, W_nonzero=%d, A_nonzero=%d",
            n_iter, h_value, rho, alpha, inner_count, n_w, n_a,
        )

        if h_value <= h_tol:
            log.info("  Converged at iteration %d (h=%.2e <= h_tol=%.2e)",
                     n_iter, h_value, h_tol)
            break

        if h_value > h_tol and n_iter == max_iter - 1:
            warnings.warn("Failed to converge. Consider increasing max_iter.")
            log.warning("  DID NOT CONVERGE after %d iterations (h=%.2e)",
                        max_iter, h_value)

    return _reshape_wa(wa_est, d_vars, p_orders)
