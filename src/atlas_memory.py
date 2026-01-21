"""
Atlas Memory Module - PRD v4.2 Section 2

Implements:
- Omega Rule: Context window optimization for memory updates
- Polynomial feature expansion integration
- Newton-Schulz internal update rule
- Data-dependent gates (alpha, eta, theta, gamma)

CRITICAL: This is the MEMORY UPDATE algorithm, not the full block.
The AtlasMAGBlock combines this with sliding window attention.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

from .polynomial import PolynomialFeatureLayer
from .newton_schulz import newton_schulz_k, get_newton_schulz_iterations


class AtlasMemory(nn.Module):
    """
    Atlas Deep Memory Module with Omega Rule.

    The Omega rule optimizes memory over a sliding window of c tokens,
    allowing the memory to capture context rather than single-token patterns.

    Update equations:
        S_t = θ_t · S_{t-1} + ∇(Omega loss)
        M_t = α_t · M_{t-1} - η_t · NS_K(S_t)

    Where:
        - α_t: Memory decay gate (learned per-token relevance)
        - η_t: Learning rate gate
        - θ_t: Momentum coefficient
        - γ_i: Per-token relevance weights for Omega gradient
        - NS_K: Newton-Schulz iteration (K iterations)

    Args:
        d_model: Model dimension
        poly_degree: Polynomial feature degree (default: 2 for O(d²) capacity)
        context_window: Omega context window size (default: 8)
        ns_iterations: Base Newton-Schulz iterations (default: 1)
        adaptive_ns: Whether to use adaptive NS iterations based on steps (default: True)
    """

    # Type annotations for register_buffer attributes
    steps_since_reset: torch.Tensor
    gradient_norm_ema: torch.Tensor

    def __init__(
        self,
        d_model: int,
        poly_degree: int = 2,
        context_window: int = 8,
        ns_iterations: int = 1,
        adaptive_ns: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_window = context_window
        self.ns_iterations = ns_iterations
        self.adaptive_ns = adaptive_ns

        # Polynomial feature expansion
        self.poly_layer = PolynomialFeatureLayer(d_model, degree=poly_degree)

        # Learnable initial states
        self.M_init = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.S_init = nn.Parameter(torch.zeros(d_model, d_model))

        # Data-dependent gate projections
        # These project input x to gate values
        self.alpha_proj = nn.Linear(d_model, d_model)  # Memory decay
        self.eta_proj = nn.Linear(d_model, d_model)    # Learning rate
        self.theta_proj = nn.Linear(d_model, d_model)  # Momentum
        self.gamma_proj = nn.Linear(d_model, 1)        # Per-token relevance

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Track steps for adaptive NS
        self.register_buffer('steps_since_reset', torch.tensor(0))

        # Telemetry
        self.register_buffer('gradient_norm_ema', torch.tensor(0.0))

    def _compute_omega_gradient(
        self,
        M: torch.Tensor,           # [B, d_out, d_in]
        keys_phi: torch.Tensor,    # [B, c, D] - polynomial expanded
        values: torch.Tensor,      # [B, c, D]
        gamma: torch.Tensor,       # [B, c] - per-token relevance
    ) -> torch.Tensor:
        """
        Compute the Omega gradient over context window.

        Objective:
                        t
            min_M   Σ      γ_i · ||M(φ(k_i)) - v_i||²
                   i=t-c+1

        Gradient:
            ∇ = Σ γ_i · (M @ φ(k_i) - v_i) ⊗ φ(k_i)
        """
        B, c, D = keys_phi.shape

        grad = torch.zeros(B, D, D, device=keys_phi.device, dtype=keys_phi.dtype)

        for i in range(c):
            k_i = keys_phi[:, i, :]  # [B, D]
            v_i = values[:, i, :]    # [B, D]

            # Prediction: M @ k_i
            pred = torch.bmm(M, k_i.unsqueeze(-1)).squeeze(-1)  # [B, D]

            # Error
            error = pred - v_i  # [B, D]

            # Weighted outer product: γ_i * error ⊗ k_i
            # [B, D, 1] @ [B, 1, D] = [B, D, D]
            outer = torch.bmm(error.unsqueeze(-1), k_i.unsqueeze(1))
            grad = grad + gamma[:, i:i+1, None] * outer

        return grad / c  # Average over window

    def forward(
        self,
        x: torch.Tensor,           # [B, L, D] - input
        k_aligned: torch.Tensor,   # [B, L, D] - aligned keys (from Q-K projection)
        v: torch.Tensor,           # [B, L, D] - values
        M_prev: torch.Tensor,      # [B, D, D] - previous memory
        S_prev: torch.Tensor,      # [B, D, D] - previous momentum
        shard_boundary: bool = False,
        steps_since_reset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Atlas memory forward pass with Omega rule update.

        NOTE: k_aligned should already be processed through QKProjectionLayer
        (which applies tanh safety gate). This layer applies polynomial
        expansion to the aligned, safe keys.

        Args:
            x: Input sequence [B, L, D]
            k_aligned: Keys after Q-K projection alignment [B, L, D]
            v: Values [B, L, D]
            M_prev: Previous memory state [B, D, D]
            S_prev: Previous momentum state [B, D, D]
            shard_boundary: Whether at shard boundary (for initialization)
            steps_since_reset: Steps since last shard boundary (for adaptive NS)

        Returns:
            output: Memory output [B, L, D]
            M_new: Updated memory [B, D, D]
            S_new: Updated momentum [B, D, D]
            telemetry: Dict with monitoring metrics
        """
        B, L, D = x.shape

        # Initialize states if at shard boundary
        if shard_boundary:
            M = self.M_init.unsqueeze(0).expand(B, -1, -1).clone()
            S = self.S_init.unsqueeze(0).expand(B, -1, -1).clone()
            self.steps_since_reset.zero_()
        else:
            M = M_prev
            S = S_prev

        # Apply polynomial features to aligned keys
        k_phi = self.poly_layer(k_aligned)  # [B, L, D]

        outputs = []
        gradient_norms = []

        for t in range(L):
            # Current token
            x_t = x[:, t, :]  # [B, D]
            k_t = k_phi[:, t, :]  # [B, D]
            v_t = v[:, t, :]  # [B, D]

            # Compute data-dependent gates
            alpha = torch.sigmoid(self.alpha_proj(x_t))  # [B, D] - memory decay
            eta = torch.sigmoid(self.eta_proj(x_t)) * 0.1  # [B, D] - learning rate (scaled)
            theta = torch.sigmoid(self.theta_proj(x_t))  # [B, D] - momentum

            # Build context window for Omega gradient
            window_start = max(0, t - self.context_window + 1)
            window_end = t + 1
            c = window_end - window_start

            if c > 0:
                keys_window = k_phi[:, window_start:window_end, :]   # [B, c, D]
                values_window = v[:, window_start:window_end, :]     # [B, c, D]
                x_window = x[:, window_start:window_end, :]         # [B, c, D]

                # Compute per-token relevance (gamma)
                gamma = torch.sigmoid(self.gamma_proj(x_window)).squeeze(-1)  # [B, c]

                # Compute Omega gradient
                grad = self._compute_omega_gradient(M, keys_window, values_window, gamma)

                # Track gradient norm for telemetry
                grad_norm = torch.norm(grad, p='fro', dim=(-2, -1)).mean().item()
                gradient_norms.append(grad_norm)

                # Update momentum: S_t = θ_t · S_{t-1} + grad
                # Note: theta is per-dimension, broadcast to matrix
                S = theta.unsqueeze(-1) * S + grad

                # Get number of NS iterations (adaptive if enabled)
                if self.adaptive_ns:
                    ns_k = get_newton_schulz_iterations(steps_since_reset + t)
                else:
                    ns_k = self.ns_iterations

                # Apply Newton-Schulz
                S_ortho = newton_schulz_k(S, k=ns_k)

                # Update memory: M_t = α_t · M_{t-1} - η_t · S_ortho
                # Note: alpha, eta are per-dimension, broadcast to matrix
                M = alpha.unsqueeze(-1) * M - eta.unsqueeze(-1) * S_ortho

            # Memory retrieval: y = M @ φ(k_t)
            y_t = torch.bmm(M, k_t.unsqueeze(-1)).squeeze(-1)  # [B, D]

            # Output projection
            y_t = self.out_proj(y_t)
            outputs.append(y_t)

        output = torch.stack(outputs, dim=1)  # [B, L, D]

        # Update telemetry
        if gradient_norms:
            avg_grad_norm = sum(gradient_norms) / len(gradient_norms)
            self.gradient_norm_ema = 0.99 * self.gradient_norm_ema + 0.01 * avg_grad_norm

        self.steps_since_reset += L

        telemetry = {
            'gradient_norm_ema': self.gradient_norm_ema.item(),
            'steps_since_reset': self.steps_since_reset.item(),
            'context_window': self.context_window,
            'ns_iterations': ns_k if self.adaptive_ns else self.ns_iterations,
        }

        return output, M, S, telemetry

    def get_init_states(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial states for a new sequence."""
        M = self.M_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)
        S = self.S_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)
        return M, S

    def reset_telemetry(self) -> None:
        """Reset telemetry counters."""
        self.gradient_norm_ema.zero_()
        self.steps_since_reset.zero_()

    def extra_repr(self) -> str:
        return (
            f'd_model={self.d_model}, '
            f'context_window={self.context_window}, '
            f'ns_iterations={self.ns_iterations}, '
            f'adaptive_ns={self.adaptive_ns}'
        )


def atlas_memory_update(
    M_prev: torch.Tensor,      # Previous memory [B, d_out, d_in]
    S_prev: torch.Tensor,      # Previous momentum [B, d_out, d_in]
    keys: torch.Tensor,        # Keys for window [B, c, D]
    values: torch.Tensor,      # Values for window [B, c, D]
    alpha: torch.Tensor,       # Memory decay gate [B, D]
    eta: torch.Tensor,         # Learning rate gate [B, D]
    theta: torch.Tensor,       # Momentum coefficient [B, D]
    gamma: torch.Tensor,       # Per-token relevance [B, c]
    poly_layer: nn.Module,
    ns_iterations: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standalone Atlas update function (for reference/testing).

    Atlas update equations:
        S_t = θ_t · S_{t-1} + ∇(Omega loss)
        M_t = α_t · M_{t-1} - η_t · NS_K(S_t)

    Args:
        M_prev: Previous memory state
        S_prev: Previous momentum state
        keys: Keys for context window (raw, will be expanded)
        values: Values for context window
        alpha: Memory decay gate
        eta: Learning rate gate
        theta: Momentum coefficient
        gamma: Per-token relevance weights
        poly_layer: Polynomial feature layer
        ns_iterations: Number of Newton-Schulz iterations

    Returns:
        M_t: Updated memory
        S_t: Updated momentum
    """
    # Apply polynomial features
    keys_phi = poly_layer(keys)

    B, c, D = keys_phi.shape

    # Compute Omega gradient (over context window)
    grad = torch.zeros(B, D, D, device=keys_phi.device, dtype=keys_phi.dtype)

    for i in range(c):
        k_i = keys_phi[:, i, :]  # [B, D]
        v_i = values[:, i, :]    # [B, D]

        # Prediction
        pred = torch.bmm(M_prev, k_i.unsqueeze(-1)).squeeze(-1)

        # Error
        error = pred - v_i

        # Weight by γ (learned relevance)
        outer = torch.bmm(error.unsqueeze(-1), k_i.unsqueeze(1))
        grad = grad + gamma[:, i:i+1, None] * outer

    grad = grad / c

    # Update momentum: S_t = θ · S_{t-1} + grad
    S_t = theta.unsqueeze(-1) * S_prev + grad

    # Apply Newton-Schulz (internal update rule)
    S_ortho = newton_schulz_k(S_t, k=ns_iterations)

    # Update memory: M_t = α · M_{t-1} - η · S_ortho
    M_t = alpha.unsqueeze(-1) * M_prev - eta.unsqueeze(-1) * S_ortho

    return M_t, S_t
