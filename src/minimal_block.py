"""
Minimal Atlas Block - PRD v4.2 Section 10.1

Simplified Atlas block for component isolation testing.
Strips away complexity to test core mechanisms:
- Tanh saturation (can it learn a line?)
- M3 mixing (can it recover from poisoning?)
- Polynomial features (does capacity scale?)

Use this for the isolation test suite BEFORE full training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .polynomial import PolynomialFeatureLayer
from .qk_projection import QKProjectionLayer
from .m3_mixing import M3MixingState, reset_atlas_states
from .newton_schulz import newton_schulz_k


class MinimalAtlasBlock(nn.Module):
    """
    Minimal Atlas block for component isolation testing.

    Stripped-down version that preserves the critical signal path:
    x → Q-K Projection → tanh gate → Polynomial → Memory → output

    No attention branch, no complex gating - just the memory path.
    This isolates the mechanisms that can fail silently.

    Args:
        d_model: Model dimension (keep small for testing, e.g., 16-64)
        poly_degree: Polynomial feature degree
        m3_alpha_target: Target alpha for M3 mixing
        m3_alpha_start: Initial alpha during warmup
        m3_warmup_steps: Steps to ramp alpha
    """

    def __init__(
        self,
        d_model: int = 64,
        poly_degree: int = 2,
        m3_alpha_target: float = 0.5,
        m3_alpha_start: float = 0.1,
        m3_warmup_steps: int = 100,
    ):
        super().__init__()
        self.d_model = d_model

        # Q, K projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

        # Q-K Projection Layer (with tanh safety gate)
        self.qk_proj = QKProjectionLayer(d_model)

        # Polynomial Feature Layer
        self.poly_layer = PolynomialFeatureLayer(d_model, degree=poly_degree)

        # M3 Mixing
        self.m3_mixer = M3MixingState(
            d_model=d_model,
            alpha_target=m3_alpha_target,
            alpha_start=m3_alpha_start,
            alpha_warmup_steps=m3_warmup_steps,
        )

        # Learnable initial states
        self.M_init = nn.Parameter(torch.randn(d_model, d_model) * 0.01)
        self.S_init = nn.Parameter(torch.zeros(d_model, d_model))
        self.P_init = nn.Parameter(torch.eye(d_model) * 0.01)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # State tracking
        self._M: Optional[torch.Tensor] = None
        self._S: Optional[torch.Tensor] = None
        self._P: Optional[torch.Tensor] = None
        self._steps: int = 0

    def reset_states(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        """Reset memory states to initial values."""
        if device is None:
            device = self.M_init.device

        self._M = self.M_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)
        self._S = self.S_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)
        self._P = self.P_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device)
        self._steps = 0

    def forward(
        self,
        x: torch.Tensor,
        shard_boundary: bool = False,
    ) -> torch.Tensor:
        """
        Minimal forward pass for testing.

        Args:
            x: Input tensor [B, L, D] or [B, D] for single step
            shard_boundary: Whether to apply M3 mixing reset

        Returns:
            output: Memory output [B, L, D] or [B, D]
        """
        # Handle both [B, L, D] and [B, D] inputs
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D]
            squeeze_output = True

        B, L, D = x.shape

        # Initialize states if needed
        if self._M is None or self._M.size(0) != B:
            self.reset_states(batch_size=B, device=x.device)

        # Assert states are initialized (mypy type narrowing)
        assert self._M is not None and self._S is not None and self._P is not None

        # Handle shard boundary with M3 mixing
        if shard_boundary:
            self._M, self._S, self._P = reset_atlas_states(
                shard_boundary=True,
                M_init=self.M_init.unsqueeze(0).expand(B, -1, -1),
                S_init=self.S_init.unsqueeze(0).expand(B, -1, -1),
                P_init=self.P_init.unsqueeze(0).expand(B, -1, -1),
                M_prev=self._M,
                S_prev=self._S,
                P_prev=self._P,
                m3_mixer=self.m3_mixer,
            )
            self._steps = 0
            self.m3_mixer.update_telemetry()

        # Project Q, K
        # v4.5: Remove query L2 norm to match AtlasMAGBlock (Issue #5)
        q = self.q_proj(x)  # No L2 norm - let gain control scale
        k = F.normalize(self.k_proj(x), p=2, dim=-1)  # Keep for P structure

        # Q-K Projection (with tanh safety gate)
        q_aligned, P_new, qk_telemetry = self.qk_proj(q, k, self._P)

        # Polynomial expansion
        k_phi = self.poly_layer(k)
        q_phi = self.poly_layer(q_aligned)

        # Simple memory update (single-step, no Omega rule for minimal testing)
        # Use local variables for in-loop computation to avoid graph retention issues
        M_curr = self._M
        S_curr = self._S

        outputs = []
        for t in range(L):
            k_t = k_phi[:, t, :]  # [B, D]
            q_t = q_phi[:, t, :]  # [B, D]

            # Simple gradient: outer product of prediction error
            pred = torch.bmm(M_curr, k_t.unsqueeze(-1)).squeeze(-1)  # [B, D]
            target = x[:, t, :]  # Use input as target for autoencoder-style test
            error = pred - target
            grad = torch.bmm(error.unsqueeze(-1), k_t.unsqueeze(1))  # [B, D, D]

            # Simple momentum update
            S_curr = 0.9 * S_curr + grad

            # Newton-Schulz
            S_ortho = newton_schulz_k(S_curr, k=1)

            # Memory update
            M_curr = 0.99 * M_curr - 0.01 * S_ortho

            # Retrieval
            y_t = torch.bmm(M_curr, q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(y_t)

        stacked = torch.stack(outputs, dim=1)  # [B, L, D]
        output: torch.Tensor = self.out_proj(stacked)

        # Detach states to avoid graph retention across backward passes
        self._M = M_curr.detach()
        self._S = S_curr.detach()
        self._P = P_new.detach()
        self._steps += L

        # Check for kill switch
        if qk_telemetry.get('KILL'):
            raise RuntimeError(qk_telemetry['kill_reason'])

        if squeeze_output:
            squeezed: torch.Tensor = output.squeeze(1)
            return squeezed

        return output

    def get_telemetry(self) -> Dict[str, Any]:
        """Get combined telemetry from all components."""
        # v4.4: input_gain renamed to log_gain, compute effective gain
        effective_gain = torch.exp(self.qk_proj.log_gain)
        return {
            'qk_saturation_ema': self.qk_proj.saturation_ema.item(),
            'qk_gain_frozen': self.qk_proj.gain_frozen.item(),
            'qk_effective_gain_max': effective_gain.abs().max().item(),
            'm3_alpha': self.m3_mixer.alpha.item(),
            'm3_warmup_complete': self.m3_mixer.warmup_complete.item(),
            'm3_alpha_ema': self.m3_mixer.alpha_ema.item(),
            'steps': self._steps,
            'poly_coeffs': self.poly_layer.coeffs.data.tolist(),
        }


class LinearTestModel(nn.Module):
    """
    Even simpler model for the linear line test.

    Just tests if tanh + polynomial can fit y = mx + b.
    """

    # Type annotations for register_buffer attributes
    saturation_ema: torch.Tensor

    def __init__(self, d_model: int = 16):
        super().__init__()
        self.d_model = d_model

        # Minimal path: input → gain → tanh → poly → linear
        self.input_gain = nn.Parameter(torch.ones(d_model) * 0.5)
        self.poly_layer = PolynomialFeatureLayer(d_model, degree=2)
        self.linear = nn.Linear(d_model, d_model)

        # P matrix for Q-K style normalization test
        self.P = nn.Parameter(torch.eye(d_model) * 0.1)

        # Telemetry
        self.register_buffer('saturation_ema', torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [B, D] or [B, L, D]

        Returns:
            y: Output same shape as input
        """
        # Normalize P
        P_norm = self.P / (torch.norm(self.P, p='fro') + 1e-7)

        # Project through P (like Q-K projection)
        if x.dim() == 3:
            B, L, D = x.shape
            x_flat = x.view(B * L, D)
            x_proj = x_flat @ P_norm.T
            x_proj = x_proj.view(B, L, D)
        else:
            x_proj = x @ P_norm.T

        # Apply learnable gain
        x_gated = self.input_gain * x_proj

        # Track saturation
        saturation = (x_gated.abs() > 1.5).float().mean()
        self.saturation_ema = 0.99 * self.saturation_ema + 0.01 * saturation

        # Tanh safety gate
        x_safe = torch.tanh(x_gated)

        # Polynomial features
        x_poly = self.poly_layer(x_safe)

        # Linear output
        y: torch.Tensor = self.linear(x_poly)

        return y

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            'saturation_ema': self.saturation_ema.item(),
            'input_gain_max': self.input_gain.abs().max().item(),
            'P_frobenius_norm': torch.norm(self.P, p='fro').item(),
        }
