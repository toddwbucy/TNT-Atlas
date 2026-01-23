"""
Atlas MAG Block - PRD v4.2 Section 7

Main architecture block combining:
- Sliding window attention (local context)
- Atlas deep memory (long-term context)
- MAG gating mechanism (attention gates memory)

CRITICAL: Uses Project-Then-Poly architecture:
1. Q-K Projection aligns query to key subspace (d-space)
2. THEN polynomial expansion (on aligned query)

State inheritance uses M3 mixing (not binary reset) per v4 principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union

from .polynomial import PolynomialFeatureLayer
from .qk_projection import QKProjectionLayer
from .m3_mixing import M3MixingState, reset_atlas_states
from .attention import SlidingWindowAttention
from .newton_schulz import newton_schulz_k, get_newton_schulz_iterations


class AtlasMAGBlock(nn.Module):
    """
    Atlas (MAG) Block with TNT Training Support.

    Combines:
    - Branch 1: Sliding window attention (local context, fast)
    - Branch 2: Atlas deep memory (long-term context, O(d²) capacity)
    - MAG Gating: Attention output gates memory output

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_size: Sliding window attention size
        n_persistent: Number of persistent memory tokens
        omega_context: Omega rule context window size
        poly_degree: Polynomial feature degree (2 = O(d²) capacity)
        ns_iterations: Base Newton-Schulz iterations
        dropout: Dropout probability
        m3_alpha_target: Target alpha for M3 mixing after warmup
        m3_alpha_start: Initial alpha during M3 warmup
        m3_warmup_steps: Steps to ramp alpha from start to target
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 16,
        window_size: int = 256,
        n_persistent: int = 4,
        omega_context: int = 8,
        poly_degree: int = 2,
        ns_iterations: int = 1,
        dropout: float = 0.0,
        m3_alpha_target: float = 0.5,
        m3_alpha_start: float = 0.1,
        m3_warmup_steps: int = 500,
    ):
        super().__init__()
        self.d_model = d_model
        self.omega_context = omega_context
        self.ns_iterations = ns_iterations

        # ═══════════════════════════════════════════════════════════════
        # BRANCH 1: Sliding Window Attention
        # ═══════════════════════════════════════════════════════════════
        self.attention = SlidingWindowAttention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            n_persistent=n_persistent,
            dropout=dropout,
        )

        # ═══════════════════════════════════════════════════════════════
        # BRANCH 2: Atlas Deep Memory
        # ═══════════════════════════════════════════════════════════════
        # Q, K, V projections for memory branch
        self.mem_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.mem_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.mem_v_proj = nn.Linear(d_model, d_model, bias=False)

        # Q-K Projection Layer (Project-Then-Poly: d-space alignment)
        self.qk_proj_layer = QKProjectionLayer(d_model)

        # Polynomial Feature Layer
        self.poly_layer = PolynomialFeatureLayer(d_model, degree=poly_degree)

        # M3 Mixing for state inheritance
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

        # Data-dependent gate projections
        self.alpha_proj = nn.Linear(d_model, d_model)  # Memory decay
        self.eta_proj = nn.Linear(d_model, d_model)    # Learning rate
        self.theta_proj = nn.Linear(d_model, d_model)  # Momentum
        self.gamma_proj = nn.Linear(d_model, 1)        # Per-token relevance

        # Memory output projection
        self.mem_out_proj = nn.Linear(d_model, d_model)

        # ═══════════════════════════════════════════════════════════════
        # NORMALIZATION (Pre-LN for stability + Post-LN for gating)
        # ═══════════════════════════════════════════════════════════════
        # Pre-LayerNorm: Normalize input before branches (critical for stability)
        self.pre_norm = nn.LayerNorm(d_model)

        # Post-LayerNorm: Normalize branch outputs before gating
        self.attn_norm = nn.LayerNorm(d_model)
        self.mem_norm = nn.LayerNorm(d_model)

        # Learnable scale factors for gating
        self.gamma_attn = nn.Parameter(torch.ones(d_model))
        self.gamma_mem = nn.Parameter(torch.ones(d_model))

        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def _atlas_memory_forward(
        self,
        q_phi: torch.Tensor,       # [B, L, D] - polynomial expanded aligned query
        k_phi: torch.Tensor,       # [B, L, D] - polynomial expanded keys
        v: torch.Tensor,           # [B, L, D] - values
        x: torch.Tensor,           # [B, L, D] - original input (for gates)
        M: torch.Tensor,           # [B, D, D] - memory state
        S: torch.Tensor,           # [B, D, D] - momentum state
        steps: int,                # Steps since last reset
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Dict[str, Any]]:
        """
        Atlas memory forward with Omega rule update.

        Returns:
            y_mem: Memory output [B, L, D]
            M_new: Updated memory
            S_new: Updated momentum
            steps_new: Updated step count
            telemetry: Dict with monitoring metrics
        """
        B, L, D = x.shape

        outputs = []
        gradient_norms = []

        for t in range(L):
            x_t = x[:, t, :]        # [B, D]
            q_t = q_phi[:, t, :]    # [B, D]
            k_t = k_phi[:, t, :]    # [B, D]
            v_t = v[:, t, :]        # [B, D]

            # Compute data-dependent gates
            alpha = torch.sigmoid(self.alpha_proj(x_t))      # [B, D]
            eta = torch.sigmoid(self.eta_proj(x_t)) * 0.1    # [B, D] scaled
            theta = torch.sigmoid(self.theta_proj(x_t))      # [B, D]

            # Build context window for Omega gradient
            window_start = max(0, t - self.omega_context + 1)
            window_end = t + 1
            c = window_end - window_start

            if c > 0:
                k_window = k_phi[:, window_start:window_end, :]
                v_window = v[:, window_start:window_end, :]
                x_window = x[:, window_start:window_end, :]

                # Per-token relevance
                gamma = torch.sigmoid(self.gamma_proj(x_window)).squeeze(-1)

                # Compute Omega gradient
                grad = torch.zeros(B, D, D, device=x.device, dtype=x.dtype)
                for i in range(c):
                    ki = k_window[:, i, :]
                    vi = v_window[:, i, :]
                    pred = torch.bmm(M, ki.unsqueeze(-1)).squeeze(-1)
                    error = pred - vi
                    outer = torch.bmm(error.unsqueeze(-1), ki.unsqueeze(1))
                    grad = grad + gamma[:, i:i+1, None] * outer
                grad = grad / c

                gradient_norms.append(torch.norm(grad, p='fro', dim=(-2, -1)).mean().item())

                # Update momentum
                S = theta.unsqueeze(-1) * S + grad

                # Adaptive Newton-Schulz iterations
                ns_k = get_newton_schulz_iterations(steps + t)
                S_ortho = newton_schulz_k(S, k=ns_k)

                # Update memory
                M = alpha.unsqueeze(-1) * M - eta.unsqueeze(-1) * S_ortho

            # Memory retrieval
            y_t = torch.bmm(M, q_t.unsqueeze(-1)).squeeze(-1)
            outputs.append(y_t)

        y_mem = torch.stack(outputs, dim=1)
        y_mem = self.mem_out_proj(y_mem)

        telemetry = {
            'memory_gradient_norm': sum(gradient_norms) / len(gradient_norms) if gradient_norms else 0,
            'ns_iterations': ns_k if 'ns_k' in dir() else self.ns_iterations,
        }

        return y_mem, M, S, steps + L, telemetry

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[Dict[str, torch.Tensor]] = None,
        shard_boundary: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Forward pass through Atlas MAG block.

        Args:
            x: Input tensor [B, L, D]
            memory_state: Dict with M, S, P, steps (or None for first call)
            shard_boundary: True if starting new shard (TNT reset)

        Returns:
            output: Block output [B, L, D]
            new_state: Updated memory state dict
            telemetry: Monitoring metrics
        """
        B, L, D = x.shape

        # ═══════════════════════════════════════════════════════════════
        # PRE-LAYERNORM: Normalize input before branches (critical for stability)
        # Residual connection uses original x, branches use normalized x_norm
        # ═══════════════════════════════════════════════════════════════
        x_norm = self.pre_norm(x)

        # ═══════════════════════════════════════════════════════════════
        # BRANCH 1: Sliding Window Attention
        # ═══════════════════════════════════════════════════════════════
        y_attn, _ = self.attention(x_norm)  # [B, L, D]

        # ═══════════════════════════════════════════════════════════════
        # BRANCH 2: Atlas Deep Memory
        # ═══════════════════════════════════════════════════════════════
        # Project Q, K, V from normalized input
        #
        # v4.5 FIX (Issue #5): Remove query L2 normalization to fix
        # "double normalization squeeze" that caused Run #7 kill switch.
        #
        # Previously: Both query AND P were normalized, squeezing signal to ~0.001
        # Now: Only keys are L2-normalized (for P subspace structure), queries
        # retain natural magnitude from projection layer.
        #
        # Key insight: Frobenius norm of P ≠ operator norm. The Frobenius
        # normalization reduced effective signal by ~6x. Without query L2 norm,
        # the learnable gain can properly control tanh activation level.
        #
        q = self.mem_q_proj(x_norm)  # No L2 norm - let gain control scale
        k = F.normalize(self.mem_k_proj(x_norm), p=2, dim=-1)  # Keep for P structure
        v = self.mem_v_proj(x_norm)

        # Initialize or unpack memory state
        if memory_state is None:
            M = self.M_init.unsqueeze(0).expand(B, -1, -1).clone()
            S = self.S_init.unsqueeze(0).expand(B, -1, -1).clone()
            P = self.P_init.unsqueeze(0).expand(B, -1, -1).clone()
            steps = 0
        else:
            M = memory_state['M']
            S = memory_state['S']
            P = memory_state['P']
            steps = int(memory_state['steps'])

        # Handle TNT shard boundary with M3 MIXING (not binary reset!)
        if shard_boundary:
            M, S, P = reset_atlas_states(
                shard_boundary=True,
                M_init=self.M_init.unsqueeze(0).expand(B, -1, -1),
                S_init=self.S_init.unsqueeze(0).expand(B, -1, -1),
                P_init=self.P_init.unsqueeze(0).expand(B, -1, -1),
                M_prev=M,
                S_prev=S,
                P_prev=P,
                m3_mixer=self.m3_mixer,
            )
            steps = 0

            # Update M3 telemetry
            self.m3_mixer.update_telemetry()

        # ════════════════════════════════════════════════════════════════
        # CRITICAL: Project-Then-Poly (Q-K projection BEFORE polynomial)
        # ════════════════════════════════════════════════════════════════
        # Step 1: Align query to key subspace (in d-space, NOT d²-space)
        q_aligned, P_new, qk_telemetry = self.qk_proj_layer(q, k, P)

        # Step 2: NOW apply polynomial features (to ALIGNED query)
        k_phi = self.poly_layer(k)           # φ(k) for storage
        q_phi = self.poly_layer(q_aligned)   # φ(q̃) for retrieval - uses ALIGNED query!

        # Atlas memory forward with Omega rule (use x_norm for gate projections)
        y_mem, M_new, S_new, steps_new, mem_telemetry = self._atlas_memory_forward(
            q_phi, k_phi, v, x_norm, M, S, steps
        )

        # ═══════════════════════════════════════════════════════════════
        # MAG GATING: Combine branches
        # ═══════════════════════════════════════════════════════════════
        y_attn_norm = self.attn_norm(y_attn) * self.gamma_attn
        y_mem_norm = self.mem_norm(y_mem) * self.gamma_mem

        gate = torch.sigmoid(y_attn_norm)
        combined = gate * y_mem_norm

        # Output projection with residual
        output = self.out_proj(combined) + x
        output = self.dropout(output)

        # Package state for next call - DETACH to avoid gradient graph retention
        # This is critical: states carry forward values but NOT gradients
        # (v4.2: memory learning happens through gate projections, not state gradients)
        new_state = {
            'M': M_new.detach(),
            'S': S_new.detach(),
            'P': P_new.detach(),
            'steps': steps_new,
        }

        # Combine telemetry
        telemetry = {
            **qk_telemetry,
            **mem_telemetry,
            'm3_alpha': self.m3_mixer.alpha.item(),
            'm3_warmup_complete': self.m3_mixer.warmup_complete.item(),
        }

        # Check kill switch from Q-K projection telemetry
        if qk_telemetry.get('KILL'):
            raise RuntimeError(qk_telemetry['kill_reason'])

        return output, new_state, telemetry

    def get_init_states(self, batch_size: int, device: torch.device) -> Dict[str, Union[torch.Tensor, int]]:
        """Get initial memory states for a new sequence."""
        return {
            'M': self.M_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device),
            'S': self.S_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device),
            'P': self.P_init.unsqueeze(0).expand(batch_size, -1, -1).clone().to(device),
            'steps': 0,
        }

    def get_regularization_loss(self) -> torch.Tensor:
        """
        Get combined regularization loss from all components.

        Add to training loss:
            loss = task_loss + 0.1 * block.get_regularization_loss()
        """
        qk_reg = self.qk_proj_layer.get_regularization_loss()
        m3_reg = self.m3_mixer.get_boundary_regularization_loss()
        return qk_reg + m3_reg

    def extra_repr(self) -> str:
        return (
            f'd_model={self.d_model}, '
            f'omega_context={self.omega_context}, '
            f'ns_iterations={self.ns_iterations}'
        )
