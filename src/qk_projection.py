"""
Q-K Projection Layer - PRD v4.2 Section 6

Aligns query to key subspace for retrieval.
MUST happen BEFORE polynomial expansion to keep P_t at d×d.

v4.2 Safety Features:
- Tanh gate clamps output before polynomial
- Learnable input gain to prevent saturation
- Saturation telemetry and kill switch
- Circuit breaker for runaway gain
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


class QKProjectionLayer(nn.Module):
    """
    Alignment layer: projects query onto key subspace.

    The projection matrix P_t = Σ k_τ k_τ^T captures the subspace of observed keys.
    Query alignment: q̃ = (P_t / ||P_t||) @ q

    v4.2 Pipeline:
    1. P_t = P_{t-1} + k_t k_t^T
    2. P_normalized = P_t / ||P_t||_F   <- NORMALIZE INPUT
    3. q_aligned = P_normalized @ q_t
    4. q_gated = gain * q_aligned        <- LEARNABLE GAIN
    5. q_safe = tanh(q_gated) * scale    <- SAFETY GATE

    Args:
        d_model: Model dimension
        gain_circuit_breaker: Max gain magnitude before freezing (default: 2.0)
        saturation_kill_threshold: Fraction of saturated activations to trigger kill (default: 0.20)
        saturation_kill_patience: Steps of sustained saturation before kill (default: 100)
    """

    # Type annotations for register_buffer attributes
    saturation_ema: torch.Tensor
    steps_observed: torch.Tensor
    gain_frozen: torch.Tensor
    gain_frozen_at_step: torch.Tensor
    consecutive_saturated_steps: torch.Tensor
    last_input_magnitude: torch.Tensor

    def __init__(
        self,
        d_model: int,
        gain_circuit_breaker: float = 6.0,  # Run #3: raised from 2.0 to allow gain=4.0
        saturation_kill_threshold: float = 0.20,
        saturation_kill_patience: int = 100,
    ):
        super().__init__()
        self.d_model = d_model

        # v4.2 -> v4.3: Learnable GAIN parameter (inside tanh, not outside)
        # Run #1 FAILURE: gain=0.5 caused DEAD gates (tanh linear region)
        # Run #2 FAILURE: gain=2.0 still linear - activations plateaued at 0.05
        #   Model learned to BYPASS gates instead of using them
        # Run #3 FIX: gain=4.0 forces nonlinear operation
        #   With activation ~0.05, gain=4.0 gives tanh input ~0.2 → tanh(0.2)≈0.197
        #   Need activations to grow OR gain high enough to force nonlinearity
        self.input_gain = nn.Parameter(torch.ones(d_model) * 4.0)

        # Learnable scale factor - "volume knob" for output magnitude
        self.output_scale = nn.Parameter(torch.ones(d_model))

        # v4.2: Telemetry for saturation monitoring
        self.register_buffer('saturation_ema', torch.tensor(0.0))
        self.register_buffer('steps_observed', torch.tensor(0))

        # v4.2 CIRCUIT BREAKER: Freeze gain if magnitude exceeds threshold
        self.gain_circuit_breaker = gain_circuit_breaker
        self.register_buffer('gain_frozen', torch.tensor(False))
        self.register_buffer('gain_frozen_at_step', torch.tensor(-1))

        # v4.2 SATURATION KILL SWITCH
        self.saturation_kill_threshold = saturation_kill_threshold
        self.saturation_kill_patience = saturation_kill_patience
        self.register_buffer('consecutive_saturated_steps', torch.tensor(0))

        # v4.3: Magnitude tracking for activation kill switch (PRD Fix 3)
        # Used to detect dead gates during training
        self.register_buffer('last_input_magnitude', torch.tensor(0.0))

    def forward(
        self,
        q: torch.Tensor,           # [B, L, D] - raw query
        k: torch.Tensor,           # [B, L, D] - key (for updating P)
        P_prev: torch.Tensor,      # [B, D, D] - projection matrix state (pre-processed)
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Q-K Projection with saturation-aware safety gate.

        NOTE: P_prev should already be processed through reset_atlas_states()
        which applies M3 mixing at shard boundaries. This layer does NOT
        handle shard boundary resets - that's done at a higher level.

        Args:
            q: Raw query [B, L, D]
            k: Key for updating P [B, L, D]
            P_prev: Projection matrix state [B, D, D]

        Returns:
            q_aligned: Aligned and safety-gated query [B, L, D]
            P_new: Updated projection matrix [B, D, D]
            telemetry: Dict with saturation metrics, warnings, potential KILL flag
        """
        B, L, D = k.shape

        # P_prev comes pre-processed from reset_atlas_states (M3 mixed at boundaries)
        P = P_prev

        outputs = []
        saturation_samples = []

        for t in range(L):
            k_t = k[:, t, :]  # [B, D]
            q_t = q[:, t, :]  # [B, D]

            # Update projection matrix: P_t = P_{t-1} + k_t k_t^T
            P = P + torch.einsum('bi,bj->bij', k_t, k_t)

            # v4.2 FIX: Normalize P_t BEFORE matrix multiplication
            # Keeps input scale to tanh bounded regardless of sequence length
            P_norm = P / (torch.norm(P, p='fro', dim=(-2, -1), keepdim=True) + 1e-7)

            # Project query (output is bounded because P_norm has unit energy)
            q_aligned = torch.einsum('bij,bj->bi', P_norm, q_t)

            # v4.2 FIX: Learnable gain INSIDE tanh + CIRCUIT BREAKER
            gain_magnitude = self.input_gain.abs().max().item()
            if not self.gain_frozen and gain_magnitude > self.gain_circuit_breaker:
                self.gain_frozen.fill_(True)
                self.gain_frozen_at_step.fill_(self.steps_observed.item())
                # Clamp gain to safe range when freezing
                with torch.no_grad():
                    self.input_gain.data.clamp_(-self.gain_circuit_breaker, self.gain_circuit_breaker)

            # Use clamped gain if circuit breaker tripped (gradient won't flow)
            if self.gain_frozen:
                effective_gain = self.input_gain.detach()  # No gradient
            else:
                effective_gain = self.input_gain

            q_gated = effective_gain * q_aligned

            # v4.3: Track input magnitude for activation kill switch
            # This is checked by ActivationKillSwitch to detect dead gates
            self.last_input_magnitude = q_gated.abs().mean().detach()

            # Track saturation for telemetry
            # tanh'(x) = 1 - tanh²(x), gradient drops rapidly:
            #   |x| = 1.0: tanh'≈0.42 (moderate gradient)
            #   |x| = 1.5: tanh'≈0.18 (weak gradient)
            #   |x| = 2.0: tanh'≈0.07 (nearly zero - "coma")
            # We use 1.5 as threshold: catches inputs where gradient < 20%
            saturation_level = (q_gated.abs() > 1.5).float().mean()
            saturation_samples.append(saturation_level)

            # SAFETY GATE: tanh squashes to [-1, 1]
            q_safe = torch.tanh(q_gated) * self.output_scale

            outputs.append(q_safe)

        q_aligned_all = torch.stack(outputs, dim=1)  # [B, L, D]

        # v4.2: Update saturation telemetry
        current_saturation = torch.tensor(0.0)
        if saturation_samples:
            current_saturation = torch.stack(saturation_samples).mean()
            self.saturation_ema = 0.99 * self.saturation_ema + 0.01 * current_saturation
            self.steps_observed += 1

            # v4.2 KILL SWITCH: Track consecutive saturated steps
            if current_saturation > self.saturation_kill_threshold:
                self.consecutive_saturated_steps += 1
            else:
                self.consecutive_saturated_steps.zero_()  # Reset counter

        telemetry: Dict[str, Any] = {
            'saturation_ema': self.saturation_ema.item(),
            'saturation_current': current_saturation.item() if saturation_samples else 0,
            'input_gain_mean': self.input_gain.mean().item(),
            'input_gain_max': self.input_gain.abs().max().item(),
            'input_magnitude': self.last_input_magnitude.item(),  # v4.3: For kill switch
            'steps': self.steps_observed.item(),
            'gain_frozen': self.gain_frozen.item(),
            'consecutive_saturated_steps': self.consecutive_saturated_steps.item(),
        }

        # v4.2 KILL SWITCH: Check if we should terminate
        if self.consecutive_saturated_steps >= self.saturation_kill_patience:
            telemetry['KILL'] = True
            telemetry['kill_reason'] = (
                f'SATURATION KILL SWITCH TRIGGERED! '
                f'{self.saturation_ema:.1%} saturation for {self.consecutive_saturated_steps.item()} consecutive steps. '
                f'Model is in coma - terminate run immediately.'
            )

        # v4.2: Circuit breaker warnings
        if self.gain_frozen:
            telemetry['warning'] = (
                f'CIRCUIT BREAKER TRIPPED at step {self.gain_frozen_at_step.item()}! '
                f'Gain frozen at magnitude {self.gain_circuit_breaker}'
            )
        elif self.saturation_ema > 0.5:
            telemetry['warning'] = f'High tanh saturation: {self.saturation_ema:.2%} of activations'

        return q_aligned_all, P, telemetry

    def get_regularization_loss(self, saturation_threshold: float = 0.05) -> torch.Tensor:
        """
        v4.2 GAIN REGULARIZATION: Penalize saturation to prevent coma.

        Add this to training loss:
            loss = task_loss + 0.01 * qk_layer.get_regularization_loss()

        The regularization:
        1. Penalizes saturation > threshold (default 5%)
        2. Penalizes large gain magnitudes (to prevent runaway)
        """
        # Penalty 1: Saturation above threshold
        saturation_penalty = torch.relu(self.saturation_ema - saturation_threshold)

        # Penalty 2: Large gain magnitude (prevents runaway toward circuit breaker)
        gain_penalty = torch.relu(self.input_gain.abs() - 1.0).mean()

        return saturation_penalty + 0.1 * gain_penalty

    def reset_telemetry(self) -> None:
        """Reset telemetry counters (useful for testing)."""
        self.saturation_ema.zero_()
        self.steps_observed.zero_()
        self.consecutive_saturated_steps.zero_()
        self.gain_frozen.fill_(False)
        self.gain_frozen_at_step.fill_(-1)
        self.last_input_magnitude.zero_()

    def extra_repr(self) -> str:
        return (
            f'd_model={self.d_model}, '
            f'gain_circuit_breaker={self.gain_circuit_breaker}, '
            f'saturation_kill_threshold={self.saturation_kill_threshold}'
        )
