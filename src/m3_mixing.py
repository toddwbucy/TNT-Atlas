"""
M3 Mixing State - PRD v4.2 Section 5.4

Continuous state inheritance with gradual forgetting.
v4 Core Principle: "Don't use binary switches - use continuous dynamics."

S_new = α · S_prev.detach() + (1 - α) · S_init

- If α=1.0: Equivalent to raw detach (inherit everything)
- If α=0.0: Equivalent to hard reset (forget everything)
- If α∈(0,1): Model has "fading memory" - recent curvature matters more
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


class M3MixingState(nn.Module):
    """
    v4.2 M3 Mixing: Continuous blending of old and new state.

    Prevents "poisoned compass" from raw detach while preserving
    the benefits of warm starting.

    CRITICAL v4.2 FIX: SCHEDULED alpha initialization.

    At START of training: History (S_prev, P_prev) is GARBAGE
    Trusting garbage history (high alpha) = poisoned compass
    So alpha should START LOW (~0.1), RAMP UP as model stabilizes

    After warmup, the learnable alpha_logit takes over completely.

    Args:
        d_model: Model dimension
        alpha_target: Target alpha after warmup (default: 0.5)
        alpha_start: Initial alpha during warmup (default: 0.1)
        alpha_warmup_steps: Steps to ramp from start to target (default: 500)
    """

    # Type annotations for register_buffer attributes
    alpha_ema: torch.Tensor
    steps_observed: torch.Tensor
    warmup_complete: torch.Tensor

    def __init__(
        self,
        d_model: int,
        alpha_target: float = 0.5,
        alpha_start: float = 0.1,
        alpha_warmup_steps: int = 500,
    ):
        super().__init__()
        self.d_model = d_model

        # v4.2 SCHEDULED: Learnable alpha initialized to TARGET value
        # The schedule overrides this during warmup, then releases control
        self.alpha_logit = nn.Parameter(torch.tensor(self._inv_sigmoid(alpha_target)))

        # v4.2 SCHEDULE PARAMETERS
        self.alpha_start = alpha_start
        self.alpha_target = alpha_target
        self.warmup_steps = alpha_warmup_steps

        # v4.2: Track alpha for telemetry guardrails
        self.register_buffer('alpha_ema', torch.tensor(alpha_start))
        self.register_buffer('steps_observed', torch.tensor(0))
        self.register_buffer('warmup_complete', torch.tensor(False))

    def _inv_sigmoid(self, x: float) -> float:
        """Inverse sigmoid for initialization."""
        x = max(1e-7, min(1 - 1e-7, x))  # Clamp to avoid inf
        return math.log(x / (1 - x))

    @property
    def alpha(self) -> torch.Tensor:
        """
        v4.2: Mixing coefficient in [0, 1] with WARMUP SCHEDULE.

        During warmup: Linear interpolation from alpha_start to alpha_target
        After warmup: Learnable alpha via sigmoid(alpha_logit)

        This ensures we don't trust garbage history early in training.
        """
        steps = self.steps_observed.item()

        if steps < self.warmup_steps:
            # WARMUP: Linear ramp from start to target
            progress = steps / self.warmup_steps
            scheduled_alpha = self.alpha_start + progress * (self.alpha_target - self.alpha_start)
            return torch.tensor(scheduled_alpha, device=self.alpha_logit.device, dtype=self.alpha_logit.dtype)
        else:
            # POST-WARMUP: Learnable alpha takes over
            if not self.warmup_complete:
                self.warmup_complete.fill_(True)
            return torch.sigmoid(self.alpha_logit)

    def update_telemetry(self, force_decay: bool = False) -> Dict[str, Any]:
        """
        v4.2 Telemetry: Monitor alpha schedule and post-warmup behavior.

        During warmup: Just track progress (alpha is deterministic)
        After warmup: Monitor for runaway alpha (model ignoring schedule intent)

        Args:
            force_decay: If True, apply guardrail interventions

        Returns:
            Telemetry dict with alpha values, warnings, interventions
        """
        current_alpha = self.alpha.item()
        self.steps_observed += 1

        # EMA tracking for smooth monitoring
        self.alpha_ema = 0.99 * self.alpha_ema + 0.01 * current_alpha

        telemetry: Dict[str, Any] = {
            'alpha_current': current_alpha,
            'alpha_ema': self.alpha_ema.item(),
            'steps': self.steps_observed.item(),
            'warmup_complete': self.warmup_complete.item(),
            'warmup_progress': min(1.0, self.steps_observed.item() / self.warmup_steps),
        }

        # GUARDRAIL: Only applies AFTER warmup (during warmup, alpha is scheduled)
        if self.warmup_complete:
            DANGER_HIGH = 0.95  # Reverting to poisoned compass
            DANGER_LOW = 0.05   # Reverting to cold start
            POST_WARMUP_STEPS = self.steps_observed.item() - self.warmup_steps

            if POST_WARMUP_STEPS < 1000:  # Monitor first 1000 steps after warmup
                if self.alpha_ema > DANGER_HIGH:
                    if force_decay:
                        with torch.no_grad():
                            self.alpha_logit.data = self.alpha_logit.data * 0.9
                        telemetry['intervention'] = 'forced_decay_high'
                        telemetry['warning'] = f'Post-warmup alpha too high: {self.alpha_ema:.3f}'
                    else:
                        telemetry['warning'] = f'Alpha drifting high: {self.alpha_ema:.3f}'

                elif self.alpha_ema < DANGER_LOW:
                    if force_decay:
                        with torch.no_grad():
                            self.alpha_logit.data = self.alpha_logit.data * 0.9 + 0.1
                        telemetry['intervention'] = 'forced_decay_low'
                        telemetry['warning'] = f'Post-warmup alpha too low: {self.alpha_ema:.3f}'
                    else:
                        telemetry['warning'] = f'Alpha drifting low: {self.alpha_ema:.3f}'

        return telemetry

    def get_boundary_regularization_loss(
        self,
        low_boundary: float = 0.1,
        high_boundary: float = 0.9,
    ) -> torch.Tensor:
        """
        v4.2 BOUNDARY REGULARIZATION (Committee Critique)

        "If alpha tries to go above 0.9 or below 0.1 in that first phase
        of training, the loss shoots up. You're just discouraging it from
        making those extreme choices until it's smart enough to know better."

        Add this to training loss:
            loss = task_loss + 0.1 * m3_mixer.get_boundary_regularization_loss()

        Only active AFTER warmup (during warmup, alpha is scheduled).
        """
        if not self.warmup_complete:
            return torch.tensor(0.0, device=self.alpha_logit.device)

        alpha = torch.sigmoid(self.alpha_logit)

        # Penalty for going too high (toward poisoned compass)
        high_penalty = torch.relu(alpha - high_boundary)

        # Penalty for going too low (toward cold start)
        low_penalty = torch.relu(low_boundary - alpha)

        # Quadratic penalty (steeper as you approach boundaries)
        return (high_penalty ** 2 + low_penalty ** 2) * 10.0

    def reset_telemetry(self) -> None:
        """Reset telemetry counters (useful for testing)."""
        self.alpha_ema.fill_(self.alpha_start)
        self.steps_observed.zero_()
        self.warmup_complete.fill_(False)

    def extra_repr(self) -> str:
        return (
            f'd_model={self.d_model}, '
            f'alpha_start={self.alpha_start}, '
            f'alpha_target={self.alpha_target}, '
            f'warmup_steps={self.warmup_steps}'
        )


def reset_atlas_states(
    shard_boundary: bool,
    M_init: torch.Tensor,
    S_init: torch.Tensor,
    P_init: torch.Tensor,
    M_prev: torch.Tensor,
    S_prev: torch.Tensor,
    P_prev: torch.Tensor,
    m3_mixer: M3MixingState,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    v4.2 M3 Mixing: Continuous state inheritance with gradual forgetting.

    - M_t: Reset to learnable M_init (memory content is shard-specific)
    - S_t: M3 MIX between inherited and fresh (curvature with forgetting)
    - P_t: M3 MIX between inherited and fresh (subspace with forgetting)

    The M3 mixing is critical:
    - Preserves accumulated curvature/subspace information (like v3)
    - BUT allows gradual forgetting of ancient/poisoned estimates
    - Learnable α lets model optimize its own forgetting rate
    - detach() still blocks gradients (maintains TNT parallelism)

    Args:
        shard_boundary: True if at shard boundary
        M_init, S_init, P_init: Initial state tensors
        M_prev, S_prev, P_prev: Previous state tensors
        m3_mixer: M3MixingState module for alpha

    Returns:
        M_new, S_new, P_new: Updated states
    """
    if shard_boundary:
        alpha = m3_mixer.alpha  # Learnable mixing coefficient

        M_new = M_init  # Memory always resets (shard-specific content)

        # v4.2 M3 Mixing: Blend old and new instead of raw inheritance
        S_new = alpha * S_prev.detach() + (1 - alpha) * S_init
        P_new = alpha * P_prev.detach() + (1 - alpha) * P_init

        return M_new, S_new, P_new

    return M_prev, S_prev, P_prev
