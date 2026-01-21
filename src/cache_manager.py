"""
Hybrid Cache Manager - PRD v4.2 Section 5.5

v4.2 Cache manager with STRICT invalidation protocol.

v4.2 CRITICAL FIX: "Slow does not mean static"

Even with lr=1e-6, weights ARE changing between chunks.
Cached state computed with old weights becomes stale.
Using stale state = calculating gradients on a map that has shifted.

The v4 approach of "cache global because it's slow" was too loose.
v4.2 enforces ONE-STEP cache lifetime with version assertions.

Policy: "Cache slow (global), recompute fast (local)"
"""

import torch
from typing import Dict, Any, Optional, Tuple


class HybridCacheManager:
    """
    v4.2 Cache manager with STRICT invalidation protocol.

    v4.2 LAZY RECOMPUTE OPTION:
    On small models with fast GPUs (A6000 + 50M params), strict
    invalidation may cause unnecessary recomputes that throttle throughput.
    Optional lazy_threshold allows micro-staleness for speedup.
    Default: OFF (safe). Enable only after verifying no loss degradation.

    Args:
        global_lr: Learning rate for global memory (slow)
        local_lr: Learning rate for local memory (fast)
        lazy_threshold: Number of steps to allow cache staleness (0 = strict)
        gradient_magnitude_threshold: Threshold for weight drift invalidation
    """

    def __init__(
        self,
        global_lr: float = 1e-6,
        local_lr: float = 1e-4,
        lazy_threshold: int = 0,  # 0 = strict (default), >0 = allow N steps staleness
        gradient_magnitude_threshold: float = 1e-8,
    ):
        self.global_lr = global_lr
        self.local_lr = local_lr
        self.lazy_threshold = lazy_threshold

        # v4.2: Track weight versions for staleness detection
        self.global_weight_version = 0
        self.local_weight_version = 0
        self.cached_global_version = -1  # -1 = no cache
        self.cached_local_version = -1

        # v4.2: Telemetry for lazy mode
        self.lazy_recompute_count = 0
        self.lazy_hit_count = 0

        # v4.2 GRADIENT MAGNITUDE TRACKING (Committee Critique)
        # "Trust the gradient magnitude, not the learning rate"
        # If actual weight change > epsilon, invalidate regardless of LR
        self.gradient_magnitude_threshold = gradient_magnitude_threshold
        self.last_global_weight_norm: Optional[float] = None
        self.cumulative_weight_drift = 0.0

        # Cached states
        self._cached_global_state: Optional[Dict[str, Any]] = None
        self._cached_local_state: Optional[Dict[str, Any]] = None

    def on_optimizer_step(self, global_weights: Optional[torch.Tensor] = None) -> None:
        """
        v4.2: MUST be called after every optimizer.step().

        This increments weight versions, invalidating ALL caches.

        OPTIONAL: Pass global weights to enable gradient-magnitude tracking.
        If weights moved more than threshold since cache was created,
        we invalidate even in lazy mode.
        """
        self.global_weight_version += 1
        self.local_weight_version += 1

        # v4.2: Track actual weight magnitude change (Committee Critique)
        if global_weights is not None:
            current_norm = global_weights.detach().norm().item()
            if self.last_global_weight_norm is not None:
                drift = abs(current_norm - self.last_global_weight_norm)
                self.cumulative_weight_drift += drift
            self.last_global_weight_norm = current_norm

    def cache_global_state(self, state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Cache global state with version tag AND drift checkpoint.

        Args:
            state: Dictionary of state tensors to cache

        Returns:
            Cached state with version metadata
        """
        self.cached_global_version = self.global_weight_version
        # v4.2: Also record cumulative drift at cache time
        cached_drift = self.cumulative_weight_drift
        self._cached_global_state = {
            **{k: v.detach().clone() for k, v in state.items()},
            '_cache_version': self.global_weight_version,
            '_cache_type': 'global',
            '_cache_drift': cached_drift,  # For magnitude-based invalidation
        }
        return self._cached_global_state

    def get_cached_global_state(self) -> Dict[str, Any]:
        """
        v4.2: Retrieve cached state WITH staleness assertion.

        WILL CRASH if cache is stale (unless lazy_threshold > 0).
        Silent training on stale state is worse than a crash.

        LAZY MODE (lazy_threshold > 0):
        Allows micro-staleness for throughput on fast GPUs.
        Use only after validating no loss degradation with boundary stress test.

        v4.2 GRADIENT MAGNITUDE CHECK (Committee Critique):
        "Trust the gradient magnitude, not the learning rate."
        Even in lazy mode, if actual weight drift > epsilon, invalidate.

        Returns:
            Cached global state

        Raises:
            RuntimeError: If cache is stale
        """
        if self._cached_global_state is None:
            raise RuntimeError("No cached global state available!")

        cached = self._cached_global_state
        cached_version = cached.get('_cache_version', -1)
        current_version = self.global_weight_version
        staleness = current_version - cached_version

        # v4.2 GRADIENT MAGNITUDE CHECK (Committee Critique)
        # Even if version-based check passes, check actual weight drift
        cached_drift = cached.get('_cache_drift', 0.0)
        actual_drift = self.cumulative_weight_drift - cached_drift

        if actual_drift > self.gradient_magnitude_threshold:
            raise RuntimeError(
                f"STALE CACHE - GRADIENT MAGNITUDE! "
                f"Weight drift since cache: {actual_drift:.2e} "
                f"(threshold: {self.gradient_magnitude_threshold:.0e}). "
                f"Weights moved more than expected. Recompute required."
            )

        # v4.2 LAZY MODE: Accept micro-staleness if within threshold
        if self.lazy_threshold > 0:
            if staleness <= self.lazy_threshold:
                self.lazy_hit_count += 1
                return cached  # Accept slightly stale cache for speed
            else:
                self.lazy_recompute_count += 1
                raise RuntimeError(
                    f"STALE CACHE EXCEEDED LAZY THRESHOLD! "
                    f"Staleness: {staleness} steps (threshold: {self.lazy_threshold}). "
                    f"Need to recompute."
                )

        # v4.2 STRICT MODE (default): Crash immediately if mismatch
        if cached_version != current_version:
            raise RuntimeError(
                f"STALE CACHE DETECTED! "
                f"Cache version: {cached_version}, "
                f"Weight version: {current_version}. "
                f"Cache was not invalidated after optimizer.step(). "
                f"This would cause incorrect gradient calculations."
            )

        return cached

    def can_cache(self, component: str, steps_since_cache: int = 0) -> bool:
        """
        v4.2: Simplified - cache is valid ONLY for current optimizer step.
        (Or within lazy_threshold if enabled.)

        The v4 heuristic of "lr * steps < threshold" was too permissive.
        In deep learning, microscopic errors compound catastrophically.

        Args:
            component: 'global' or 'local'
            steps_since_cache: Steps since cache was created

        Returns:
            True if cache is valid
        """
        # LAZY MODE: Allow limited staleness
        max_staleness = self.lazy_threshold if self.lazy_threshold > 0 else 0

        if steps_since_cache > max_staleness:
            return False

        if component == 'global':
            staleness = self.global_weight_version - self.cached_global_version
            return staleness <= max_staleness
        else:
            # Local state should NEVER be cached during training
            return False

    def get_effective_cache_lifetime(self, component: str) -> int:
        """
        v4.2: Cache lifetime is exactly ONE forward+backward pass.
        (Or lazy_threshold + 1 steps if lazy mode enabled.)

        Not "approximately 100 steps" - exactly ONE macro step (strict),
        or lazy_threshold + 1 if optimizing for throughput.

        Args:
            component: 'global' or 'local'

        Returns:
            Number of steps cache is valid
        """
        if component == 'global':
            return 1 + self.lazy_threshold  # Strict: 1, Lazy: 1 + threshold
        else:
            return 0  # Local: NEVER cache during training

    def get_lazy_stats(self) -> Dict[str, Any]:
        """v4.2: Return lazy mode telemetry for monitoring."""
        total = self.lazy_hit_count + self.lazy_recompute_count
        return {
            'lazy_threshold': self.lazy_threshold,
            'lazy_hits': self.lazy_hit_count,
            'lazy_recomputes': self.lazy_recompute_count,
            'hit_rate': self.lazy_hit_count / total if total > 0 else 0,
            'global_weight_version': self.global_weight_version,
            'local_weight_version': self.local_weight_version,
            'cumulative_weight_drift': self.cumulative_weight_drift,
        }

    def invalidate_all(self) -> None:
        """Explicitly invalidate all caches."""
        self._cached_global_state = None
        self._cached_local_state = None
        self.cached_global_version = -1
        self.cached_local_version = -1

    def reset_telemetry(self) -> None:
        """Reset telemetry counters (useful for testing)."""
        self.lazy_recompute_count = 0
        self.lazy_hit_count = 0
        self.cumulative_weight_drift = 0.0
        self.last_global_weight_norm = None


def get_omega_context_v4(
    chunk_size: int,
    warmup_buffer_size: int,
    prev_global_state: Dict[str, torch.Tensor],
    prev_local_state: Optional[Dict[str, torch.Tensor]],
    prev_chunk_tokens: torch.Tensor,
    training: bool = True
) -> Tuple[Optional[torch.Tensor], Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    v4 Hybrid Caching: Cache slow global state, recompute fast local state.

    During training:
    - Global state: Cache is valid (weights move slowly)
    - Local state: Must recompute (weights move quickly)

    During inference:
    - Both can be cached (no weight updates)

    Args:
        chunk_size: Size of current chunk
        warmup_buffer_size: Size of warmup buffer (c tokens)
        prev_global_state: Previous global state dict
        prev_local_state: Previous local state dict (may be None)
        prev_chunk_tokens: Tokens from previous chunk
        training: Whether in training mode

    Returns:
        warmup_tokens: Tokens for warmup buffer (or None)
        global_cached: Cached global state
        local_cached: Local state (None in training - signal to recompute)
    """
    if not training:
        # Inference: Cache everything (no weight changes)
        return None, {
            'global': prev_global_state,
            'local': prev_local_state,
        }, prev_local_state

    # Training mode: Hybrid caching
    global_cached = {
        # Global state CAN be cached (slow-moving weights)
        'M_global': prev_global_state['M_global'].detach(),
        'S_global': prev_global_state['S_global'].detach(),
        'P_global': prev_global_state['P_global'].detach(),
    }

    # Local state MUST be recomputed from tokens
    warmup_tokens = None
    if chunk_size >= warmup_buffer_size:
        # Large chunks: Use warm-up buffer for local state
        warmup_tokens = prev_chunk_tokens[-warmup_buffer_size:]

    # Signal to recompute local state
    local_cached = None

    return warmup_tokens, global_cached, local_cached
