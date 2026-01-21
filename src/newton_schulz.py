"""
Newton-Schulz Iteration - PRD v4.2 Section 2.3

CRITICAL: Newton-Schulz is the INTERNAL memory update rule, NOT an external optimizer.

Find nearest semi-orthogonal matrix via Newton-Schulz iteration.
K=5 iterations standard (NS-5).

Internal update applied to gradient/momentum BEFORE memory update.
External optimizer (AdamW) still used for all other parameters.

Why Newton-Schulz?
- Approximates second-order information (curvature)
- Better local optima than gradient descent
- K iterations as "test-time compute" parameter
"""

import torch
from typing import Tuple, Dict, Any


def newton_schulz_5(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Find nearest semi-orthogonal matrix via Newton-Schulz iteration.
    K=5 iterations standard (NS-5).

    Internal update applied to gradient/momentum BEFORE memory update.
    External optimizer (AdamW) still used for all other parameters.

    The iteration converges to the nearest orthogonal matrix by
    repeatedly applying: X_{k+1} = X_k @ (3I - X_k^T @ X_k) / 2

    Args:
        G: Input gradient/momentum tensor [d_out, d_in] or batched
        eps: Small constant for numerical stability

    Returns:
        X: Semi-orthogonal matrix closest to G
    """
    # Handle both 2D and batched 3D tensors
    original_shape = G.shape
    if len(G.shape) == 2:
        G = G.unsqueeze(0)  # [1, d_out, d_in]

    B, d_out, d_in = G.shape

    # Normalize input
    G_norm = torch.norm(G, p='fro', dim=(-2, -1), keepdim=True)
    X = G / (G_norm + eps)

    # Run 5 Newton-Schulz iterations
    for _ in range(5):
        # A = X @ X^T
        A = torch.bmm(X, X.transpose(-2, -1))

        # Identity matrix
        I = torch.eye(d_out, device=X.device, dtype=X.dtype)
        I = I.unsqueeze(0).expand(B, -1, -1)

        # Update: X = X @ (3I - A) / 2
        X = torch.bmm(X, (3 * I - A)) / 2

    # Restore original shape
    if len(original_shape) == 2:
        result: torch.Tensor = X.squeeze(0)
        return result

    final_X: torch.Tensor = X
    return final_X


def newton_schulz_k(G: torch.Tensor, k: int, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration with configurable number of iterations.

    This version allows specifying the number of iterations,
    useful for adaptive K scheduling during training.

    Args:
        G: Input gradient/momentum tensor [d_out, d_in] or batched
        k: Number of iterations
        eps: Small constant for numerical stability

    Returns:
        X: Semi-orthogonal matrix closest to G
    """
    # Handle both 2D and batched 3D tensors
    original_shape = G.shape
    if len(G.shape) == 2:
        G = G.unsqueeze(0)

    B, d_out, d_in = G.shape

    # Normalize input
    G_norm = torch.norm(G, p='fro', dim=(-2, -1), keepdim=True)
    X = G / (G_norm + eps)

    # Run k Newton-Schulz iterations
    for _ in range(k):
        A = torch.bmm(X, X.transpose(-2, -1))
        I = torch.eye(d_out, device=X.device, dtype=X.dtype)
        I = I.unsqueeze(0).expand(B, -1, -1)
        X = torch.bmm(X, (3 * I - A)) / 2

    # Restore original shape
    if len(original_shape) == 2:
        result: torch.Tensor = X.squeeze(0)
        return result

    final_X: torch.Tensor = X
    return final_X


def get_newton_schulz_iterations(
    steps_since_reset: int,
    warmup_steps: int = 50,
    k_max: int = 3,
    k_min: int = 1,
) -> int:
    """
    Adaptive K with linear decay for smooth transition.

    With warm start (M3 mixing), we don't need as aggressive compensation,
    but we still allow a brief stabilization period after shard boundaries.

    Steps 0-warmup: K decays linearly from k_max to k_min
    Steps warmup+: K=k_min (normal operation)

    The intuition:
    - After a shard boundary, states are mixed (not fully warmed up)
    - More iterations help refine the approximation early
    - Once states stabilize, fewer iterations needed

    Args:
        steps_since_reset: Number of steps since last shard boundary
        warmup_steps: Steps to decay from k_max to k_min
        k_max: Maximum iterations (used at shard boundary)
        k_min: Minimum iterations (used after warmup)

    Returns:
        Number of Newton-Schulz iterations to use
    """
    if steps_since_reset < warmup_steps:
        # Linear decay: k_max → k_min over warmup_steps
        progress = steps_since_reset / warmup_steps
        k = int(k_max - (k_max - k_min) * progress)
        return max(k_min, k)
    return k_min


def newton_schulz_with_telemetry(
    G: torch.Tensor,
    k: int = 5,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Newton-Schulz iteration with telemetry for monitoring convergence.

    Returns additional metrics useful for debugging and monitoring
    the training process.

    Args:
        G: Input gradient/momentum tensor
        k: Number of iterations
        eps: Small constant for numerical stability

    Returns:
        X: Semi-orthogonal matrix closest to G
        telemetry: Dict with convergence metrics
    """
    original_shape = G.shape
    if len(G.shape) == 2:
        G = G.unsqueeze(0)

    B, d_out, d_in = G.shape

    # Normalize input
    G_norm = torch.norm(G, p='fro', dim=(-2, -1), keepdim=True)
    X = G / (G_norm + eps)

    # Track convergence
    convergence_history = []
    initial_orthogonality_error = None

    for i in range(k):
        A = torch.bmm(X, X.transpose(-2, -1))
        I = torch.eye(d_out, device=X.device, dtype=X.dtype)
        I = I.unsqueeze(0).expand(B, -1, -1)

        # Orthogonality error: ||X X^T - I||_F
        orth_error = torch.norm(A - I, p='fro', dim=(-2, -1)).mean().item()
        convergence_history.append(orth_error)

        if i == 0:
            initial_orthogonality_error = orth_error

        X = torch.bmm(X, (3 * I - A)) / 2

    # Final orthogonality error
    final_A = torch.bmm(X, X.transpose(-2, -1))
    final_I = torch.eye(d_out, device=X.device, dtype=X.dtype).unsqueeze(0).expand(B, -1, -1)
    final_orth_error = torch.norm(final_A - final_I, p='fro', dim=(-2, -1)).mean().item()

    # Restore original shape
    if len(original_shape) == 2:
        X = X.squeeze(0)

    telemetry = {
        'iterations': k,
        'input_frobenius_norm': G_norm.mean().item(),
        'initial_orthogonality_error': initial_orthogonality_error,
        'final_orthogonality_error': final_orth_error,
        'convergence_history': convergence_history,
        'improvement_ratio': initial_orthogonality_error / (final_orth_error + eps) if initial_orthogonality_error else 1.0,
    }

    return X, telemetry
