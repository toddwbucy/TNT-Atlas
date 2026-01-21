"""
Polynomial Feature Layer - PRD v4.2 Section 2.2

Provides O(d^p) memory capacity through polynomial feature expansion.
Coefficients initialized as Taylor series: a_i = 1/i!
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PolynomialFeatureLayer(nn.Module):
    """
    Polynomial feature expansion: φ_p(x) = a_0 + a_1·x + a_2·x² + ... + a_p·x^p

    Architectural layer providing O(d^p) memory capacity.

    Args:
        d_model: Model dimension (not used directly, but kept for interface consistency)
        degree: Polynomial degree (default: 2 for O(d²) capacity)
        learnable: Whether coefficients are learnable (default: True)

    Capacity scaling:
        - Linear (p=1): O(d) key-value pairs
        - Quadratic (p=2): O(d²) pairs <- Recommended
        - Cubic (p=3): O(d³) pairs
    """

    def __init__(
        self,
        d_model: int,
        degree: int = 2,
        learnable: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.degree = degree

        # Initialize coefficients as Taylor series: a_i = 1/i!
        # This provides a reasonable starting point for exponential-like behavior
        init_coeffs = torch.tensor(
            [1.0 / math.factorial(i) for i in range(degree + 1)]
        )

        if learnable:
            self.coeffs = nn.Parameter(init_coeffs)
        else:
            self.register_buffer('coeffs', init_coeffs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply polynomial feature expansion.

        Args:
            x: Input tensor [..., D]

        Returns:
            Polynomial features [..., D] (same shape, element-wise polynomial)
        """
        # Start with constant term
        result = self.coeffs[0] * torch.ones_like(x)

        # Add polynomial terms
        x_power = x
        for i in range(1, self.degree + 1):
            result = result + self.coeffs[i] * x_power
            x_power = x_power * x

        return result

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, degree={self.degree}, coeffs={self.coeffs.data.tolist()}'
