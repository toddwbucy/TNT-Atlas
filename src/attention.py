"""
Sliding Window Attention - PRD v4.2 Section 3

Standard sliding window attention from Titans MAG.
Includes:
- Persistent memory tokens (P)
- Full attention to P, sliding window for sequence
- Causal convolutions for Q, K, V
- L2 normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention with persistent memory tokens.

    The attention mask allows:
    - Full attention to persistent memory tokens (P)
    - Sliding window attention for sequence tokens

    Mask structure:
                   P_1  P_2  x_1  x_2  x_3  x_4  x_5
            P_1  [  1    0    0    0    0    0    0  ]
            P_2  [  1    1    0    0    0    0    0  ]
            x_1  [  1    1    1    0    0    0    0  ]  ← Full attn to P
            x_2  [  1    1    1    1    0    0    0  ]
            x_3  [  1    1    1    1    1    0    0  ]  ← Sliding window
            x_4  [  1    1    0    1    1    1    0  ]
            x_5  [  1    1    0    0    1    1    1  ]

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_size: Sliding window size
        n_persistent: Number of persistent memory tokens
        dropout: Dropout probability
        use_causal_conv: Whether to use causal convolutions on Q, K, V
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 128,
        n_persistent: int = 4,
        dropout: float = 0.0,
        use_causal_conv: bool = True,
        conv_kernel_size: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.n_persistent = n_persistent
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Persistent memory tokens (learnable)
        self.persistent_memory = nn.Parameter(
            torch.randn(1, n_persistent, d_model) * 0.02
        )

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Optional causal convolutions
        self.use_causal_conv = use_causal_conv
        if use_causal_conv:
            # Causal conv: pad left to ensure causality
            self.q_conv = nn.Conv1d(
                d_model, d_model, kernel_size=conv_kernel_size,
                padding=conv_kernel_size - 1, groups=d_model
            )
            self.k_conv = nn.Conv1d(
                d_model, d_model, kernel_size=conv_kernel_size,
                padding=conv_kernel_size - 1, groups=d_model
            )
            self.v_conv = nn.Conv1d(
                d_model, d_model, kernel_size=conv_kernel_size,
                padding=conv_kernel_size - 1, groups=d_model
            )

        self.dropout = nn.Dropout(dropout)

    def _create_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Create sliding window attention mask with persistent memory.

        Returns:
            Mask tensor [1, 1, total_len, total_len] where total_len = n_persistent + seq_len
        """
        n_p = self.n_persistent
        total_len = n_p + seq_len

        # Initialize with -inf (no attention)
        mask = torch.full((total_len, total_len), float('-inf'), device=device, dtype=dtype)

        # Persistent tokens attend to themselves causally
        for i in range(n_p):
            mask[i, :i+1] = 0  # Can attend to previous P tokens

        # Sequence tokens:
        for i in range(seq_len):
            pos = n_p + i  # Position in full sequence

            # Full attention to ALL persistent tokens
            mask[pos, :n_p] = 0

            # Sliding window attention for sequence positions
            window_start = max(n_p, pos - self.window_size + 1)
            mask[pos, window_start:pos+1] = 0  # Can attend within window

        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, total_len, total_len]

    def _causal_conv(self, x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        """Apply causal convolution with SiLU activation."""
        # x: [B, L, D] -> [B, D, L] for conv1d
        x = x.transpose(1, 2)
        x = conv(x)
        # Remove padding to ensure causality: [B, D, L + padding] -> [B, D, L]
        x = x[:, :, :x.size(2) - (conv.kernel_size[0] - 1)]
        x = F.silu(x)
        # Back to [B, L, D]
        return x.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with sliding window attention.

        Args:
            x: Input tensor [B, L, D]
            return_attn_weights: Whether to return attention weights

        Returns:
            output: Attention output [B, L, D]
            attn_weights: Optional attention weights [B, H, L+P, L+P]
        """
        B, L, D = x.shape

        # Expand persistent memory for batch
        P = self.persistent_memory.expand(B, -1, -1)  # [B, n_p, D]

        # Prepend persistent memory to input
        x_with_P = torch.cat([P, x], dim=1)  # [B, L + n_p, D]

        # Project Q, K, V
        Q = self.q_proj(x_with_P)  # [B, L + n_p, D]
        K = self.k_proj(x_with_P)
        V = self.v_proj(x_with_P)

        # Optional causal convolutions with SiLU
        if self.use_causal_conv:
            Q = self._causal_conv(Q, self.q_conv)
            K = self._causal_conv(K, self.k_conv)
            V = self._causal_conv(V, self.v_conv)

        # L2 normalize Q and K
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)

        # Reshape for multi-head attention
        total_len = L + self.n_persistent
        Q = Q.view(B, total_len, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L+P, head_dim]
        K = K.view(B, total_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, total_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, L+P, L+P]

        # Apply sliding window mask
        mask = self._create_sliding_window_mask(L, x.device, x.dtype)
        attn_scores = attn_scores + mask

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, H, L+P, head_dim]

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, total_len, D)

        # Output projection
        attn_output = self.out_proj(attn_output)

        # Remove persistent memory positions from output
        output = attn_output[:, self.n_persistent:, :]  # [B, L, D]

        if return_attn_weights:
            return output, attn_weights
        return output, None

    def extra_repr(self) -> str:
        return (
            f'd_model={self.d_model}, '
            f'n_heads={self.n_heads}, '
            f'window_size={self.window_size}, '
            f'n_persistent={self.n_persistent}'
        )
