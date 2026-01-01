import torch
import torch.nn as nn

from model.attention import ScaledDotProductAttention


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention implemented from scratch.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Splits last dimension into (num_heads, head_dim).
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (B, H, L, D)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combines heads back to d_model.
        """
        batch_size, num_heads, seq_len, head_dim = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        mask: (batch_size, 1, 1, seq_len)
        """

        # Linear projections
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Apply attention
        attn_output, _ = self.attention(Q, K, V, mask)

        # Combine heads
        combined = self.combine_heads(attn_output)

        # Final linear projection
        output = self.out_proj(combined)
        output = self.dropout(output)

        return output
