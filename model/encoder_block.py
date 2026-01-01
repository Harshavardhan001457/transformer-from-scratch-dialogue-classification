import torch
import torch.nn as nn

from model.multihead import MultiHeadSelfAttention


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        self.attn_norm = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.ff_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        mask: (batch_size, 1, 1, seq_len)
        """

        # ---- Self-Attention Block ----
        attn_output = self.self_attention(x, mask)
        x = self.attn_norm(x + self.dropout(attn_output))

        # ---- Feed Forward Block ----
        ff_output = self.feed_forward(x)
        x = self.ff_norm(x + self.dropout(ff_output))

        return x
