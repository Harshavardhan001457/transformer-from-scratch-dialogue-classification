import torch
import torch.nn as nn

from model.encoder_block import TransformerEncoderBlock


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder stack.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_len: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        input_ids: (batch_size, seq_len)
        mask: (batch_size, 1, 1, seq_len)
        """

        batch_size, seq_len = input_ids.size()

        positions = torch.arange(
            0, seq_len,
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x
