import torch
import torch.nn as nn

from model.transformer import TransformerEncoder


class TransformerDialogueClassifier(nn.Module):
    """
    Transformer-based dialogue classifier.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        input_ids: (batch_size, seq_len)
        mask: (batch_size, 1, 1, seq_len)
        """

        encoder_outputs = self.encoder(input_ids, mask)

        # Use [CLS] token representation
        cls_representation = encoder_outputs[:, 0, :]

        logits = self.classifier(cls_representation)
        return logits
