import torch


def create_padding_mask(
    input_ids: torch.Tensor,
    pad_idx: int
) -> torch.Tensor:
    """
    Creates padding mask.

    input_ids: (batch_size, seq_len)
    return: (batch_size, 1, 1, seq_len)
    """
    mask = (input_ids != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def apply_attention_mask(
    scores: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Applies mask to attention scores.

    scores: (batch_size, num_heads, seq_len, seq_len)
    mask:   (batch_size, 1, 1, seq_len)
    """
    scores = scores.masked_fill(mask == 0, float("-inf"))
    return scores
