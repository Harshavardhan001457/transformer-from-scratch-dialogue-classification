import torch
import torch.nn.functional as F


class ScaledDotProductAttention:

    def __init__(self, dropout: float = 0.0):
        self.dropout = torch.nn.Dropout(dropout)

    def __call__(self,query,key,value,mask= None):

        d_k = query.size(-1)

        # (B, H, L, D) x (B, H, D, L) -> (B, H, L, L)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # (B, H, L, L) x (B, H, L, D) -> (B, H, L, D)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights
