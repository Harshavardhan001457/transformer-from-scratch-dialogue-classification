import torch


def create_padding_mask(input_ids,pad_idx):
    mask = (input_ids != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def apply_attention_mask(scores,mask):
    scores = scores.masked_fill(mask == 0, float("-inf"))
    return scores
