import torch
from training.metrics import classification_metrics
from utils.masking import create_padding_mask

def evaluate(
    model,
    dataloader,
    device
):
    """
    Evaluates model on validation/test set.
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            # mask = create_padding_mask(input_ids, pad_idx=model.encoder.token_embedding.padding_idx)
            # Use pad_idx=0 as it is the index for <PAD> token
            mask = create_padding_mask(input_ids, pad_idx=0)
            logits = model(input_ids)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = classification_metrics(all_logits, all_labels)
    return metrics