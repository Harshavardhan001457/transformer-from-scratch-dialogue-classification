import torch


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Computes accuracy.
    """
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Returns accuracy from logits.
    """
    preds = torch.argmax(logits, dim=1)
    acc = accuracy(preds, labels)

    return {
        "accuracy": acc
    }
