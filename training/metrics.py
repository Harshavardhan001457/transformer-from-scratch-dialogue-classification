import torch


def accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total


def classification_metrics(logits, labels):
    preds = torch.argmax(logits, dim=1)
    acc = accuracy(preds, labels)

    return {"accuracy": acc}
