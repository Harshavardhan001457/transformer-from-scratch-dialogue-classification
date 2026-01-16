import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from training.evaluate import evaluate
from utils.masking import create_padding_mask

def train(model,train_loader,val_loader,device,epochs= 5,lr= 1e-4):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            mask = create_padding_mask(input_ids, pad_idx=0)
            optimizer.zero_grad()

            logits = model(input_ids, mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.4f}"
        )
