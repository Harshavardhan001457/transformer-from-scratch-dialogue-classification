import torch, os
from torch.utils.data import DataLoader

from utils.datautils import (
    build_vocab,
    load_dailydialog_json,
    DailyDialogDataset
)
from utils.masking import create_padding_mask
from utils.datautils import load_dataset
from model.classifier import TransformerDialogueClassifier
from training.train import train
from training.evaluate import evaluate


# -------------------------
# Configuration
# -------------------------
DATA_PATH1 = "./data/train_data.json"
DATA_PATH2 = "./data/val_data.json"
DATA_PATH3 = "./data/test_data.json"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4

CONTEXT_SIZE = 2
MAX_LEN = 128

D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 512
DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Main
# -------------------------
def main():
    
    dataset1, vocab_size1, num_classes1 = load_dataset("train", DATA_PATH1,CONTEXT_SIZE, MAX_LEN)
    dataset2, vocab_size2, num_classes2 = load_dataset("validation", DATA_PATH2,CONTEXT_SIZE, MAX_LEN)
    dataset3, vocab_size3, num_classes3 = load_dataset("test", DATA_PATH3,CONTEXT_SIZE, MAX_LEN)


    train_loader = DataLoader(
        dataset1, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        dataset2, batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        dataset3, batch_size=BATCH_SIZE
    )

    print("Initializing model...")
    model = TransformerDialogueClassifier(
        vocab_size=vocab_size1,
        num_classes=num_classes1,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT
    )
    model_path = './data/transformer_classifier_weights.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("MOdel already trained")
    else:
        print("Starting training...")
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=DEVICE,
            epochs=EPOCHS,
            lr=LEARNING_RATE
        )

    print("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, DEVICE)

    print("Test Metrics:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
