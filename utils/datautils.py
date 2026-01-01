import json
import re
from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

# -------------------------
# Special Tokens
# -------------------------
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"
SEP_TOKEN = "<SEP>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN]


# -------------------------
# Basic Tokenizer
# -------------------------
def tokenize(text: str) -> List[str]:
    """
    Simple word-level tokenizer.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


# -------------------------
# Vocabulary Builder
# -------------------------
def build_vocab(dialogues: List[List[str]],
                min_freq: int = 2) -> dict:
    """
    Builds word-to-index vocabulary.
    """
    counter = Counter()

    for dialogue in dialogues:
        for utterance in dialogue:
            tokens = tokenize(utterance)
            counter.update(tokens)

    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


# -------------------------
# Encoding Utilities
# -------------------------
def encode_tokens(tokens: List[str], vocab: dict) -> List[int]:
    """
    Converts tokens to indices.
    """
    return [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]


def pad_sequence(seq: List[int], max_len: int, pad_value: int) -> List[int]:
    """
    Pads or truncates sequence.
    """
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_value] * (max_len - len(seq))


# -------------------------
# Dataset Class
# -------------------------
class DailyDialogDataset(Dataset):
    """
    Dataset for multi-turn dialogue classification.
    """

    def __init__(
        self,
        dialogues: List[List[str]],
        labels: List[List[int]],
        vocab: dict,
        context_size: int = 2,
        max_len: int = 128
    ):
        """
        dialogues: List of dialogues (each dialogue is list of utterances)
        labels: List of label sequences (aligned with utterances)
        """
        self.samples = []
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = vocab[PAD_TOKEN]

        for dialog, label_seq in zip(dialogues, labels):
            for idx in range(len(dialog)):
                start = max(0, idx - context_size)
                context = dialog[start:idx + 1]

                tokens = [CLS_TOKEN]
                for utt in context:
                    tokens.extend(tokenize(utt))
                    tokens.append(SEP_TOKEN)

                token_ids = encode_tokens(tokens, vocab)
                token_ids = pad_sequence(token_ids, max_len, self.pad_idx)

                self.samples.append(
                    (
                        torch.tensor(token_ids, dtype=torch.long),
                        torch.tensor(label_seq[idx], dtype=torch.long)
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# -------------------------
# Data Loader Helper
# -------------------------
def load_dailydialog_json(path: str) -> Tuple[List[List[str]], List[List[int]]]:
    """
    Loads DailyDialog-style JSON file.
    Expected keys: 'dialogues', 'labels'
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dialogues = data["dialogues"]
    labels = data["labels"]

    return dialogues, labels

def load_dataset(data_type, DATA_PATH,CONTEXT_SIZE, MAX_LEN):
    print(f"Loading {data_type} dataset...")
    dialogues, labels = load_dailydialog_json(DATA_PATH)

    print("Building vocabulary...")
    vocab = build_vocab(dialogues)

    num_classes = len(set(label for seq in labels for label in seq))
    vocab_size = len(vocab)

    print(f"Vocab size: {vocab_size}")
    print(f"Number of classes: {num_classes}")

    print("Creating datasets...")
    dataset = DailyDialogDataset(
        dialogues=dialogues,
        labels=labels,
        vocab=vocab,
        context_size=CONTEXT_SIZE,
        max_len=MAX_LEN
    )
    return [dataset, vocab_size, num_classes]