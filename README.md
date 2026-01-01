##  Transformer From Scratch for Multi-Turn Dialogue Classification

###  Project Overview

This project implements a **Transformer Encoder architecture entirely from scratch using PyTorch** and applies it to **multi-turn conversational dialogue classification** using the **DailyDialog dataset**.

The primary objective is **conceptual understanding and correct implementation** of Transformer internals—particularly **multi-head self-attention, masking, and encoder blocks**—without relying on high-level abstractions or pretrained models.

Unlike typical NLP pipelines that use pretrained Transformers, this project builds every core component manually to demonstrate a deep understanding of modern sequence modeling.



##  Task Formulation

* **Input**: Multi-turn dialogue context (previous utterances + current utterance)
* **Output**: Emotion label of the current utterance
* **Granularity**: Utterance-level classification with conversational context

Each dialogue is modeled as a sequence of utterances, where contextual information from previous turns is incorporated to improve prediction quality.



##  Core Approach & Logic

### 1️ Data Representation

* Word-level tokenization (from scratch)
* Custom vocabulary built from training data
* Special tokens: `<CLS>`, `<SEP>`, `<PAD>`, `<UNK>`
* Dialogue context constructed using a fixed-size sliding window of previous utterances

**Input Format**

```
[CLS] utterance(t-2) [SEP] utterance(t-1) [SEP] utterance(t) [SEP]
```



### 2️ Transformer Architecture (From First Principles)

The model is a **pure Transformer Encoder**, implemented manually using low-level PyTorch operations.

**Key Components**

* Token Embedding + Learned Positional Embedding
* Scaled Dot-Product Self-Attention
* Multi-Head Attention (manual head splitting & concatenation)
* Residual Connections + Layer Normalization
* Position-wise Feed-Forward Network
* Padding Masking to ignore padded tokens during attention

No PyTorch Transformer modules or pretrained components were used.



### 3️ Classification Strategy

* The contextual representation of the `[CLS]` token is used as a global summary
* A linear classification head maps this representation to emotion classes
* Cross-entropy loss used for optimization



## Training & Evaluation

* Dataset: **DailyDialog**
* Separate official validation split used
* Optimizer: Adam
* Loss Function: Cross-Entropy Loss
* Metric: Validation Accuracy

**Final Validation Accuracy**: **~0.8132**

This performance is achieved **without pretrained embeddings or models**, focusing purely on architectural correctness.



## ⚠️ Limitations

* Word-level tokenization may lose subword semantics
* Class imbalance in emotion labels affects metric sensitivity
* Accuracy alone does not fully reflect minority-class performance
* Fixed context window limits very long-range dialogue dependencies

These limitations are acknowledged as trade-offs made to prioritize architectural clarity and from-scratch implementation.

## ⚠️ Note on Training

* Full training was performed on GPU (Google Colab).
* For convenience, pretrained weights are provided.
* Inference and evaluation can be run on CPU in seconds.


##  Key Learnings

* Deep understanding of Transformer internals
* Practical experience implementing attention and masking manually
* Handling real-world conversational data and multi-turn context
* Debugging training dynamics and validation behavior in NLP models



##  Project Structure

```
├── data/
│   ├── DailyDialog/
│        ├── test.csv
│        ├── train.csv
│        ├── validation.csv
│   ├── test_data.json
│   ├── train_data.json
│   ├── val_data.json
│   ├── transformer_classifier_weights.pt
│   └── data_processing.ipynb
│
├── model/
│   ├── attention.py
│   ├── multihead.py
│   ├── encoder_block.py
│   ├── transformer.py
│   └── classifier.py
│
├── utils/
│   ├── datautils.py
│   └── masking.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── metrics.py
│
├── main.py
├── .gitignore
├── requirements.txt
└── README.md
```



##  Future Improvements

* Add macro-F1 score for better evaluation under class imbalance
* Experiment with different context window sizes
* Extend to multi-task learning (emotion + dialogue act)
* Add attention visualization for interpretability



##  Conclusion

This project demonstrates a **ground-up implementation of a Transformer model for dialogue understanding**, emphasizing conceptual clarity, correctness, and practical NLP engineering skills.
