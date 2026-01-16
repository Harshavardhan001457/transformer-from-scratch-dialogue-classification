import torch
import torch.nn as nn

from model.transformer import TransformerEncoder


class TransformerDialogueClassifier(nn.Module):
    def __init__(self,vocab_size,num_classes,d_model= 256,num_heads= 8,num_layers= 4,d_ff= 512,max_len= 128,dropout= 0.1):

        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self,input_ids,mask= None):

        encoder_outputs = self.encoder(input_ids, mask)

        # Using [CLS] token representation
        cls_representation = encoder_outputs[:, 0, :]

        logits = self.classifier(cls_representation)
        return logits