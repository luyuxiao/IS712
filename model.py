import torch
import torch.nn as nn
import numpy as np
from transformers import RobertaTokenizer, RobertaModel, BertModel, AutoModel, AutoModelForSequenceClassification
import os


class RobertaClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fnn_size = args.fnn_size
        self.class_num = args.class_num
        # self.roberta = RobertaModel.from_pretrained("roberta-large")
        # self.roberta = RobertaModel.from_pretrained("roberta-base")
        # self.roberta = AutoModel.from_pretrained("gsarti/scibert-nli")
        # self.roberta = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
        # self.roberta = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-distilroberta-base")
        self.roberta = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").base_model
        # self.roberta = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").base_model
        # self.roberta = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-xlarge-mnli").base_model
        # self.roberta = AutoModelForSequenceClassification.from_pretrained("cross-encoder/nli-distilroberta-base")
        for param in self.roberta.parameters():
            param.requires_grad = True
            # param.requires_grad = False
        self.fc1 = nn.Linear(args.pretrained_model_dim, self.fnn_size)
        self.fc2 = nn.Linear(self.fnn_size, self.class_num)

    def forward(self, text_ids, mask):
        features = self.roberta(text_ids, mask).last_hidden_state  # batch_size, seq_len, dim
        # features = self.roberta(text_ids, mask)
        divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1).type_as(text_ids)
        average_features = features.sum(dim=1) / divisor
        inter_features = self.fc1(torch.relu(average_features))
        output_features = self.fc2(torch.relu(inter_features))
        return output_features
