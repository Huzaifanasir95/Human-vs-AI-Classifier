import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with Attention mechanism.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, n_layers: int, bidirectional: bool, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        # text: [batch size, sent len]
        embedded = self.dropout(self.embedding(text))
        
        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Concat the final forward and backward hidden states
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
            
        return self.fc(hidden)

class TransferLearningClassifier:
    """
    Handles BERT/RoBERTa sequence classification.
    """
    def __init__(self, model_name: str, num_labels: int = 2):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
    def get_optimizer(self, lr: float = 2e-5):
        return torch.optim.AdamW(self.model.parameters(), lr=lr)
