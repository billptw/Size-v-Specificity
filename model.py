import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

### LSTM Classifier taken from https://www.kaggle.com/vadbeg/pytorch-lstm-with-disaster-tweets/comments#Predictions
### However, do note that word embeddings are not yet uploaded to this model.
class LSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, weights=None):
        super(LSTMClassifier, self).__init__()
        
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        self.word_embeddings = torch.nn.Embedding(vocab_size,
                                                  embedding_dim)
        if weights is not None:
            self.word_embeddings.weight = torch.nn.Parameter(weights,
                                                         requires_grad=False)

        self.dropout_1 = torch.nn.Dropout(0.5)
        self.lstm = torch.nn.LSTM(embedding_dim,
                                  hidden_dim,
                                  batch_first=True)
        
        self.dropout_2 = torch.nn.Dropout(0.5)
        self.label_layer = torch.nn.Linear(hidden_dim, output_size)
        
        self.act = torch.nn.Sigmoid()
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        
        x = self.word_embeddings(x)
        
        x = self.dropout_1(x)

        lstm_out, hidden = self.lstm(x, hidden)
        
        # print(hidden[0].size())
        # lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        output_hidden = hidden[0].squeeze() # 32, 256 # LSTM returns hidden, cell only need hidden
        # print(output_hidden.size())
        # input()
        out = self.dropout_2(output_hidden)
        output = self.label_layer(out)  # 32, 2
        # print(out.size())
        # input()
        return output, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to("cuda"),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to("cuda"))
        
        return hidden