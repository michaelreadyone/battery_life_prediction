import numpy as np
import math
import torch
import torch.nn as nn
from utils import get_train_test, get_train_test_from_df, load_csv_data, setup_seed
import pandas as pd
import os
import plotly.graph_objects as go

seed=0
setup_seed(seed)

class PositionalEncoding(nn.Module):
    def __init__(self, feature_len, feature_size, dropout=0.0):
        # feature_len: max length of sequence, feature_size: embedding size(d_model)
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(feature_len, feature_size)
        position = torch.arange(0, feature_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_size, 2).float() * (-math.log(10000.0) / feature_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        # print(f'pe.shape: {pe.shape}')
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

  
class Transformer(nn.Module):
    def __init__(self, feature_size, hidden_dim, feature_num, num_layers, nhead):
        super(Transformer, self).__init__()
        half_feature_size = int(feature_size)
        
        
        self.autoencoder = nn.Linear(1, feature_num)
        self.pe = PositionalEncoding(feature_size, feature_num)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_num, nhead=nhead, dim_feedforward=hidden_dim, dropout=0.0, batch_first=True)
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(feature_num*half_feature_size, 1)
        
    def forward(self, x):
        
        B,T,C = x.shape
        # print(f'input x.shape: {x.shape}')
        x = self.autoencoder(x)
        # print(f'after autoencoder, x.shape: {x.shape}')
        x = self.pe(x)
        # print(f'after positional encoding, x.shape: {x.shape}')
        x = self.layers(x)
        # print(f'after transformer encoder, x.shape: {x.shape}')
        x = x.reshape(B, -1)
        # print(f'after reshape, x.shape: {x.shape}')
        x = self.lm_head(x)
        # print(f'after lm_head, x.shape: {x.shape}')
        return x