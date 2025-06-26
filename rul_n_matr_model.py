import numpy as np
import math
import torch
import torch.nn as nn
from utils import get_train_test, get_train_test_from_df, setup_seed
import pandas as pd

seed=0
setup_seed(seed)

K = 16
Rated_Capacity = 1.1
feature_size = 64
feature_num = K
dropout = 0.0
epochs = 500
nhead = 1
hidden_dim = 16
num_layers = 3
lr = 0.001  
weight_decay = 0.0
noise_level = 0.0
alpha = 0.1
metric = 're'
seed = 0
device = "cpu"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
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
        half_feature_size = int(feature_size/2)
        
        
        self.autoencoder = nn.Linear(feature_size, half_feature_size)
        self.pe = PositionalEncoding(half_feature_size, feature_num)
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_num, nhead=nhead, dim_feedforward=hidden_dim, dropout=0.0, batch_first=True)
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(feature_num*half_feature_size, 1)
        
    def forward(self, x):
        
        B,T,C = x.shape
        # print(f'input x.shape: {x.shape}')
        x = self.autoencoder(x)
        # print(f'after autoencoder, x.shape: {x.shape}')
        x = x.reshape(B, -1, T)
        # print(f'after reshape before positional encoding, x.shape: {x.shape}')
        x = self.pe(x)
        # print(f'after positional encoding, x.shape: {x.shape}')
        x = self.layers(x)
        # print(f'after transformer encoder, x.shape: {x.shape}')
        x = x.reshape(B, -1)
        # print(f'after reshape, x.shape: {x.shape}')
        x = self.lm_head(x)
        # print(f'after lm_head, x.shape: {x.shape}')
        return x
    
def load_data_from_npy():
    Battery = np.load('datasets/CALCE/CALCE.npy', allow_pickle=True)
    Battery = Battery.item() # A dict with cell name as key and dataframe as value
    name = 'CS2_35'
    feature_size = 64
    train_x, train_y, train_data, test_data = get_train_test(Battery, name, feature_size)
    x = np.reshape(train_x/Rated_Capacity,(-1, 1, feature_size)).astype(np.float32)
    y = np.reshape(train_y/Rated_Capacity,(-1,1)).astype(np.float32) 

    x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    x = x.repeat(1, K, 1)
    return x, y

def load_csv_data(cell_type):
    input_data_dict = {
        "CALCE": "./datasets/CALCE/CALCE.csv",
        "matr": "./datasets/matr/matr_part.csv"
    }
    wanted_columns = ['cycle', 'capacity', 'cell_name']
    
    if cell_type == "CALCE":
        df = pd.read_csv(input_data_dict[cell_type])
        return df[wanted_columns]
    if cell_type == "matr":
        df = pd.read_csv(input_data_dict[cell_type])
        # print(f'dfread {df}')
        df["cycle"] = df["cycle_index"]
        df["capacity"] = df["discharge_capacity"]
        df["cell_name"] = df["file_name"]
        return df[wanted_columns]
    
def load_data_from_csv(cell_type, test_cell_name):
    battery_df = load_csv_data(cell_type)
    battery_df['cycle'] = battery_df['cycle'].astype(int)
    battery_df = battery_df[battery_df['cycle'] != 0]
    
    feature_size = 64
    train_x, train_y, train_data, test_data = get_train_test_from_df(battery_df, test_cell_name, feature_size)
    x = np.reshape(train_x/Rated_Capacity,(-1, 1, feature_size)).astype(np.float32)
    y = np.reshape(train_y/Rated_Capacity,(-1,1)).astype(np.float32) 

    x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    x = x.repeat(1, K, 1)
    return x, y


    
if __name__ == "__main__":
    
    # x, y = load_data_from_npy()
    # x, y = load_data_from_csv(cell_type="CALCE", test_cell_name="CS2_35")
    x, y = load_data_from_csv(cell_type="matr", test_cell_name="FastCharge_000001_CH38_structure")
    print(f'x.shape: {x.shape}, y.shape: {y.shape}')

    model = Transformer(feature_size, hidden_dim, feature_num, num_layers, nhead)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    
    for epoch in range(epochs):
        output = model(x)
        output = output.reshape(-1, 1)
        loss = nn.MSELoss()(output, y)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            loss_avg = np.mean(losses[-10:])
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss_avg))
        
        
    
    
    
    