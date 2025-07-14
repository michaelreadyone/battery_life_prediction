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

    
def get_train_data(cell_type, test_cell_name):
    battery_df = load_csv_data(cell_type)
    battery_df['cycle'] = battery_df['cycle'].astype(int)
    battery_df = battery_df[battery_df['cycle'] != 0]
    
    feature_size = 64
    train_x, train_y, train_data, test_data = get_train_test_from_df(battery_df, test_cell_name, feature_size)
    print(f'train_x.shape: {train_x.shape}, train_y.shape: {train_y.shape}')
    x = np.reshape(train_x/Rated_Capacity,(-1, 1, feature_size)).astype(np.float32)
    y = np.reshape(train_y/Rated_Capacity,(-1,1)).astype(np.float32) 
    print(f'after reshape, x.shape: {x.shape}, y.shape: {y.shape}')

    x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    x = x.repeat(1, K, 1)
    print(f'after repeat, x.shape: {x.shape}, y.shape: {y.shape}')
    return x, y

def generate(model_weights_path, input_sequence, generate_length):
    model = Transformer(feature_size, hidden_dim, feature_num, num_layers, nhead)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()
    
    input_seq = torch.tensor(input_sequence, dtype=torch.float32).to(device)
    if len(input_seq.shape) == 2:
        input_seq = input_seq.unsqueeze(0)
    
    output_sequence = []
    current_seq = input_seq.clone()
    
    with torch.no_grad():
        for _ in range(generate_length):
            pred = model(current_seq)
            next_value = pred.unsqueeze(-1).repeat(1, 1, feature_size)
            output_sequence.append(pred.cpu().numpy())
            
            current_seq = torch.cat([current_seq[:, 1:, :], next_value], dim=1)
    
    output_sequence = np.concatenate(output_sequence, axis=0).squeeze()
    total_sequence = np.concatenate([input_sequence[:, 0], output_sequence], axis=0)
    
    return total_sequence

def plot_cyclelife(sequence, input_length=None):
    idx = np.arange(len(sequence))
    
    fig = go.Figure()
    
    if input_length is not None:
        fig.add_trace(go.Scatter(
            x=idx[:input_length], 
            y=sequence[:input_length],
            mode='lines',
            name='Input Sequence',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=idx[input_length:], 
            y=sequence[input_length:],
            mode='lines',
            name='Generated Sequence',
            line=dict(color='red')
        ))
    else:
        fig.add_trace(go.Scatter(
            x=idx, 
            y=sequence,
            mode='lines',
            name='Sequence',
            line=dict(color='blue')
        ))
    
    fig.update_layout(
        title='Battery Capacity Sequence',
        xaxis_title='Index',
        yaxis_title='Normalized Capacity',
        showlegend=True
    )
    
    fig.show()

def train():
    os.makedirs('./saved_model/rul_n_matr_model/', exist_ok=True)
    
    # x, y = load_data_from_npy()
    # x, y = get_train_data(cell_type="CALCE", test_cell_name="CS2_35")
    x, y = get_train_data(cell_type="matr", test_cell_name="FastCharge_000001_CH38_structure")
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
            torch.save(model.state_dict(), f'./saved_models/rul_n_matr_model/epoch_{epoch+1}.pth')
    
if __name__ == "__main__":
    # train()
    
    # Test generate function
    model_path = './saved_models/rul_n_matr_model/epoch_500.pth'
    x, y = get_train_data(cell_type="matr", test_cell_name="FastCharge_000001_CH38_structure")
    
    # Use first sequence as input
    input_seq = x[0].cpu().numpy()
    generate_len = 500
    
    print(f'\nTesting generate function:')
    print(f'Input sequence shape: {input_seq.shape}')
    print(f'Generate length: {generate_len}')
    
    total_seq = generate(model_path, input_seq, generate_len)
    print(f'Total sequence shape: {total_seq.shape}')
    print(f'Original input length: {len(input_seq)}')
    print(f'Generated sequence length: {len(total_seq) - len(input_seq)}')
    
    # Plot total_seq using plot_cyclelife function
    plot_cyclelife(total_seq, len(input_seq))
        
        
    
    
    
    