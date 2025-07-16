import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import load_csv_data

def prepare_sequences(battery_df, window_size, test_cell_name):
    """
    Prepare windowed sequences for time series prediction.
    
    Args:
        battery_df: DataFrame with columns 'cycle', 'capacity', 'cell_name'
        window_size: Size of input window (x)
        test_cell_name: Cell name to exclude from training data
    
    Returns:
        X_tensor: Tensor of shape (n_samples, window_size) - input sequences
        y_tensor: Tensor of shape (n_samples,) - target values
    """
    # Exclude test cell
    train_df = battery_df[battery_df['cell_name'] != test_cell_name].copy()
    
    X_list = []
    y_list = []
    
    # Process each cell separately
    for cell_name in train_df['cell_name'].unique():
        cell_data = train_df[train_df['cell_name'] == cell_name].copy()
        cell_data = cell_data.sort_values('cycle').reset_index(drop=True)
        
        capacities = cell_data['capacity'].values
        
        # Generate windowed sequences
        for i in range(len(capacities) - window_size):
            X_window = capacities[i:i + window_size]
            y_target = capacities[i + window_size]
            
            X_list.append(X_window)
            y_list.append(y_target)
    
    X_tensor = torch.FloatTensor(np.array(X_list))
    y_tensor = torch.FloatTensor(np.array(y_list))
    return X_tensor, y_tensor

def create_data_loader(X_tensor, y_tensor, batch_size=32, shuffle=True):
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def prepare_train_data(cell_type, test_cell_name, window_size):
    battery_df = load_csv_data(cell_type)
    battery_df['cycle'] = battery_df['cycle'].astype(int)
    battery_df = battery_df[battery_df['cycle'] != 0]
    X, y = prepare_sequences(battery_df, window_size, test_cell_name)
    return X, y

# Example usage
if __name__ == "__main__":
    cell_type = "matr"
    test_cell_name = "FastCharge_000001_CH38_structure"
    window_size = 64
    batch_size = 32
    
    
    # Create DataLoader
    X_tensor, y_tensor = prepare_train_data(cell_type, test_cell_name, window_size)
    dataloader = create_data_loader(X_tensor, y_tensor, batch_size=batch_size, shuffle=True)
    
    # Test DataLoader
    for X, y in dataloader:
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f'X: {X}, y: {y}')
        break