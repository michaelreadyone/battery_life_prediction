import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from diy_prepare_train_data import prepare_train_data, prepare_train_data_original_style, create_data_loader
from diy_model import Transformer
device = "cpu"

feature_size = 64
hidden_dim = 16
feature_num = 16
num_layers = 3
nhead = 1
Rated_Capacity = 1.1
use_original_data_prep = True
use_full_batch = True
lr = 0.001
weight_decay = 0.0

epochs = 50


if __name__ == "__main__":
    cell_type = "matr"
    test_cell_name = "FastCharge_000001_CH38_structure"
    window_size = 64
    batch_size = 32
    
    
    # Create data using improved approach
    if use_original_data_prep:
        print("Using original-style data preparation (like rul_n_matr_model_use_common_transformer.py)")
        X_tensor, y_tensor = prepare_train_data_original_style(cell_type, test_cell_name, window_size, Rated_Capacity)
        print(f"Data shapes: X={X_tensor.shape}, y={y_tensor.shape}")
        print(f"Data already normalized by Rated_Capacity={Rated_Capacity}")
    else:
        print("Using windowing data preparation")
        X_tensor, y_tensor = prepare_train_data(cell_type, test_cell_name, window_size)
        
        # Normalize data for better training stability
        X_mean, X_std = X_tensor.mean(), X_tensor.std()
        y_mean, y_std = y_tensor.mean(), y_tensor.std()
        X_tensor = (X_tensor - X_mean) / X_std
        y_tensor = (y_tensor - y_mean) / y_std
        
        print(f"Data shapes: X={X_tensor.shape}, y={y_tensor.shape}")
        print(f"Data stats: X_mean={X_mean:.4f}, X_std={X_std:.4f}, y_mean={y_mean:.4f}, y_std={y_std:.4f}")
        
        # Add channel dimension for windowing approach
        X_tensor = X_tensor.unsqueeze(-1)
        y_tensor = y_tensor.unsqueeze(-1)
        
    if not use_full_batch:
        dataloader = create_data_loader(X_tensor, y_tensor, batch_size=batch_size, shuffle=True)
    
    model = Transformer(feature_size, hidden_dim, feature_num, num_layers, nhead)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    losses = []
    
    for epoch in range(epochs):
        model.train()
        
        if use_full_batch:
            # Full batch training like original
            optimizer.zero_grad()
            output = model(X_tensor)
            output = output.reshape(-1, 1)
            loss = nn.MSELoss()(output, y_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                loss_avg = np.mean(losses[-10:])
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss_avg))
        else:
            # Mini-batch training
            epoch_losses = []
            
            for batch_idx, (X, y) in enumerate(dataloader):
                if not use_original_data_prep:
                    X = X.unsqueeze(-1)  # Add channel dimension: (B, T) -> (B, T, 1)
                    y = y.unsqueeze(-1)  # Add dimension for loss calculation
                
                optimizer.zero_grad()
                output = model(X)
                loss = nn.MSELoss()(output, y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_losses.append(loss.item())
                
            epoch_loss = np.mean(epoch_losses)
            print('Epoch [{}/{}], Loss: {:.6f}, Batches: {}'.format(epoch + 1, epochs, epoch_loss, len(epoch_losses)))
    
