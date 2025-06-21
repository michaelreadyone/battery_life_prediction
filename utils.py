import os
import random
import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch


def build_sequences(text, window_size):
    #text:list of capacity
    # print(f'text: {text}')
    x, y = [],[]
    for i in range(len(text) - window_size):
        sequence = text[i:i+window_size]
        # print(f'sequence: {sequence}')
        target = text[i+window_size]
        # print(f'target: {target}')

        x.append(sequence)
        y.append(target)
        
    return np.array(x), np.array(y)

# leave-one-out evaluation: one battery is sampled randomly; the remainder are used for training.
def get_train_test(data_dict, name, window_size=8):
    data_sequence=data_dict[name]['capacity']
    train_data, test_data = data_sequence[:window_size+1], data_sequence[window_size+1:]
    # why add test data?
    # train_x, train_y = build_sequences(text=train_data, window_size=window_size)
    train_x, train_y = np.empty((0, window_size)), np.empty((0,))

    for k, v in data_dict.items():
        if k != name:
            data_x, data_y = build_sequences(text=v['capacity'], window_size=window_size)
            train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]
            
    return train_x, train_y, list(train_data), list(test_data)

def get_train_test_from_df(battery_df, name, window_size):
    battery_names = list(battery_df['cell_name'].unique())
    print(f'battery_names: {battery_names}')
    
    test_df = battery_df[battery_df['cell_name'] == name]
    data_sequence=test_df['capacity']
    train_data, test_data = data_sequence[:window_size+1], data_sequence[window_size+1:]
    # why add test data?
    # train_x, train_y = build_sequences(text=train_data, window_size=window_size)
    train_x, train_y = np.empty((0, window_size)), np.empty((0,))
    
    

    for battery_name in battery_names:
        if battery_name == name:
            continue
        cell_df = battery_df[battery_df['cell_name'] == battery_name].reset_index(drop=True)
        # print(f'cell_df: {cell_df}')

        data_x, data_y = build_sequences(text=cell_df['capacity'], window_size=window_size)
        train_x, train_y = np.r_[train_x, data_x], np.r_[train_y, data_y]
            
    return train_x, train_y, list(train_data), list(test_data)


def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True