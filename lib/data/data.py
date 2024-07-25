import pandas as pd 
import numpy as np 
import torch 


def get_data_and_preprocess(
        csv_path, 
        target, 
        timesteps, 
        train_split, 
        val_split
): 
    data = pd.read_csv(csv_path)
    data.shape[1] - 1

    # placeholders
    X = np.zeros((len(data), timesteps, data.shape[1]-1))
    y = np.zeros((len(data), timesteps, 1))

    # fill X : 
    # for each time serie
    for i, name in enumerate(list(data.columns[:-1])):
        # for each timestep
        for j in range(timesteps):
            X[:, j, i] = data[name].shift(timesteps - j - 1).fillna(method="bfill")

    
    for j in range(timesteps):
       y[:, j, 0] = data[target].shift(timesteps - j - 1).fillna(method="bfill")

    prediction_horizon = 1
    # The prediction horizon is everything that comes after the timestamps. 
    # If you are using [t=1, ..., t=10] as input, your target will be t=11. 
    target = data[target].shift(-prediction_horizon).fillna(method="ffill").values
    
    train_length = int(len(data) * train_split)
    val_length = int(len(data) * val_split)

    X_train = X[:train_length]
    y_his_train = y[:train_length]
    X_val = X[train_length:train_length+val_length]
    y_his_val = y[train_length:train_length+val_length]
    X_test = X[-val_length:]
    y_his_test = y[-val_length:]
    target_train = target[:train_length]
    target_val = target[train_length:train_length+val_length]
    target_test = target[-val_length:]

    # min max scaling 
    X_train_max = X_train.max(axis=0)
    X_train_min = X_train.min(axis=0)
    y_his_train_max = y_his_train.max(axis=0)
    y_his_train_min = y_his_train.min(axis=0)
    target_train_max = target_train.max(axis=0)
    target_train_min = target_train.min(axis=0)

    X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
    X_val = (X_val - X_train_min) / (X_train_max - X_train_min)
    X_test = (X_test - X_train_min) / (X_train_max - X_train_min)

    y_his_train = (y_his_train - y_his_train_min) / (y_his_train_max - y_his_train_min)
    y_his_val = (y_his_val - y_his_train_min) / (y_his_train_max - y_his_train_min)
    y_his_test = (y_his_test - y_his_train_min) / (y_his_train_max - y_his_train_min)

    target_train = (target_train - target_train_min) / (target_train_max - target_train_min)
    target_val = (target_val - target_train_min) / (target_train_max - target_train_min)
    target_test = (target_test - target_train_min) / (target_train_max - target_train_min)

    X_train_t = torch.Tensor(X_train)
    X_val_t = torch.Tensor(X_val)
    X_test_t = torch.Tensor(X_test)
    y_his_train_t = torch.Tensor(y_his_train)
    y_his_val_t = torch.Tensor(y_his_val)
    y_his_test_t = torch.Tensor(y_his_test)
    target_train_t = torch.Tensor(target_train)
    target_val_t = torch.Tensor(target_val)
    target_test_t = torch.Tensor(target_test)

    return [
        X_train_t,
        X_val_t,
        X_test_t,
        y_his_train_t,
        y_his_val_t,
        y_his_test_t,
        target_train_t,
        target_val_t,
        target_test_t
    ]

