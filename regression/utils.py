from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import torch
from typing import Tuple
import os
from neural_net import ANN
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def perform_statistics(y_pred: np.ndarray, y_test: np.ndarray) -> None:
    print("Statistics")

    mrse = np.sqrt(mean_squared_error(y_pred, y_test))
    print(f"rmse is {mrse}")

    mae = mean_absolute_error(y_test,y_pred)
    print(f"mae is {mae}")

    mape = mean_absolute_percentage_error(y_test,y_pred)
    print(f"mape is {mape}")

    r2=r2_score(y_test,y_pred)
    print(f"R2 is {r2}\n")


def prepare_dataloader(X_train, y_train):
    train_data = []
    for i in range(len(X_train)):
        train_data.append([X_train[i], y_train[i]])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

    return train_loader


def prepare_data(interval_str: str, period: int) -> Tuple:
        df = pd.read_csv(f"./data/features_selected/continuous/{interval_str}/{period}d_indicators.csv", index_col=False)

        y = df[f"bitcoin-price_raw_{period}d"]
        y = torch.tensor(y.values).float()[:-period]  # Get dependent variables
        
        x = df.drop(columns=[f"bitcoin-price_raw_{period}d", "Date"])  # Get features
        x = torch.tensor(x.values).float()[:-period]

        min_max_scaler = MinMaxScaler()
        robust_scaler = RobustScaler()
        x = torch.tensor(min_max_scaler.fit_transform(x)).float()
        x = torch.tensor(robust_scaler.fit_transform(x)).float()

        return x, y


def prepare_dataloader(X_train, y_train):
    train_data = []
    for i in range(len(X_train)):
        train_data.append([X_train[i], y_train[i]])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

    return train_loader
