from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import torch
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from data_loaders import SequenceDataset, StandardDataset


def perform_statistics(y_pred: np.ndarray, y_test: np.ndarray) -> None:
    print("Statistics")

    mrse = np.sqrt(mean_squared_error(y_pred, y_test))
    print(f"rmse is {mrse}")

    mae = mean_absolute_error(y_test, y_pred)
    print(f"mae is {mae}")

    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"mape is {mape}")

    r2 = r2_score(y_test, y_pred)
    print(f"R2 is {r2}\n")


def prepare_dataset(interval_str: str, period: int, nn_type: str, n_nets=None) -> Tuple:
    target = f"bitcoin-price_raw_{period}d"
    df = pd.read_csv(f"./data/continuous/{interval_str}/{period}d_indicators.csv", index_col=False)
    df = df.set_index("Date")

    x, y = transform(df, target)

    # if nn_type == "SANN":
    #     kf = GroupKFold(n_splits=n_nets)
    #     dataset_dict = {net_nr: {"train": StandardDataset((x[train_index], y[train_index])), "test": StandardDataset((x[test_index], y[test_index]))} for net_nr, (train_index, test_index) in enumerate(kf.split(x))}
    #     return dataset_dict
    # else:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

        if nn_type == "ANN" or nn_type == "dropout":
            train_dataset = StandardDataset(X_train, y_train)
            test_dataset = StandardDataset(X_test, y_test)
            val_dataset = StandardDataset(X_val, y_val)

        elif nn_type == "LSTM":
            train_dataset = SequenceDataset(X_train, y_train)
            test_dataset = SequenceDataset(X_test, y_test)
            val_dataset = SequenceDataset(X_val, y_val)

        return train_dataset, test_dataset, val_dataset


def transform(df: pd.DataFrame, target) -> pd.DataFrame:
    minmax = MinMaxScaler()
    robust = RobustScaler()
    x = df.drop(columns=[target])
    y = torch.tensor(df[target]).float()
    x = minmax.fit_transform(x.values)
    x = torch.tensor(robust.fit_transform(x)).float()
    return x, y
