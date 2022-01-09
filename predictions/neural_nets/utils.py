from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
import numpy as np
from typing import Union
import pandas as pd
import torch
from regression.ann_regression_model import Regression_ANN
from regression.lstm_regression_model import Regression_LSTM
from classification.lstm_classification_model import Classification_LSTM
from classification.ann_classification_model import Classification_ANN
from typing import List
from torch.utils import data
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from data_loaders import SequenceDataset, StandardDataset
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    auc,
    roc_curve,
)


def perform_statistics_regression(
    y_pred: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    interval: int,
    period: int,
) -> None:
    log = open(f"predictions/logs/regression/{model_type}_log.txt", "a")
    stats_dict = {}
    stats_dict["mrse"] = np.sqrt(mean_squared_error(y_pred, y_test))
    stats_dict["mae"] = mean_absolute_error(y_test, y_pred)
    stats_dict["mape"] = mean_absolute_percentage_error(y_test, y_pred)

    print(f"{model_type} regression, interval {interval} period {period}", file=log)
    for stat_name, stat_val in stats_dict.items():
        print(f"{stat_name} is {stat_val}", file=log)


def perform_statistics_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_type: str,
    interval: int,
    period: int,
) -> None:
    log = open(f"predictions/logs/classification/{model_type}_log.txt", "a")
    stats_dict = {}
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    stats_dict["specificity"] = tn / (tn + fp)
    stats_dict["f1"] = f1_score(y_true, y_pred)
    stats_dict["accuracy"] = accuracy_score(y_true, y_pred)
    stats_dict["precision"] = precision_score(y_true, y_pred)
    stats_dict["recall"] = recall_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    stats_dict["area under curve"] = auc(fpr, tpr)

    print(f"{model_type} classification, interval {interval} period {period}", file=log)
    for stat_name, stat_val in stats_dict.items():
        print(f"{stat_name} is {stat_val}", file=log)


def prepare_dataset(
    prediction: str, interval_str: str, period: int, nn_type: str, shuffle: bool
) -> Tuple:
    y_type = "continuous" if prediction == "regression" else "categorical"
    y_target = (
        f"bitcoin-price_raw_{period}d"
        if prediction == "regression"
        else f"{period}d_binary_price_change"
    )

    df = pd.read_csv(
        f"./data/{y_type}/{interval_str}/{period}d_indicators.csv", index_col=False
    )
    df = df.set_index("Date")

    x, y = transform(df, y_target)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=shuffle, random_state=3
    )
    X_train, X_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, shuffle=shuffle, random_state=3
    )

    if nn_type == "ANN" or nn_type == "dropout":
        train_dataset = StandardDataset(X_train, y_train)
        test_dataset = StandardDataset(X_test, y_test)
        val_dataset = StandardDataset(X_val, y_val)

    elif nn_type == "LSTM":
        train_dataset = SequenceDataset(X_train, y_train)
        test_dataset = SequenceDataset(X_test, y_test)
        val_dataset = SequenceDataset(X_val, y_val)

    return train_dataset, test_dataset, val_dataset


def transform(df: pd.DataFrame, y_target: List) -> Tuple:
    minmax = MinMaxScaler()
    robust = RobustScaler()
    x = df.drop(columns=[y_target])
    y = torch.tensor(df[y_target]).float()
    x = minmax.fit_transform(x.values)
    x = torch.tensor(robust.fit_transform(x)).float()
    return x, y


def create_loaders(
    train_dataset: Union[StandardDataset, SequenceDataset],
    val_dataset: Union[StandardDataset, SequenceDataset],
    test_dataset: Union[StandardDataset, SequenceDataset],
    batch_size: int = 64,
) -> Tuple:
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader


def pick_nn_model(
    prediction_type: str,
    nn_type: str,
    n_features: int,
    interval: int,
    dropout: float = None,
):
    if prediction_type == "regression":
        if nn_type == "LSTM":
            net = Regression_LSTM(
                n_features, hidden_units=n_features, interval=interval
            )

        elif nn_type == "ANN":
            net = Regression_ANN(n_features, interval)

        elif nn_type == "dropout":
            net = Regression_ANN(n_features, interval, dropout)

    if prediction_type == "classification":
        if nn_type == "LSTM":
            net = Classification_LSTM(
                n_features, hidden_units=n_features, interval=interval
            )

        elif nn_type == "ANN":
            net = Classification_ANN(n_features, interval)

        elif nn_type == "dropout":
            net = Classification_ANN(n_features, interval, dropout)

    return net
