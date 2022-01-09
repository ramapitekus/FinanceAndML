import torch
from utils import perform_statistics_regression
import math
from typing import Union
from torch.utils.data import DataLoader
from .lstm_regression_model import Regression_LSTM
from .ann_regression_model import Regression_ANN


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2.0 * x) - math.log(2.0)

    return torch.mean(_log_cosh(y_pred - y_true))


def train_regression(
    net: Union[Regression_ANN, Regression_LSTM],
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_path: str,
    loss_type: str,
    epochs: int,
    lr: float,
) -> None:
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    if loss_type == "logcosh":
        loss = LogCoshLoss()
    elif loss_type == "mse":
        loss = torch.nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss type {loss_type} not implemented.")

    min_val_loss = 1e20

    for t in range(epochs):
        total_loss = 0.0

        for data, target in train_loader:
            y_pred = torch.flatten(net(data))

            train_loss = loss(y_pred, target)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            total_loss += train_loss

        with torch.no_grad():
            val_loss = 0.0
            for data, target in val_loader:
                y_pred = torch.flatten(net(data))
                val_loss += loss(y_pred, target)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(net.state_dict(), model_path)

        print(
            f"\rEpoch {t + 1} validation loss is {int(val_loss)}, training loss {int(total_loss)}",
            flush=True,
            end="",
        )

    print("\n")


def evaluate_regression(
    n_features: int,
    test_loader: DataLoader,
    interval: int,
    period: int,
    model_path: str,
    nn_type: str,
    hidden_units: int = None,
    dropout: float = None,
) -> None:
    if nn_type == "LSTM":
        model = Regression_LSTM(n_features, hidden_units, interval)
    elif nn_type == "ANN":
        model = Regression_ANN(n_features, interval)
    elif nn_type == "dropout":
        model = Regression_ANN(n_features, interval, dropout=dropout)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    for data, target in test_loader:
        y_pred = torch.flatten(model(data))
        y_pred = y_pred.detach().numpy()
        y_test = target.detach().numpy()

        perform_statistics_regression(
            y_pred,
            y_test,
            nn_type,
            interval,
            period,
        )
