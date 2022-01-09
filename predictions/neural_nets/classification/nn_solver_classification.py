import torch
from typing import Union
from torch.utils.data import DataLoader
from utils import perform_statistics_classification
from .lstm_classification_model import Classification_LSTM
from .ann_classification_model import Classification_ANN


def train_classification(
    net: Union[Classification_ANN, Classification_LSTM],
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_path: str,
    epochs: int,
    loss_type: str,
    lr: float,
) -> None:
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    if loss_type != "BCELoss":
        raise NotImplementedError(
            f"Loss {loss_type} not implemented. Please use 'BCELoss' for classification"
        )
    criterion = torch.nn.BCELoss()

    min_val_loss = 1e20

    for t in range(epochs):
        epoch_loss = 0.0
        epoch_val_loss = 0.0

        for data, target in train_loader:
            y_pred = net(data)

            train_loss = criterion(y_pred, target.reshape(-1, 1))

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            epoch_loss += train_loss.item()

        with torch.no_grad():
            for data, target in val_loader:
                y_pred = net(data)

                val_loss = criterion(y_pred, target.reshape(-1, 1))
                epoch_val_loss += val_loss.item()

            if epoch_val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(net.state_dict(), model_path)

        print(
            f"\rEpoch {t + 1} validation loss is {epoch_val_loss}, training loss is {epoch_loss}",
            flush=True,
            end="",
        )

    print("\n")


def evaluate_classification(
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
        model = Classification_LSTM(n_features, hidden_units, interval)
    elif nn_type == "ANN":
        model = Classification_ANN(n_features, interval)
    elif nn_type == "dropout":
        model = Classification_ANN(n_features, interval, dropout=dropout)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    for data, target in test_loader:
        y_pred = torch.round(torch.flatten(model(data)))
        y_pred = y_pred.detach().numpy()
        y_true = target.detach().numpy()

        perform_statistics_classification(
            y_true, y_pred, model_type=nn_type, interval=interval, period=period
        )
