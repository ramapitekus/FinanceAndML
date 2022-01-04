import torch
from utils import *
import math
from lstm_model import LSTM
from ann_model import ANN


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))


def train_model(net, train_loader, val_loader, model_path, loss_type: str, epochs, lr: float) -> None:
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    if loss_type == "logcosh":
        loss = LogCoshLoss()
    elif loss_type == "mse":
        loss = torch.nn.MSELoss()
    else:
        raise NotImplementedError("Error type not implemented.")

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

        print(f"\rEpoch {t + 1} validation loss is {int(val_loss)}, training loss {int(total_loss)}", flush=True, end="")

    print("\n")


def evaluate(n_features, test_loader, interval, model_path, nn_type: str, hidden_units=None):
    if nn_type == "LSTM":
        model = LSTM(n_features, hidden_units)
    elif nn_type == "ANN":
        model = ANN(n_features, interval)
    elif nn_type == "dropout":
        model = ANN(n_features, interval, dropout=0.2)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    for data, target in test_loader:
        y_pred = torch.flatten(model(data))
        y_pred = y_pred.detach().numpy()
        y_test = target.detach().numpy()

        perform_statistics(y_pred, y_test)
