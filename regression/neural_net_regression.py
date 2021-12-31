import torch
import math
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from utils import prepare_data_NN, prepare_dataloader
from neural_net import ANN, train, evaluate


INTERVALS = {1: "01-04-2013-19-07-2016", 2: "01-04-2013-01-01-2017", 3: "01-04-2013-31-12-2020"}
PERIODS = [1, 7, 30, 90]


def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


def train_interval(interval: int) -> None:
    for period in PERIODS:
        interval_str = INTERVALS[interval]
        x, y = prepare_data_NN(interval_str, period)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

        n_features = X_train.shape[1]

        train_loader = prepare_dataloader(X_train, y_train)

        net = ANN(n_features, interval)

        print(f"Training interval {interval} for {period} day(s) period")
        model_path = f"./regression/models/ANN/interval_{interval}/model_interval_{interval}_period_{period}.pth"

        train(net, train_loader, X_val, y_val, model_path)
        evaluate(n_features, X_test, y_test, interval, model_path)


for interval in INTERVALS.keys():
    train_interval(interval)
