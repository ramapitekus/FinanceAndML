from torch.utils import data
from utils import prepare_dataset
from nn_solver import train_model, evaluate
from ann_model import ANN
from lstm_model import LSTM


INTERVALS = {1: "01-04-2013-19-07-2016", 2: "01-04-2013-01-01-2017", 3: "01-04-2013-31-12-2020"}
PERIODS = [1, 7, 30, 90]


def train_interval(interval: int, nn_type: str, epochs: int, loss: str, dropout=None) -> None:
    for period in PERIODS:
        interval_str = INTERVALS[interval]
        train_dataset, test_dataset, val_dataset = prepare_dataset(interval_str, period, nn_type)

        n_features = train_dataset.x.shape[1]

        if nn_type == "LSTM":
            net = LSTM(n_features, hidden_units=320)

        elif nn_type == "ANN":
            net = ANN(n_features, interval)

        elif nn_type == "dropout":
            net = ANN(n_features, interval, dropout)

        train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        test_loader = data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0)

        print(f"Training {nn_type} network, interval {interval} for {period} day(s) period")
        model_path = f"./regression/models/{nn_type}/interval_{interval}/model_interval_{interval}_period_{period}.pth"

        train_model(net, train_loader, val_loader, model_path, epochs=epochs, loss_type=loss)
        evaluate(n_features, test_loader, interval, model_path, nn_type)


for interval in INTERVALS.keys():
    train_interval(interval, nn_type="LSTM", loss="mse", epochs=1200)
