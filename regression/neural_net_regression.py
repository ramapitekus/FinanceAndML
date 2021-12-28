import torch
import torch.nn as nn
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import torch.nn.functional as F
from utils import perform_statistics
from sklearn.model_selection import train_test_split


INTERVALS = {1: "01-04-2013-19-07-2016", 2: "01-04-2013-01-01-2017", 3: "01-04-2013-31-12-2020"}
PERIODS = [1, 7, 30, 90]

# Adapted from https://datascience.stackexchange.com/questions/96271/logcoshloss-on-pytorch
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



class ANN(nn.Module):

    def __init__(self, n_features, interval):
        super(ANN, self).__init__()
        self.interval = interval

        if interval == 1:
            n_hidden = 128
        if interval == 2:
            n_hidden = 128
        if interval == 3:
            n_hidden = 300

        self.layer1 = torch.nn.Linear(n_features, n_hidden)
        self.generic_layer = torch.nn.Linear(n_hidden, n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden, 1)

    def forward(self, x):
        if self.interval == 2:
            x = self.layer1(x)
            x = F.relu(x)
            x = self.generic_layer(x)
            x = F.relu(x)
            x = self.layer3(x)
            return x

        if self.interval == 3:
            x = self.layer1(x)
            x = F.relu(x)
            x = self.generic_layer(x)
            x = F.relu(x)
            x = self.generic_layer(x)
            x = F.relu(x)
            x = self.layer3(x)
            return x

        if self.interval == 1:
            x = self.layer1(x)
            x = F.relu(x)
            x = self.generic_layer(x)
            x = F.relu(x)
            x = self.layer3(x)
            return x


def train_interval(interval: int):
    interval_str = INTERVALS[interval]
    for period in PERIODS:
        model_path = f"./models/ANN/interval_{interval}/model_interval_{interval}_period_{period}.pth"
        df = pd.read_csv(f"../data/features_selected/continuous/{interval_str}/{period}d_indicators.csv", index_col=False)

        y = df[f"bitcoin-price_raw_{period}d"]
        y = torch.tensor(y.values).float()[:-period]  # Get dependent variables
        
        x = df.drop(columns=[f"bitcoin-price_raw_{period}d", "Date"])  # Get features
        x = torch.tensor(x.values).float()[:-period]
    
        n_features = x.shape[1]

        min_max_scaler = MinMaxScaler()
        robust_scaler = RobustScaler()
        x = torch.tensor(min_max_scaler.fit_transform(x)).float()
        x = torch.tensor(robust_scaler.fit_transform(x)).float()

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2)


        train_loader = prepare_dataloader(X_train, y_train)

        net = ANN(n_features, interval)

        print(f"Training interval {interval} for {period} day(s) period")

        train(net, train_loader, X_val, y_val, model_path)

        evaluate(n_features, X_test, y_test, interval, model_path)


def prepare_dataloader(X_train, y_train):
    train_data = []
    for i in range(len(X_train)):
        train_data.append([X_train[i], y_train[i]])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)

    return train_loader


def train(net, train_loader, X_val, y_val, model_path):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    mse_loss = torch.nn.MSELoss()
    min_val_loss = 1e20

    for t in range(5000):
        total_loss = 0

        for sample in train_loader:
            prediction = torch.flatten(net(sample[0]))

            train_loss = mse_loss(prediction, sample[1])
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            total_loss += train_loss

        with torch.no_grad():
            y_pred = torch.flatten(net(X_val))
            val_loss = mse_loss(y_pred, y_val)
            if val_loss < min_val_loss:
                torch.save(net.state_dict(), model_path)

        print(f"\rEpoch {t + 1} validation loss is {int(total_loss)}", flush=True, end="")

    print("\n")


def evaluate(n_features, X_test, y_test, interval, path):
    model = ANN(n_features, interval)
    model.load_state_dict(torch.load(path))
    model.eval()

    y_pred = torch.flatten(model(X_test))

    y_pred = y_pred.detach().numpy()
    y_test = y_test.detach().numpy()

    perform_statistics(y_pred, y_test)


for interval in INTERVALS.keys():
    train_interval(interval)
