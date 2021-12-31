import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import torch.nn.functional as F
from utils import perform_statistics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GroupKFold
from utils import *
from neural_net import ANN

INTERVALS = {1: "01-04-2013-19-07-2016",
             2: "01-04-2013-01-01-2017",
             3: "01-04-2013-31-12-2020"}

PERIODS = [1, 7, 30, 90]
N_NETS = 5


def train_interval(interval: int) -> None:
    for period in PERIODS:

        interval_str = INTERVALS[period]
        x, y = prepare_data_NN(interval_str, period)
        n_features = x.shape[1]
        kf = GroupKFold(n_splits=N_NETS)

        dataset_dict = {net_nr: {"train": (x[train_index], y[train_index]), "test": (x[test_index], y[test_index])} for net_nr, (train_index, test_index) in enumerate(kf.split(x))}

        for net_nr in range(N_NETS):
            dataset = dataset_dict[net_nr]
            X_train, y_train = dataset["train"]
            X_test, y_test = dataset["test"]

            train_loader = prepare_dataloader(X_train, y_train)

            net = ANN(n_features, interval)

            model_path = f"./models/SANN/interval_{interval}/model_net_nr_{net_nr}interval_{interval}_period_{period}.pth"
            print(f"Training interval {interval} for {period} day(s) period")

            train(net, train_loader, X_val, y_val, model_path, net_nr)
            evaluate(n_features, X_test, y_test, interval, model_path)

train_interval(1)