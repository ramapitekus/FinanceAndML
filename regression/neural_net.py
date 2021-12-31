import torch
import torch.nn.functional as F
from utils import *


class ANN(torch.nn.Module):

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


def train(net: ANN, train_loader, X_val, y_val, model_path):
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


def evaluate(n_features, X_test, y_test, interval, model_path):
    model = ANN(n_features, interval)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_pred = torch.flatten(model(X_test))

    y_pred = y_pred.detach().numpy()
    y_test = y_test.detach().numpy()

    perform_statistics(y_pred, y_test)
