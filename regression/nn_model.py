import torch
import torch.nn.functional as F


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
