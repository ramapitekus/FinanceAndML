import torch
import torch.nn.functional as F


class ANN(torch.nn.Module):

    def __init__(self, n_features, interval, dropout=None):
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
        if dropout is not None:
            self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.dropout(x) if self.dropout else x
        x = self.generic_layer(x)
        x = F.relu(x)
        x = self.dropout(x) if self.dropout else x
        # If 3. interval, add additional layer
        if self.interval == 3:
            x = self.generic_layer(x)
            x = F.relu(x)
            x = self.dropout(x) if self.dropout else x
        x = self.layer3(x)
        return x
