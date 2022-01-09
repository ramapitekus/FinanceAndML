import torch


class Classification_LSTM(torch.nn.Module):
    def __init__(self, n_features, hidden_units, interval):
        super().__init__()
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.num_layers = 2 if interval == 3 else 1
        self.dropout = 0 if self.num_layers == 1 else 0.5

        self.lstm = torch.nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )

        self.linear = torch.nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_()
        c0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))

        out = self.linear(hn[0])

        return torch.sigmoid(out)
