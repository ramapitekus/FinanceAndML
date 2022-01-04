import torch


class LSTM(torch.nn.Module):
    def __init__(self, n_features, hidden_units):
        super().__init__()
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.num_layers = 2

        self.lstm = torch.nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=0.5
        )

        self.linear = torch.nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))

        out = self.linear(hn[0]).flatten()

        return out
