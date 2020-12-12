from torch import nn
import torch

from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator


@variational_estimator
class BayesLSTM(nn.Module):
    def __init__(self, input_size=672, hidden_size=100, output_size=672):
        # TODO add more parameters so we can configure this
        super().__init__()
        self.hidden_layer_size = hidden_size

        self.lstm = BayesianLSTM(input_size, hidden_size)

        self.linear = nn.Linear(hidden_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, hidden_size),
                            torch.zeros(1, 1, hidden_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1))#, self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# @variational_estimator
# class NN(nn.Module):
#     def __init__(self):
#         super(NN, self).__init__()
#         self.lstm_1 = BayesianLSTM(1, 10)
#         self.linear = nn.Linear(10, 1)
            
#     def forward(self, x):
#         x_, _ = self.lstm_1(x)
        
#         #gathering only the latent end-of-sequence for the linear layer
#         x_ = x_[:, -1, :]
#         x_ = self.linear(x_)
#         return x_