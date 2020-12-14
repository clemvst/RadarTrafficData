from torch import nn
import torch

from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator


@variational_estimator
class BayesLSTM(nn.Module):
    """
    Function inspired (BUT WITH MANY CHANGES) by Piero Esposito's functions for Bayesian Model
    
    https://towardsdatascience.com/bayesian-lstm-on-pytorch-with-blitz-a-pytorch-bayesian-deep-learning-library-5e1fec432ad3
    """

    def __init__(self, input_size=672, hidden_size=100, output_size=672):
        super().__init__()
        self.hidden_layer_size = hidden_size

        self.lstm = BayesianLSTM(input_size, hidden_size)

        self.linear = nn.Linear(hidden_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, hidden_size),
                            torch.zeros(1, 1, hidden_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
