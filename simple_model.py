from torch import nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size=576, hidden_size=100, output_size=96):
        # TODO add more parameters so we can configure this
        super().__init__()
        self.hidden_layer_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)

        self.linear = nn.Linear(hidden_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, hidden_size),
                            torch.zeros(1, 1, hidden_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

    def predict(self,input_batch,target_len):
        return self.forward(input_batch)

