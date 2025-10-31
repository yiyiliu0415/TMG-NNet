import torch
import torch.nn as nn


class TMGnet(nn.Module):
    def __init__(self, input_size, output_size):
        super(TMGnet, self).__init__()
        self.hidden = 4096
        self.fc_real = nn.Sequential(nn.Linear(input_size, self.hidden), nn.ReLU(),
                                     # nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                                     nn.Linear(self.hidden, output_size))
        self.fc_imag = nn.Sequential(nn.Linear(input_size, self.hidden), nn.ReLU(),
                                     # nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                                     nn.Linear(self.hidden, output_size))

    def forward(self, x):
        x_real = self.fc_real(x)
        x_imag = self.fc_imag(x)
        return x_real, x_imag
