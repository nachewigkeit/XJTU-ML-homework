import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, hidden=30, layer=1):
        super(MLP, self).__init__()
        model = [nn.Linear(784, hidden), nn.ReLU()]
        for i in range(layer - 1):
            model += [nn.Linear(hidden, hidden), nn.ReLU()]
        model.append(nn.Linear(hidden, 10))
        self.seq = nn.Sequential(*model)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.seq(x)
        output = F.log_softmax(x, dim=1)
        return output
