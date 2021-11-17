import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, hidden=1000, layer=1, rate=0.3):
        super(MLP, self).__init__()
        model = [nn.Linear(784, hidden), nn.ReLU(), nn.Dropout(rate)]
        for i in range(layer - 1):
            model += [nn.Linear(hidden, hidden), nn.ReLU()]
        model.append(nn.Linear(hidden, 10))
        self.seq = nn.Sequential(*model)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.seq(x)
        output = F.log_softmax(x, dim=1)
        return output


class Conv(nn.Module):
    def __init__(self, channel=16, layer=1):
        super(Conv, self).__init__()
        model = [nn.Conv2d(1, channel, (3, 3)), nn.ReLU()]
        for i in range(layer - 1):
            model += [nn.Conv2d(channel, channel, (3, 3)), nn.ReLU()]
        model += [nn.Conv2d(channel, 10, (3, 3)), nn.AdaptiveAvgPool2d((1, 1))]
        self.seq = nn.Sequential(*model)

    def forward(self, x):
        x = self.seq(x)
        x = torch.flatten(x, 1)
        output = F.log_softmax(x, dim=1)
        return output
