from torch import nn
from custom_modules import MaskedConv2d, MaskedLinear
from torch import functional as F
import torch


class ConvNet(nn.Module):
    def __init__(self, forward_type='simple', sigmoid=False):
        super(ConvNet, self).__init__()
        self.conv1 = MaskedConv2d(3, 6, 5, sigmoid=sigmoid)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = MaskedConv2d(6, 16, 5, sigmoid=sigmoid)
        self.fc1 = MaskedLinear(16 * 5 * 5, 120, sigmoid=sigmoid)
        self.fc2 = MaskedLinear(120, 84, sigmoid=sigmoid)
        self.fc3 = MaskedLinear(84, 10, sigmoid=sigmoid)

        if forward_type == 'simple':
            self.forward = self.forward_simple
        elif forward_type == 'flatten':
            self.forward = self.forward_flatten
        else:
            raise NotImplementedError

    def forward_simple(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def forward_flatten(self, x):
        activations = []
        x = self.conv1(x)
        activations.append(torch.flatten(x, start_dim=1))
        x = F.relu(x)
        x = self.pool(x)
        activations.append(torch.flatten(x, start_dim=1))
        x = self.conv2(x)
        activations.append(torch.flatten(x, start_dim=1))
        x = F.relu(x)
        x = self.pool(x)
        activations.append(torch.flatten(x, start_dim=1))
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        activations.append(torch.flatten(x, start_dim=1))
        x = F.relu(x)
        x = self.fc2(x)
        activations.append(torch.flatten(x, start_dim=1))
        x = F.relu(x)
        x = self.fc3(x)
        activations.append(torch.flatten(x, start_dim=1))
        return x, torch.cat(activations, dim=1)

    def set_learning_mode(self, mode: str):
        flow = (None, None)
        if mode == 'mask':
            flow = (False, True)
        elif mode == 'weight':
            flow = (True, False)
        elif mode == 'both':
            flow = (True, True)

        for name, param in self.named_modules():
            if type(param) in [MaskedConv2d, MaskedLinear]:
                param.set_gradient_flow(*flow)
                print(f'{name} set to {mode}')


class PrunerNetFlat(nn.Module):
    def __init__(self, inputsize):
        super(PrunerNetFlat, self).__init__()
        self.fc1 = nn.Linear(inputsize, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return(x)
