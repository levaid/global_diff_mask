import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable


class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, sigmoid: bool = False):

        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size)

        if sigmoid:
            self.mask = nn.parameter.Parameter(Variable(torch.zeros(self.weight.shape)), requires_grad=True)
            self.forward = self.forward_sigmoid
        else:
            self.mask = nn.parameter.Parameter(Variable(torch.ones(self.weight.shape)), requires_grad=True)
            self.forward = self.forward_simple

    def forward_simple(self, x: Tensor) -> Tensor:

        x = F.conv2d(x, torch.mul(self.weight, self.mask), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return(x)

    def forward_sigmoid(self, x: Tensor) -> Tensor:

        x = F.conv2d(x, torch.mul(self.weight, torch.sigmoid(self.mask)), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return(x)

    def set_gradient_flow(self, weight_grad: bool, mask_grad: bool):

        self.mask.requires_grad = mask_grad
        self.weight.requires_grad = weight_grad


class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, sigmoid: bool = False):

        super(MaskedLinear, self).__init__(in_features, out_features)

        if sigmoid:
            self.mask = nn.parameter.Parameter(Variable(torch.zeros(self.weight.shape)), requires_grad=True)
            self.forward = self.forward_sigmoid
        else:
            self.mask = nn.parameter.Parameter(Variable(torch.ones(self.weight.shape)), requires_grad=True)
            self.forward = self.forward_simple

    def forward_simple(self, x: Tensor) -> Tensor:

        x = F.linear(x, torch.mul(self.weight, self.mask), self.bias)
        return(x)

    def forward_sigmoid(self, x: Tensor) -> Tensor:

        x = F.linear(x, torch.mul(self.weight, torch.sigmoid(self.mask)), self.bias)
        return(x)

    def set_gradient_flow(self, weight_grad: bool, mask_grad: bool):

        self.mask.requires_grad = mask_grad
        self.weight.requires_grad = weight_grad
