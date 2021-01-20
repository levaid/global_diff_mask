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

    def discretize_with_quantile(self, quantile: float, how: str = 'from_mask'):
        assert 0 <= quantile <= 1, 'Quantile must be between 0 and 1.'
        if how == 'from_mask':
            threshold = torch.quantile(torch.abs(self.mask), q=quantile)
            self.mask.data = (torch.abs(self.mask) > threshold).type_as(self.mask)
        elif how == 'from_weight':
            threshold = torch.quantile(torch.abs(self.weight), q=quantile)
            self.mask.data = (torch.abs(self.weight) > threshold).type_as(self.mask)
        else:
            raise(NotImplementedError, 'you have to choose either `from_mask` or `from_weight`')

    def discretize_with_threshold(self, threshold: float, how: str = 'from_mask'):
        if how == 'from_mask':
            # print((torch.sum(torch.abs(self.mask) > threshold)) / torch.prod(torch.tensor(self.mask.size())))
            self.mask.data = (torch.abs(self.mask) > threshold).type_as(self.mask)
        elif how == 'from_weight':
            self.mask.data = (torch.abs(self.weight) > threshold).type_as(self.mask)
        else:
            raise(NotImplementedError, 'you have to choose either `from_mask` or `from_weight`')


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

    def discretize_with_quantile(self, quantile: float, how: str = 'from_mask'):
        assert 0 <= quantile <= 1, 'Quantile must be between 0 and 1.'
        if how == 'from_mask':
            threshold = torch.quantile(torch.abs(self.mask), q=quantile)
            self.mask.data = (torch.abs(self.mask) > threshold).type_as(self.mask)
        elif how == 'from_weight':
            threshold = torch.quantile(torch.abs(self.weight), q=quantile)
            self.mask.data = (torch.abs(self.weight) > threshold).type_as(self.mask)
        else:
            raise(NotImplementedError, 'you have to choose either `from_mask` or `from_weight`')

    def discretize_with_threshold(self, threshold: float, how: str = 'from_mask'):
        if how == 'from_mask':
            # print((torch.sum(torch.abs(self.mask) > threshold)) / torch.prod(torch.tensor(self.mask.size())))
            self.mask.data = (torch.abs(self.mask) > threshold).type_as(self.mask)
        elif how == 'from_weight':
            self.mask.data = (torch.abs(self.weight) > threshold).type_as(self.mask)
        else:
            raise(NotImplementedError, 'you have to choose either `from_mask` or `from_weight`')
