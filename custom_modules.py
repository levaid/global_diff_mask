import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable

class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size):

        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size)

        self.mask = nn.parameter.Parameter(Variable(torch.ones(self.weight.shape)), requires_grad=True)
        
        
    def forward(self, x: Tensor) -> Tensor:
        
        x = F.conv2d(x, torch.mul(self.weight, self.mask), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return(x)

    def set_gradient_flow(self, weight_grad: bool, mask_grad: bool):
        
        self.mask.requires_grad = mask_grad
        self.weight.requires_grad = weight_grad

class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):

        super(MaskedLinear, self).__init__(in_features, out_features)

        self.mask = nn.parameter.Parameter(Variable(torch.ones(self.weight.shape)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:

        x = F.linear(x, torch.mul(self.weight, self.mask), self.bias)
        return(x)

    def set_gradient_flow(self, weight_grad: bool, mask_grad: bool):
        
        self.mask.requires_grad = mask_grad
        self.weight.requires_grad = weight_grad
