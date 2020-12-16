import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pkbar
from torch.autograd import Variable

x = Variable(torch.Tensor([2]), requires_grad=True)
m = Variable(torch.Tensor([1]), requires_grad=True)

y = x * m

y.backward()

print(x.grad, m.grad)