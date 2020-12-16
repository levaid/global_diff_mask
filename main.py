import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pkbar
from torch.autograd import Variable
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_style('whitegrid')
import KDEpy
import numpy as np
import neptune
import os

neptune.init(project_qualified_name='cucuska2/diffmask',
             api_token=os.environ['NEPTUNE_API_TOKEN'],
             )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


BATCH_SIZE = 16

trainset = torchvision.datasets.CIFAR10(root='./.datasets', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./.datasets', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = MaskedConv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = MaskedConv2d(6, 16, 5)
        self.fc1 = MaskedLinear(16 * 5 * 5, 120)
        self.fc2 = MaskedLinear(120, 84)
        self.fc3 = MaskedLinear(84, 10)

    def forward(self, x):
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
            


net = Net().to(device)
net.set_learning_mode('mask')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

NUM_EPOCHS = 30
EPOCH_TO_CHANGE = 30



for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
    kbar = pkbar.Kbar(target=len(trainloader), epoch=epoch, num_epochs=NUM_EPOCHS, width=12, always_stateful=False)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        
        
        kbar.update(i, values=[("loss", loss)])

        if i % 200 == 0:
            currentiter = f'{i + epoch * len(trainloader):05}'
            plt.figure(facecolor='w')
            for name, param in net.named_modules():
                if type(param) in [MaskedConv2d, MaskedLinear]:
                    data = param.mask.detach().cpu().numpy().flatten()
                    x, y = KDEpy.FFTKDE(bw=0.1).fit(data).evaluate()
                    plt.plot(x, y, label=name)

            plt.title(f'Masks\' distribution in iter {currentiter}')
            plt.legend()
            plt.ylim((0, 2))
            plt.xlim((-4, 6))
            plt.savefig(f'run_details/iter_mask_{currentiter}.png', dpi=300)

            plt.figure(facecolor='w')
            for name, param in net.named_modules():
                if type(param) in [MaskedConv2d, MaskedLinear]:
                    data = param.weight.detach().cpu().numpy().flatten()
                    x, y = KDEpy.FFTKDE(bw=0.1).fit(data).evaluate()
                    plt.plot(x, y, label=name)

            plt.title(f'Weights in epoch {currentiter}')
            plt.legend()
            plt.savefig(f'run_details/iter_weight_{currentiter}.png', dpi=300)
            plt.close('all')


    if epoch == EPOCH_TO_CHANGE:
        net.set_learning_mode('weight')
    correct = 0
    total = 0
    val_loss = 0

    # viz

    # for name, param in net.named_modules():
    #     if type(param) in [MaskedConv2d, MaskedLinear]:
    #         for val, mask in zip(param.weight.detach().cpu().numpy().flatten(), 
    #                              param.mask.detach().cpu().numpy().flatten()):
    #             data.append([val, mask, name])
    #             layers.append(name)
    
    # df = pd.DataFrame(data, columns = ['weight', 'mask', 'layer'])


    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            val_loss += criterion(outputs, labels)/len(testloader)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    kbar.add(1, values=[("val_loss", val_loss), ("val_acc", correct/total)])


print('Finished Training')
