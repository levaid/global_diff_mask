from visualization import plot_mask_and_weight
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
from custom_modules import MaskedConv2d, MaskedLinear
import time



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



class MainNet(nn.Module):
    def __init__(self, forward_type='simple'):
        super(MainNet, self).__init__()
        self.conv1 = MaskedConv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = MaskedConv2d(6, 16, 5)
        self.fc1 = MaskedLinear(16 * 5 * 5, 120)
        self.fc2 = MaskedLinear(120, 84)
        self.fc3 = MaskedLinear(84, 10)

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
            

PARAMS = {'num_epochs': 30, 'initial_mode': 'weight', 'epoch_to_change': 50, 'lr': 0.0001}

net = MainNet().to(device)

net.set_learning_mode(PARAMS['initial_mode'])

neptune.create_experiment(name = time.strftime('%Y-%m-%d_%H:%M:%S'),
                          description='',
                          params=PARAMS,
                          )

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=PARAMS['lr'])


for epoch in range(PARAMS['num_epochs']):  # loop over the dataset multiple times
    kbar = pkbar.Kbar(target=len(trainloader), epoch=epoch, num_epochs=PARAMS['num_epochs'], width=12, always_stateful=False)
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
        
        neptune.log_metric('train_loss', loss.item())
        
        kbar.update(i, values=[("loss", loss)])

        current_iteration = i + epoch * len(trainloader)
        if current_iteration % 200 == 0:
            plot_mask_and_weight(net, current_iteration)


    if epoch == PARAMS['epoch_to_change']:
        net.set_learning_mode('weight')
    correct = 0
    total = 0
    val_loss = 0


    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            val_loss += criterion(outputs, labels)/len(testloader)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            neptune.log_metric('valid_loss', val_loss.item())
            
    neptune.log_metric('valid_acc', correct/total)

    kbar.add(1, values=[("val_loss", val_loss), ("val_acc", correct/total)])


print('Finished Training')
