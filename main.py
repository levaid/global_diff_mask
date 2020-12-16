from visualization import plot_mask_and_weight
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pkbar
import numpy as np
import neptune
import os
from custom_modules import MaskedConv2d, MaskedLinear
import time
import itertools



# neptune.init(project_qualified_name='cucuska2/diffmask',
#              api_token=os.environ['NEPTUNE_API_TOKEN'],
#              )


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
            
class PrunerNet(nn.Module):
    def __init__(self):
        super(PrunerNet, self).__init__()
        self.fc1 = nn.Linear(8094, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return(x)

PARAMS = {'num_epochs': 30, 'initial_mode': 'mask', 'epoch_to_change': 10, 'lr_net': 0.0001, 'lr_mask': 0.0001}

net = MainNet(forward_type='flatten').to(device)
pruner_net = PrunerNet().to(device)

net.set_learning_mode(PARAMS['initial_mode'])

# neptune.create_experiment(name = time.strftime('%Y-%m-%d_%H:%M:%S'),
#                           description='still under dev',
#                           params=PARAMS,
#                           )

mask_parameters = [p for n, p in net.named_parameters() if 'mask' in n]

criterion = nn.CrossEntropyLoss()
optimizer_main = optim.Adam(net.parameters(), lr=PARAMS['lr_net'])
# we are chaining the pruner parameters to modify the main network's mask
optimizer_pruner = optim.Adam(itertools.chain(pruner_net.parameters(), mask_parameters), 
                                              lr=PARAMS['lr_mask'])


for epoch in range(PARAMS['num_epochs']):  # loop over the dataset multiple times
    kbar = pkbar.Kbar(target=len(trainloader), epoch=epoch, num_epochs=PARAMS['num_epochs'], width=12, always_stateful=False)
    running_loss_main = 0.0
    running_loss_pruner = 0.0
    loss_pruner = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer_main.zero_grad()
        optimizer_pruner.zero_grad()
        # forward + backward + optimize
        outputs, activations = net(inputs)
        loss_main = criterion(outputs, labels)

        # we step with the pruner net if we are before the midpoint
        if epoch < PARAMS['epoch_to_change']:
            outputs_by_activations = pruner_net(activations)
            loss_pruner = criterion(outputs_by_activations, labels)
            loss_pruner.backward()

            optimizer_pruner.step()
        
            # print statistics
            running_loss_pruner += loss_pruner.item()
        # else we step on the main network with the masks hopefully frozen
        else:
            loss_main = criterion(outputs, labels)
            loss_main.backward()
            optimizer_main.step()

        # neptune.log_metric('train_loss', loss.item())
        
        kbar.update(i, values=[("loss_main", loss_main), ('loss_pruner', loss_pruner)])

        current_iteration = i + epoch * len(trainloader)
        if current_iteration % 500 == 0:
            plot_mask_and_weight(net, current_iteration)


    if epoch == PARAMS['epoch_to_change'] - 1:
        net.set_learning_mode('weight')
        
    correct_main = 0
    total = 0
    val_loss_main = 0

    correct_pruner = 0
    val_loss_pruner = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs, activations = net(images)
            loss_main = criterion(outputs, labels)
            
            
            val_loss_main += loss_main/len(testloader)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_main += (predicted == labels).sum().item()

            outputs_pruner = pruner_net(activations)
            loss_pruner = criterion(outputs_pruner, labels)
            val_loss_pruner += loss_pruner/len(testloader)
            _, predicted_pruner = torch.max(outputs_pruner.data, 1)
            correct_pruner += (predicted_pruner == labels).sum().item()
            # neptune.log_metric('valid_loss', loss.item())
            
    # neptune.log_metric('valid_acc', correct/total)

    kbar.add(1, values=[("val_loss_main", val_loss_main), ("val_acc_main", correct_main/total), ('val_loss_pruner', val_loss_pruner), ('val_acc_pruner', correct_pruner/total)])


print('Finished Training')
