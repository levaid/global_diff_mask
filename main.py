from visualization import plot_mask_and_weight, plot_pruner
from local_datasets import create_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pkbar
import os
from custom_modules import MaskedConv2d, MaskedLinear
import time
import glob
import neptune
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num_epochs', required=True, type=int, help='Number of total epochs.')
parser.add_argument('--initial_mode', required=True, type=str, help='Inital state of the network. Can be mask, weight or both.')
parser.add_argument('--epoch_to_change', required=True, type=int, help='Epoch in which to change behavior from mask to weight or vica-versa.')
parser.add_argument('--lr_net', default=0.0005, type=float, help='Learning rate for the original network.')
parser.add_argument('--lr_mask', default=0.001, type=float, help='Learning rate of the mask.')
parser.add_argument('--lr_pruner', default=0.0005, type=float, help='Learning rate of the pruner network.')
parser.add_argument('--sigmoid', default=False, type=bool, help='Whether to sigmoid the masks. If true, then the masks are initialized at 0, else at 1.')
parser.add_argument('--neptune', default=False, type=bool, help='Use Neptune.')


args = vars(parser.parse_args())

if args['neptune']:
    neptune.init(project_qualified_name='cucuska2/diffmask',
                 api_token=os.environ['NEPTUNE_API_TOKEN'],
                 )

BATCH_SIZE = 128

trainloader, testloader = create_dataloaders(batch_size=BATCH_SIZE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"


class MainNet(nn.Module):
    def __init__(self, forward_type='simple', sigmoid=False):
        super(MainNet, self).__init__()
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


net = MainNet(forward_type='flatten', sigmoid=args['sigmoid']).to(device)

pruner_net = PrunerNetFlat(inputsize=8094).to(device)

net.set_learning_mode(args['initial_mode'])

if args['neptune']:
    neptune.create_experiment(name=time.strftime('%Y-%m-%d_%H:%M:%S'),
                              description='last layer tests',
                              params=args,
                              )

mask_parameters = [p for n, p in net.named_parameters() if 'mask' in n]

criterion = nn.CrossEntropyLoss()

optimizer_main = optim.Adam(net.parameters(), lr=args['lr_net'])
# we are training masks with a separate lr because they learn SLOOOOOWLY
optimizer_mask = optim.Adam(mask_parameters, lr=args['lr_mask'])
optimizer_pruner = optim.Adam(pruner_net.parameters(), lr=args['lr_pruner'])

for image in glob.glob('run_details/*.png'):
    os.remove(image)

for epoch in range(args['num_epochs']):  # loop over the dataset multiple times
    kbar = pkbar.Kbar(target=len(trainloader), epoch=epoch, num_epochs=args['num_epochs'], width=12, always_stateful=False)
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
        outputs_by_activations = pruner_net(activations)
        loss_pruner = criterion(outputs_by_activations, labels)
        # we step with the pruner net if we are before the midpoint
        if epoch < args['epoch_to_change']:

            loss_pruner.backward()
            optimizer_pruner.step()
            optimizer_mask.step()

            # print statistics
            running_loss_pruner += loss_pruner.item()
        # else we step on the main network with the masks frozen
        else:
            loss_main.backward()
            optimizer_main.step()

        if args['neptune']:
            neptune.log_metric('train_loss_main', loss_main.item())
            neptune.log_metric('train_loss_pruner', loss_pruner)

        kbar.update(i, values=[("loss_main", loss_main), ('loss_pruner', loss_pruner)])

        current_iteration = i + epoch * len(trainloader)
        if current_iteration % 50 == 0:
            plot_mask_and_weight(net, current_iteration)
            plot_pruner(pruner_net, current_iteration)

    if epoch == args['epoch_to_change'] - 1:
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

            if args['neptune']:
                neptune.log_metric('valid_loss_main', loss_main.item())
                neptune.log_metric('valid_loss_pruner', loss_pruner.item())

    if args['neptune']:
        neptune.log_metric('valid_acc_main', correct_main/total)
        neptune.log_metric('valid_acc_pruner', correct_pruner/total)

    kbar.add(1, values=[("val_loss_main", val_loss_main), ("val_acc_main", correct_main/total),
                        ('val_loss_pruner', val_loss_pruner), ('val_acc_pruner', correct_pruner/total)])

neptune.stop()
print('Finished Training')
