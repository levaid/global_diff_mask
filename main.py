from visualization import plot_mask_and_weight, plot_pruner
from local_datasets import create_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim
import pkbar
import os
import time
import glob
import neptune
import argparse
from networks import ConvNet, PrunerNetFlat


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num_first_stage', required=True, type=int, help='Number of total epochs.')
parser.add_argument('--initial_mode', required=True, type=str, help='Inital state of the network. Can be `mask`, `weight` or `both`.')
parser.add_argument('--num_second_stage', required=True, type=int, help='Epoch in which to change behavior from mask to weight or vica-versa.')
parser.add_argument('--discretize', default=False, type=str2bool, help='Whether to discretize when learning with the mask.')
parser.add_argument('--discretization_method', default='from_weight', type=str, help='Discretization method. Can be `from_mask` and `from_weight`.')
parser.add_argument('--discretization_quantile', default=0.5, type=float, help='Quantile for discretization.')
parser.add_argument('--lr_net', default=0.0005, type=float, help='Learning rate for the original network.')
parser.add_argument('--lr_mask', default=0.001, type=float, help='Learning rate of the mask.')
parser.add_argument('--lr_pruner', default=0.0005, type=float, help='Learning rate of the pruner network.')
parser.add_argument('--sigmoid', default=False, type=str2bool, help='Whether to sigmoid the masks. If true, then the masks are initialized at 0, else at 1.')
parser.add_argument('--neptune', default=False, type=str2bool, help='Use Neptune.')
parser.add_argument('--visualize', default=False, type=str2bool, help='Whether to create distribution plots every 50 batches.')


args = vars(parser.parse_args())

print(args)

if args['neptune']:
    neptune.init(project_qualified_name='cucuska2/diffmask',
                 api_token=os.environ['NEPTUNE_API_TOKEN'],
                 )

BATCH_SIZE = 128

trainloader, testloader = create_dataloaders(batch_size=BATCH_SIZE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = ConvNet(forward_type='flatten', sigmoid=args['sigmoid']).to(device)

# magic number coming from the parameter count of convnet
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


current_mode = args['initial_mode']

for epoch in range(args['num_first_stage'] + args['num_second_stage']):  # loop over the dataset multiple times
    kbar = pkbar.Kbar(target=len(trainloader), epoch=epoch, num_epochs=args['num_first_stage'] + args['num_second_stage'],
                      width=12, always_stateful=False)
    running_loss_main = 0.0
    running_loss_pruner = 0.0
    loss_pruner = 0.0

    if current_mode not in {'weight', 'mask', 'both'}:
        raise(ValueError, 'current_mode must be mask, weight or both')

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
        if args['neptune']:
            neptune.log_metric('mean_abs_activation', torch.mean(torch.abs(activations)))
        # we step with the pruner net if we are before the midpoint
        if current_mode == 'mask' or current_mode == 'both':

            loss_pruner.backward()
            optimizer_pruner.step()
            optimizer_mask.step()

            # print statistics
            running_loss_pruner += loss_pruner.item()
        # else we step on the main network with the masks frozen
        if current_mode == 'weight' or current_mode == 'both':
            loss_main.backward()
            optimizer_main.step()

        if args['neptune']:
            neptune.log_metric('train_loss_main', loss_main.item())
            neptune.log_metric('train_loss_pruner', loss_pruner)

        kbar.update(i, values=[("loss_main", loss_main), ('loss_pruner', loss_pruner)])

        current_iteration = i + epoch * len(trainloader)

        if current_iteration % 50 == 0 and args['visualize']:
            plot_mask_and_weight(net, current_iteration)
            plot_pruner(pruner_net, current_iteration)

    if epoch == args['num_first_stage'] - 1:
        if current_mode == 'mask':
            net.set_learning_mode('weight')
            current_mode = 'weight'
            if args['discretize']:
                net.discretize_layerwise_locally(args['discretization_quantile'], args['discretization_method'])
                print('stop')
        elif current_mode == 'weight':
            if args['discretize']:
                net.discretize_layerwise_locally(args['discretization_quantile'], args['discretization_method'])

            # net.set_learning_mode('mask')
            # current_mode = 'mask'

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

if args['neptune']:
    neptune.stop()
print('Finished Training')
