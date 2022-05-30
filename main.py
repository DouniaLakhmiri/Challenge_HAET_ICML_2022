import torch
import gc
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import os
import sys
from datahandler import *
from autoaugment import CIFAR10Policy, Cutout
from convmixer import ConvMixer

criterion = nn.CrossEntropyLoss()

import time



def train(epoch):
    # print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100. * correct / total
    return acc


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total

    return acc


gc.collect()
torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device == 'cuda':
    net = torch.nn.DataParallel(model)
    cudnn.benchmark = True

print('==> Preparing data..')

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(initial_image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),  # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# --------------------------------------------
# Dataset - Cifar10
# Plugin new dataset here
# --------------------------------------------

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

y_train = trainset.targets
y_test = testset.targets

subset_indices_1, subset_indices_test_1 = get_subset_data(y_train, y_test)
partial_trainset = torch.utils.data.Subset(trainset, subset_indices_1)
partial_testset = torch.utils.data.Subset(testset, subset_indices_test_1)

# --------------------------------------------

trainloader = torch.utils.data.DataLoader(
    partial_trainset, batch_size=256, num_workers=8, shuffle=True)

testloader = torch.utils.data.DataLoader(
    partial_testset, batch_size=128, shuffle=False)

##### Model #########

model=ConvMixer(256,8,patch_size=2,kernel_size=5,n_classes=10)
model.to(device)
#######################


# ------------------------------------------------
####### Optimizer ############
print('==> Defining the Optimizer and its hyperparameters..')
criterion = nn.CrossEntropyLoss()
base_optimizer=optim.SGD
optimizer = optim.SGD(model.parameters(), lr=0.057, momentum=0.98, weight_decay=0.0008,dampening=0.22)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=140, eta_min=0.01)
#################################

# pytorch_total_params = sum(p.numel() for p in model.parameters())
# print('nombre de prametres='+str(pytorch_total_params))

####### Training ##############

start_epoch = 0
training_accuracies = []
testing_accuracies = []
t0 = time.time()
execution_time = 0
total_epochs = 0
epoch = 0
best_test_acc = 0

while execution_time < 600:
    tr_acc = train(epoch)
    training_accuracies.append(tr_acc)
    te_acc = test(epoch)
    testing_accuracies.append(te_acc)
    if epoch <= 215:
        scheduler.step()

    if epoch == 230:
      for param_group in optimizer.param_groups:
        param_group['lr'] /= 10
        
    execution_time = time.time() - t0

    if te_acc > best_test_acc:
        best_test_acc = te_acc
        print('Saving checkpoint..')
        state = {
            'net': model.state_dict(),
            'acc': best_test_acc,
            'epoch': epoch,
        }
        torch.save(state, 'ckpt.pth')
    lr = scheduler.get_last_lr()[0]

    print(
        "Epoch {}, Execution time: {:.1f}, LR: {:.3f}, Train accuracy: {:.3f}, Val accuracy: {:.3f} "
            .format(epoch, execution_time, lr, tr_acc, best_test_acc))

    epoch += 1

print('Best valid acc', max(testing_accuracies))
print('Best train acc', max(training_accuracies))
