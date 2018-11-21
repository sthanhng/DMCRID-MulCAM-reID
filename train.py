# ----------------------------------------------------------
#
# Project name: DMCRID-MulCAM-reID
# File name: train.py
#
# Last modified: 2018-11-21
#
# ----------------------------------------------------------


from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import time
import os
import json

from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms

from model import ResNet50, PCB
from shutil import copyfile

from utils import draw_curve, save_model

######################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--model-path', default='models', type=str,
                    help='path to the model')
parser.add_argument('--model-name', default='reid-ResNet50', type=str,
                    help='output model name')
parser.add_argument('--data-dir', default='./datasets/Market-1501/market-reid',
                    type=str,
                    help='training dir path')
parser.add_argument('--train-all', action='store_true',
                    help='use all training data')
parser.add_argument('--color-jitter', action='store_true',
                    help='use color jitter in training')
parser.add_argument('--batch-size', default=32, type=int, help='batch-size')
parser.add_argument('--num-epochs', default=30, type=int,
                    help='number of epochs in training')
parser.add_argument('--num-workers', default=8, type=int, help='num-workers')
parser.add_argument('--lr-decay-epoch', default=40, type=int,
                    help='decay lr after lr-decay-epoch')
parser.add_argument('--PCB', action='store_true', default=False,
                    help='use PCB-based model')
args = parser.parse_args()


######################################################################

model_path_full = os.path.join(args.model_path, args.model_name)
if not os.path.exists(model_path_full):
    os.makedirs(model_path_full)

# save arguments
with open('{}/args.json'.format(model_path_full), 'w') as fp:
    json.dump(vars(args), fp, indent=4)


# -------------------------------------------------------------------
#
# Load Data
#
# -------------------------------------------------------------------

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Data transforms
transform_train_list = [
    transforms.RandomResizedCrop(size=128,
                                 scale=(0.75, 1.0),
                                 ratio=(0.75, 1.3333),
                                 interpolation=3),
    transforms.Resize((288, 144), interpolation=3),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
]

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
]

if args.PCB:
    transform_train_list = [
        transforms.Resize((384, 192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    transform_val_list = [
        transforms.Resize(size=(384, 192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

# Color jitter
if args.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                                   saturation=0.1, hue=0)] + \
                           transform_train_list

print(transform_train_list)

data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

######################################################################

train_all = ''
if args.train_all:
    train_all = '_all'

image_dsets = dict()
image_dsets['train'] = datasets.ImageFolder(os.path.join(args.data_dir, 'train' + train_all),
                                            data_transforms['train'])
image_dsets['val'] = datasets.ImageFolder(os.path.join(args.data_dir, 'val'),
                                          data_transforms['val'])

dset_loaders = {
    x: torch.utils.data.DataLoader(image_dsets[x], batch_size=args.batch_size,
                                   shuffle=(x != 'val'), num_workers=args.num_workers)
    # 8 workers may work faster
    for x in ['train', 'val']
}

dset_sizes = {x: len(image_dsets[x]) for x in ['train', 'val']}
class_names = image_dsets['train'].classes

use_gpu = torch.cuda.is_available()

inputs, classes = next(iter(dset_loaders['train']))


######################################################################


# --------------------------------------------------------------------
#
# Training the model
#
# --------------------------------------------------------------------

# The loss history
y_loss = dict()
y_loss['train'] = []
y_loss['val'] = []

# The error history
y_err = dict()
y_err['train'] = []
y_err['val'] = []


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    A general function to train the defined model. Here, we will illustrate:

    - Scheduling the learning rate
    - Saving the last model

    The parameter `scheduler` is an LR scheduler object from `torch.optim.lr_scheduler`

    :param model: The defined model
    :param criterion:
    :param optimizer: Optimizer
    :param scheduler: LR scheduler
    :param num_epochs: Number of epochs
    :return: The trained model
    """

    start_time = time.time()

    for epoch in range(num_epochs):
        print('#' * 70)
        print('==> Training Epoch {}/{}'.format(epoch + 1, num_epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < args.batch_size:  # skip the last batch
                    continue

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                #########################################################
                # forward
                outputs = model(inputs)
                if not args.PCB:
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)
                else:
                    part = dict()
                    smax = nn.Softmax(dim=1)
                    num_part = 6
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = smax(part[0]) + smax(part[1]) + smax(part[2]) + smax(
                        part[3]) + smax(part[4]) + smax(part[5])
                    _, preds = torch.max(score.data, 1)

                    loss = criterion(part[0], labels)
                    for i in range(num_part - 1):
                        loss += criterion(part[i + 1], labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                #########################################################
                # statistics
                running_loss += loss.item() * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('[{}] Phase | Loss: {:.4f}\tAccuracy: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)

            #########################################################
            # save the model
            if phase == 'val':
                last_model = model.state_dict()
                if epoch + 1 % 10 == 0:
                    save_model(model, epoch, model_path_full)
                draw_curve(epoch, y_loss, y_err, args.model_name)

    time_elapsed = time.time() - start_time
    print('[i] training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                            time_elapsed % 60))

    # load last model weights
    model.load_state_dict(last_model)
    save_model(model, 'last', model_path_full)

    # debug the model structure
    print(model)

    return model


######################################################################


if __name__ == '__main__':
    # --------------------------------------------------------------------
    #
    # Fine-tuning the model
    #
    # Description:
    #       Load a pretrainied model and reset final fully connected layer
    #
    # --------------------------------------------------------------------

    if args.PCB:
        model = PCB(len(class_names))
    else:
        model = ResNet50(len(class_names))

    print(model)

    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    if not args.PCB:
        ignored_params = list(map(id, model.model.fc.parameters())) + list(
            map(id, model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                             model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.01},
            {'params': model.model.fc.parameters(), 'lr': 0.1},
            {'params': model.classifier.parameters(), 'lr': 0.1}
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        ignored_params = list(map(id, model.model.fc.parameters()))
        ignored_params += (list(map(id, model.classifier0.parameters()))
                           + list(map(id, model.classifier1.parameters()))
                           + list(map(id, model.classifier2.parameters()))
                           + list(map(id, model.classifier3.parameters()))
                           + list(map(id, model.classifier4.parameters()))
                           + list(map(id, model.classifier5.parameters()))
                           )
        base_params = filter(lambda p: id(p) not in ignored_params,
                             model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.01},
            {'params': model.model.fc.parameters(), 'lr': 0.1},
            {'params': model.classifier0.parameters(), 'lr': 0.1},
            {'params': model.classifier1.parameters(), 'lr': 0.1},
            {'params': model.classifier2.parameters(), 'lr': 0.1},
            {'params': model.classifier3.parameters(), 'lr': 0.1},
            {'params': model.classifier4.parameters(), 'lr': 0.1},
            {'params': model.classifier5.parameters(), 'lr': 0.1},
        ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    # --------------------------------------------------------------------
    #
    # Decay LR by a factor of 0.1 every `lr_decay_epoch` epochs
    #
    # --------------------------------------------------------------------
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer,
                                           step_size=args.lr_decay_epoch,
                                           gamma=0.1)


    ######################################################################

    # --------------------------------------------------------------------
    #
    # Train and evaluate
    #
    # --------------------------------------------------------------------

    model = train_model(model, criterion, optimizer, exp_lr_scheduler,
                        num_epochs=args.num_epochs)
