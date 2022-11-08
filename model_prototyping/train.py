#__author__ = "Islam Fadl"

from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from DrillImagesCls import ArtifitialImages, DrillImages
from PEModel import MyModel
import os
import time


# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

# Configurations
DEVICE = 'cuda'
BATCH_SIZE = 1
IMG_SIZE = 224
LR = 0.01   # increased from 0.001 to prevent pc from freezing
EPOCHS = 1

# TODO: Tensorboard for visualizing losses and accuracy estimation, similar graphs to coopick notebook.
# todo: print best loss only for each fold, or print loss very 10 epochs. less printing.
# todo: k=5, epochs=120
# todo: : remove pemodel and drillimagecls from deeper directories.


def simple_get_z(file_name):
    angle = re.findall(r'\d+', file_name)
    return angle[1]

# Model Creation
model = MyModel()
model.to(DEVICE)
summary(model, (3, 224, 224))

# Create Dataloaders
path = os.path.expanduser("~/PycharmProjects/syndataset")
#path = os.path.expanduser("~/PycharmProjects/qualifyar_rpi/imgfolder")

drill_dataset = DrillImages(path)
train_ds, valid_ds = train_test_split(drill_dataset, test_size=0.2,
                                      random_state=42)  # remove randam state in production to get a truely random split.

trainloader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)


# Train and Evaluation Functions (+MSE loss Function; L2 Loss)
def train_fn(model, dataloader, optimizer, loss_fn=nn.MSELoss()):
    total_loss = 0.0
    model.train()  # Sets mode, dropout layers are activated. Does not call the forward function.
    for data in dataloader:
        t, angles = data
        t, angles = t.to(DEVICE), angles.to(DEVICE)
        output = model(t)
        loss = loss_fn(output.float(), angles.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        return total_loss / len(dataloader)

def eval_fn(model, dataloader, loss_fn=nn.MSELoss()):
    total_loss = 0.0
    model.eval()  # Dropout OFF, no backpropagation I guess. And gradients are not calculated.
    with torch.no_grad():
        for data in dataloader:
            t, angles = data
            t, angles = t.to(DEVICE), angles.to(DEVICE)
            output = model(t)
            loss = loss_fn(output, angles)
            total_loss += loss.item()
            return total_loss / len(dataloader)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(),
                             lr=4e-4)  # have you moved the model to the gpu? if no, the model won't be optimized
train_loss = train_fn(model, trainloader, optimizer)

# K-folds Division
k = 2
foldperf = {}
drill_dataset = DrillImages(path)
splits = KFold(n_splits=k, shuffle=True,
               random_state=42)  # delete random state in production to get truly random splits.

for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(drill_dataset)))):
    print('\n')
    print('Fold {}'.format(fold + 1))
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(drill_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    valid_loader = DataLoader(drill_dataset, batch_size=BATCH_SIZE, sampler=valid_sampler)

    history = {'train_loss': [], 'valid_loss': []}

    best_valid_loss = np.Inf
    for i in range(EPOCHS):
        train_loss = train_fn(model, trainloader, optimizer)
        valid_loss = eval_fn(model, validloader)

        writer.add_scalar('Loss/train', train_loss, i)
        writer.add_scalar('Loss/test', valid_loss, i)

        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            print("Weights are saved")
            best_valid_loss = valid_loss

        print(f"Epoch : {i + 1} train loss : {train_loss}, valid loss : {valid_loss}")
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)

    foldperf['fold{}'.format(fold + 1)] = history
    # todo: add date to saved model name

timestr = time.strftime("%Y%m%d-%H%M%S")
torch.save(model, f'k_cross_CNN_Time_{timestr}.pt')

#writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#writer.add_scalar('Accuracy/test', np.random.random(), n_iter)