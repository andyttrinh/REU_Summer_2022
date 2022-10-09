#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import sys
from resnet20 import resnet20
import torch
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import argparse
import math
import numpy as np
import pandas as pd
from pyvacy import optim, analysis
from blurnn import optim
import time

if len(sys.argv) != 3:
    print("Wrong nubmer of inputs")
    exit(0)



# In[2]:


# Data Loading
acc = 0
acc_best = 0
epoch = 0

DATA_ROOT = './cifar10'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

data_train = CIFAR10(DATA_ROOT,
                   transform=transform_train,
                   download=True)
data_test = CIFAR10(DATA_ROOT,
                  train=False,
                  transform=transform_test)

data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8) # original batch_size=256
data_test_loader = DataLoader(data_test, batch_size=100, num_workers=2) # original batch size = 100


# In[3]:


# Model
net = resnet20().cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()


# In[27]:


# # Optimizer DPSGD Pyvacy
# l2_norm_clip = 1.0
# noise_multiplier = 1.1
# delta = 1e-5
# lr = 0.1

# optimizer = optim.DPSGD(
#     l2_norm_clip=l2_norm_clip,
#     noise_multiplier=noise_multiplier,
#     batch_size=256,
#     params=net.parameters(),
#     lr=lr
# )


# In[12]:


# In[15]:


lr = 0.1
momentum = 0.9
weight_decay=5e-4
# noise_scale = 0.03
# norm_bound = 1.5
noise_scale = float(sys.argv[1])
norm_bound = float(sys.argv[2])

optimizer = optim.DPSGD(params=net.parameters(), lr=lr, momentum=momentum, nesterov=False, noise_scale=noise_scale, norm_bound=norm_bound)


# In[16]:


#Data Parallelism
net = torch.nn.DataParallel(net)


# In[17]:

model_path = './models/addernet_DPSGD_E{}_C{}.pth'.format(noise_scale, norm_bound)
df_path = './dataframes/addernet_DPSGD_E{}_C{}.csv'.format(noise_scale, norm_bound)


# Checkpoint
if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    net.eval()
    net.train()
#     loss = checkpoint['loss']


# In[18]:


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    lr = 0.05 * (1+math.cos(float(epoch)/400*math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# In[19]:


def train(epoch):
    adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
 
        optimizer.zero_grad()
 
        output = net(images)
 
        loss = criterion(output, labels)
 
        loss_list.append(loss.data.item())
        batch_list.append(i+1)
 
        if i == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.data.item()))
 
        loss.backward()
        optimizer.step()


# In[20]:


def test():
    global acc, acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
 
    avg_loss /= len(data_test)
    acc = float(total_correct) / len(data_test)
    if acc_best < acc:
        acc_best = acc
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
    return (avg_loss.data.item(), acc)


# In[21]:


def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)


# In[22]:


# DataFrame
if os.path.exists(df_path):
    df = pd.read_csv(df_path, index_col = 0)
else:
    df = pd.DataFrame(columns=['Epoch', 'Loss', 'Accuracy'])


# In[23]:


def train_and_test(epoch):
    curr_list = [epoch]
    # start = time.time()
    train(epoch)
    # end = time.time()
    # print("Training Time: {}".format(end - start))
    loss, acc = test()
    curr_list.append(loss)
    curr_list.append(acc)
    df.loc[len(df.index)] = curr_list


# In[24]:


def main(start = 1):
    epoch = 400# original epoch = 400
    for e in range(start, epoch):
        train_and_test(e)
        if e % 5 == 0:
            save_checkpoint(net, optimizer, model_path, e)
            df.to_csv(df_path)
    torch.save(net.state_dict(),model_path)
    df.to_csv(df_path)


# In[25]:


main(epoch + 1)


# In[ ]:




