#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:21:25 2022

@author: apple
"""

import torchvision
from scipy import stats
import numpy as np
import torch
import random
from torch.utils.data import SubsetRandomSampler #Dataset, DataLoader, 

# Function gets a new train_set and gets a new set of random indicies to noisify
def get_data(batch_size_train=64, batch_size_test=64, noisy_prob=.3, imb_a = int(6000), imb_b = int(6000)):
    train_set = torchvision.datasets.MNIST('./data', train=True, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor()
    ]))

    class_dic = {}
    for idx, target in enumerate(train_set.targets.tolist()):
        if target not in class_dic.keys():
            class_dic[target] = [idx]
        else:
            class_dic[target].append(idx)
    #num_classes = 10
    
    imbalance = [0,imb_a,0,0,0,0,0,imb_b,0,0]
    new_train_set_id = {}
    new_train_set = [] #id of all training set (1,7)
    for idx, im in enumerate(imbalance):
        sample = random.sample(class_dic[idx],imbalance[idx])
        new_train_set_id[idx] = sample
        for ids in new_train_set_id[idx]:
            new_train_set.append(ids)
            if train_set.targets[ids] == 1:
                train_set.targets[ids] = 0
            elif train_set.targets[ids] == 7:
                train_set.targets[ids] = 1

    # noisy_prob = float(sys.argv[1])
    noisy_indices = new_train_set.copy()
    random.shuffle(noisy_indices)
    noisy_indices = noisy_indices[:int(noisy_prob*len(noisy_indices))]

    for idx in noisy_indices:
        label = train_set.targets.tolist()[idx]
        choices = list(range(2))
        choices.remove(label)
        new_label = np.random.choice(choices)
        train_set.targets[idx] = int(new_label) #torch.LongTensor([new_label])
    
    dsamples = np.array([len(new_train_set_id[i]) for i in range(10)])
    data_size = imb_a + imb_b
    train_subset_loader  = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train,sampler=SubsetRandomSampler(new_train_set),drop_last=True)
    train_subset_loaderfull  = torch.utils.data.DataLoader(train_set, batch_size=data_size, sampler=SubsetRandomSampler(new_train_set),drop_last=True)
    test_set = torchvision.datasets.MNIST('./data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()
                                ]))
    new_test_set = []
    for idx, target in enumerate(test_set.targets.tolist()):
        if target == 1:
            test_set.targets[idx] = 0
            new_test_set.append(idx)
        elif target == 7:
            test_set.targets[idx] = 1
            new_test_set.append(idx)
    test_subset_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size_test,sampler=SubsetRandomSampler(new_test_set))
    return train_subset_loader, test_subset_loader, train_subset_loaderfull, dsamples


import torch.nn as nn
import torch.nn.functional as F


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
    
class _netDmnist(nn.Module):

    def __init__(self, ngpu, input_size=784, hidden_dim=32, num_classes=1):
        super(_netDmnist, self).__init__()
        self.ngpu = ngpu
        self.num_classes = num_classes
        self.main = nn.Sequential(
                Reshape(-1, input_size),
                nn.Linear(input_size, hidden_dim*4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim*4, hidden_dim*2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes)
                
        )
        if self.num_classes == 1:
          self.main.add_module('prob', nn.Sigmoid())
#        
#        
        
    def forward(self, input):       
        
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(input.size(0), self.num_classes).squeeze(1)

class Model(nn.Module):

    def __init__(self,num_classes=1):
        
        super(Model,self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1,32,(3,3))
        self.batch1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(32,64,(3,3))
        self.batch2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2,2))

        self.linear1 = nn.Linear(1600,128)
        self.batch3 = nn.BatchNorm1d(128)
        self.linear2 = nn.Linear(128,num_classes)
          

    def forward(self,x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = x.view(x.size()[0],-1)
        x = self.linear1(x)
        x = self.batch3(x)
        x = F.relu(x)

        x = self.linear2(x)
        if self.num_classes==1:
            x = F.sigmoid(x)
        return x.squeeze(1)

def prior_calc(x, end_1, end_2, mean, stdev):
    a = (end_1-mean)/stdev
    b = (end_2-mean)/stdev
    return np.log(stats.truncnorm.pdf(x, a, b, loc=mean, scale=stdev))

def likelihood_calc(output, target, alpha):
    #calculate the loglikelihood
    

    if alpha ==1:
        t1 = target * np.log(output+1e-8)
        t2 = (1-target) * np.log(1-output+1e-8)
        loss = -(t1 + t2)
        normalzt_1 = np.exp(np.log(output+1e-8) )
        normalzt_0 = np.exp(np.log(1-output+1e-8))
        log_sum_normalzt = np.log(normalzt_0+normalzt_1)

    else:
        pow_ = (alpha-1)/alpha

        t1 = target*np.power(output+1e-8, pow_)
        t2 = (1-target) * np.power(1-output+1e-8, pow_)
        loss = (alpha/(alpha-1)) * (1 - t1 - t2)
        normalzt_1 = np.exp(-(alpha/(alpha-1)) *(1-np.power(output+1e-8, pow_)) )
        normalzt_0 = np.exp(-(alpha/(alpha-1)) *(1-np.power(1-output+1e-8, pow_)) )
        log_sum_normalzt = np.log(normalzt_0+normalzt_1+1e-8)
    
    loss_total = loss+log_sum_normalzt
    

    return -np.sum(loss_total)


def post_1(alpha, target, output):
                
    loglike = likelihood_calc(output, target, alpha)
    # print('likelihood: {}'.format(like))

    end1 = 0.5
    end2 = 10.0
    mean = (end1+end2)/2 #if not uniform_prior else 2.25
    stdev = 20 # if not uniform_prior else 10
    logprior = prior_calc(alpha, end1, end2, mean, stdev)
    # print('prior: {}'.format(prior))
    return (loglike + logprior)


def alpha_loss(output, target, alpha=1, num_number=2, opt_cuda=False):
    
    loss = 0
    one = torch.FloatTensor([1.0]).cuda() if opt_cuda else torch.FloatTensor([1.0])
    if alpha == 1.0:
        loss = torch.mean(-target * torch.log(output + 1e-8) - (one - target) * torch.log(one - output + 1e-8))
    else:
        alpha = torch.FloatTensor([alpha]).cuda() if opt_cuda else torch.FloatTensor([alpha])    
        #(alpha/alpha-1)*(1-p(D(x_i)=y_i)^(1-1/alpha))
        if num_number>2:
            loss= (alpha/(alpha-one))*torch.mean(one - F.softmax(output).gather(1,target.view(-1,1)).pow(one - (one/alpha)))
        else:
            loss = (alpha/(alpha-one))*torch.mean(one - target*(output+ 1e-8).pow(one - (one/alpha)) - (one - target) * (one - output+ 1e-8).pow(one - (one/alpha)))
    return loss


#Function to find the mode of a set of samples (i.e. finds MAP for posterior samples)
def map_fun(chain, min=0.5, max=10, step=0.1):
    hist, bins = np.histogram(chain, bins=np.arange(min,max+step,step))
    map_arr = np.argmax(hist)
    map_val = bins[map_arr]
    return map_val


#Function that finds the MSE between model weights
    
def dict_mse(dict1, dict2):
    MSE = 0
    for key, values1 in dict1.items():
        if values1.size() == torch.Size([]):
            continue
        values2 = dict2[key]
        mse = nn.MSELoss()
        MSE += mse(torch.Tensor.float(values1), torch.Tensor.float(values2))
    return MSE


#Function that adds 0-mean  noise with specified variance to a dictionary (set of model weights)
def dict_noisy(dict_in, var):
    dict_noisy = dict(dict_in)
    for keys, vals in dict_noisy.items():
        # skip adding noise to single value parameters (iteration number)
        if vals.size() == torch.Size([]):
            continue
        noise = torch.normal(mean=0, std=var ** 1/2, size=vals.size()) # adds 0-mean noise with specified variance
        dict_noisy[keys] = vals + noise
    return dict_noisy


class AverageMeter(object):
    """ Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        