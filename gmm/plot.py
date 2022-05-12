#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main function to plot
Created on Thu May  5 14:39:29 2022
__author__ =: Shuyi Li
__credit__=: Erika Cole
"""
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
os.makedirs('./res_compare', exist_ok=True)


########## plot gmm data  ###########
from util import gmm_generator_function
import random
M = 1000 
pmix1 = 0.5 #prob of being 0
nmix = int(pmix1*M)
mu1 = -1
mu = np.array([[mu1, mu1], [-mu1, -mu1]])
sigma = 0.5*np.eye(2)
noisy_prob = 0

[X,y] = gmm_generator_function(M, nmix, mu, sigma)
if noisy_prob != 0:
    n_flip = int(M*noisy_prob/2)

    idx_flip = random.sample(range(int(M/2)), n_flip)
    for i in range(len(idx_flip)):
        y[idx_flip[i]] = 1
y = (y+1)/2

num_ones = int(sum(y))
x_ones = np.zeros((num_ones,2))
x_zeros = np.zeros((M-num_ones,2))
count_1 = 0
count_0 = 0

for i in range(M):
  if y[i] == 1:
    x_ones[count_1,:] = X[i,:]
    count_1 += 1
  else:
    x_zeros[count_0,:] = X[i,:]
    count_0 += 1

plt.figure(figsize=(8,5), dpi=100)
plt.scatter(x_ones[:,0], x_ones[:,1], label='Class +1')
plt.scatter(x_zeros[:,0], x_zeros[:,1], label='Class -1')
plt.title('GMM Data with Label Noise: {:.0%} Flip Probability'.format(noisy_prob))
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend()
plt.show()

########## plot posterior distributions ##########
uniform_prior = False
fld_name = 'result/flat_prior' if uniform_prior else 'result/nonflat_prior'
alphas = []
thetas = []
for i in np.arange(0,50,10):
    #(chain_len*experiments), experiments = 10 -> (chain_len*10) -> (1*chain_len*10)
    alphas.append(np.loadtxt(os.path.join(fld_name,'alpha_chains_'+str(i)+'.txt'), delimiter=',')[np.newaxis] )
    thetas.append(np.load(os.path.join(fld_name,'theta_chains_'+str(i)+'.npy')) )

alphas = np.concatenate(alphas,axis=0) #5*chain_len*experiments
experiments = 1 if len(alphas.shape)==2 else alphas.shape[2]
if experiments > 1:
    alphas = alphas.mean(axis=2) #5*chain_len*experiments -> 5*chain_len

burnin = 10000
plt.figure(figsize=(16,8))

plt.subplot(2,3,1)
sns.kdeplot(alphas[0][burnin:])
plt.hist(alphas[0][burnin:], density=True)
plt.xlabel('alpha')
plt.ylabel('density')
plt.title('0% noise')

plt.subplot(2,3,2)
sns.kdeplot(alphas[1][burnin:])
plt.hist(alphas[1][burnin:], density=True)
plt.xlabel('alpha')
plt.ylabel('density')
plt.title('10% noise')

plt.subplot(2,3,3)
sns.kdeplot(alphas[2][burnin:])
plt.hist(alphas[2][burnin:], density=True)
plt.xlabel('alpha')
plt.ylabel('density')
plt.title('20% noise')

plt.subplot(2,3,4)
sns.kdeplot(alphas[3][burnin:])
plt.hist(alphas[3][burnin:], density=True)
plt.xlabel('alpha')
plt.ylabel('density')
plt.title('30% noise')

plt.subplot(2,3,5)
sns.kdeplot(alphas[4][burnin:])
plt.hist(alphas[4][burnin:], density=True)
plt.xlabel('alpha')
plt.ylabel('density')
plt.title('40% noise')

plt.tight_layout()
plt.savefig('res_compare'+'/flat_'+str(uniform_prior)[0]+'.png',bbox_inches='tight', dpi=800)


########## plot Geweke Diagnostic for posterior alpha chain ##########
#!pip uninstall arviz
#!pip install arviz==0.11.0
#!pip install pymc3==3.10.0
import pymc3 as pm3
gw_plot = []
for i in np.arange(0,50,10):
    gw_plot.append( pm3.geweke(alphas[int(i/10)][burnin:]) )
    
plt.figure(figsize=[16,10])
for j in range(1,6):
    plt.subplot(2,3,j)
    plt.scatter(gw_plot[j-1][:,0],gw_plot[j-1][:,1])
    plt.axhline(-1.98, c='r')
    plt.axhline(1.98, c='r')
    plt.ylim(-2.5,2.5)
    plt.title('{:.0f}% Noise'.format((j-1)*10))
    
plt.suptitle('Geweke Plot Comparing first 10% and Slices of the Last 50% of Chain', fontsize=15, y=1.05)
plt.tight_layout()
plt.savefig('res_compare/geweke.png',bbox_inches='tight', dpi=800)



########## plot baseline(fixed alpha, find optimal weights) ##########

fld_name = 'result/baseline'
losses, normal_losses = [], []
alphas = np.linspace(0.5,4,21)

for i in np.arange(0,50,10):
    
    losses.append(np.loadtxt(os.path.join(fld_name,'loss_noise{:.0f}.txt'.format(i) ), delimiter=',') )
    normal_losses.append(np.loadtxt(os.path.join(fld_name,'normal_loss_noise{:.0f}.txt'.format(i) ), delimiter=',') )

def plot_base(losses, normalise=False):
    plt.figure(figsize=(8,8))
    plt.plot(alphas, np.exp(-np.array(losses[0])*(-1)**normalise), label='0% noise')
    plt.plot(alphas, np.exp(-np.array(losses[1])*(-1)**normalise), label='10% noise')
    plt.plot(alphas, np.exp(-np.array(losses[2])*(-1)**normalise), label='20% noise')
    plt.plot(alphas, np.exp(-np.array(losses[3])*(-1)**normalise), label='30% noise')
    plt.plot(alphas, np.exp(-np.array(losses[4])*(-1)**normalise), label='40% noise')
    
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel('likelihood')
    plt.title(r'Likelihood: exp(-$l^{\alpha}$)' if not normalise else r'Likelihood: exp(-$l^{\alpha}$)/$Z_\alpha$')
    plt.savefig('res_compare/baseline_'+'normal'+str(normalise)[0]+'.png',bbox_inches='tight', dpi=800)

plot_base(losses)
plot_base(normal_losses, normalise=True)



########## plot baseline accuracy(fixed alpha, find optimal weights) ##########

from util import accuracy_calc_thetas, gmm_generator_function
fld_name = 'result/baseline'
weight = []
alphas = np.linspace(0.5,4,21)

for i in np.arange(0,50,10):
    
    weight.append(np.load(os.path.join(fld_name,'weight_noise{:.0f}.npy'.format(i) )) )

def generate_noisy(noisy_prob=0):
    
    M = 1000
    pmix1 = 0.5 #prob of being 0
    nmix = int(pmix1*M)
    mu1 = -1
    mu = np.array([[mu1, mu1], [-mu1, -mu1]])
    sigma = 0.5*np.eye(2)
    
    [X,y] = gmm_generator_function(N=M, n_min=nmix, Mu=mu, Sigma=sigma)
    
    if noisy_prob != 0:
        n_flip = int(M*noisy_prob/2)
        idx_flip = random.sample(range(int(M/2)), n_flip)
        for i in range(len(idx_flip)):
            y[idx_flip[i]] = 1
    y = (y+1)/2
    return X, y

acc = []
for i in np.arange(0,50,10):
    
    X_test, y_test = generate_noisy(i/100)
    acc.append( accuracy_calc_thetas(X_test, y_test, weight[int(i/10)].T) )

def plot_acc_base(acc):
    plt.figure(figsize=(8,8))
    plt.plot(alphas, acc[0], label='0% noise')
    plt.plot(alphas, acc[1], label='10% noise')
    plt.plot(alphas, acc[2], label='20% noise')
    plt.plot(alphas, acc[3], label='30% noise')
    plt.plot(alphas, acc[4], label='40% noise')
    
    plt.legend()
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Accuracy')
    plt.title(r'Accuracy for different alphas')
    plt.savefig('res_compare/baseline_acc'+'.png',bbox_inches='tight', dpi=800)

plot_acc_base(acc)