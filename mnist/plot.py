#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main function to plot
Created on Thu May  5 14:39:29 2022
__author__ =: Shuyi Li
"""
import numpy as np
import os
import matplotlib.pyplot as plt
os.makedirs('./res_compare', exist_ok=True)

########## plot posterior distributions ##########

fld_name = 'result' 

alphas = []
experiments = 1

for i in np.arange(0,50,10):
    for experiment in range(experiments):
        alphas.append(np.loadtxt(os.path.join(fld_name,'alphachain_noise{:.0f}_exp{:d}.txt'.format(i, experiment+1)), delimiter=',')[np.newaxis,] )

alphas = np.concatenate(alphas,axis=0).reshape((5,experiments,-1)) #5*experiments*chain_len
avgalphas = alphas.mean(axis=1) #5*experiments*chain_len -> 5*chain_len

burnin = 10000
plt.figure(figsize=(16,8))

plt.subplot(2,3,1)
plt.hist(avgalphas[0][burnin:], density=True)
plt.xlabel('alpha')
plt.ylabel('density')
plt.title('0% noise')

plt.subplot(2,3,2)
plt.hist(avgalphas[1][burnin:], density=True)
plt.xlabel('alpha')
plt.ylabel('density')
plt.title('10% noise')

plt.subplot(2,3,3)
plt.hist(avgalphas[2][burnin:], density=True)
plt.xlabel('alpha')
plt.ylabel('density')
plt.title('20% noise')

plt.subplot(2,3,4)
plt.hist(avgalphas[3][burnin:], density=True)
plt.xlabel('alpha')
plt.ylabel('density')
plt.title('30% noise')

plt.subplot(2,3,5)
plt.hist(avgalphas[4][burnin:], density=True)
plt.xlabel('alpha')
plt.ylabel('density')
plt.title('40% noise')

plt.tight_layout()
plt.savefig('res_compare/'+'posterior.png',bbox_inches='tight', dpi=800)



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
alphas = np.linspace(0.5,10,20)

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


