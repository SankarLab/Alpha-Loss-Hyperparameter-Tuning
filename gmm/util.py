#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GMM_modGibbs_alphaloss

Created on Wed May  4 16:11:19 2022

__author__ = "Erika Cole"
__credits__ = "Shuyi Li"
"""

import numpy as np
import os
from scipy import stats
# from scipy.integrate import quad
import torch
import torch.optim as optim



def gmm_generator_function(N, n_min, Mu, Sigma):
  

    n_maj = N - n_min

    # initializes mean and covariance matrices for class +1
    A = Sigma
    b1_full = np.tile(Mu[0,:],(n_min,1)).T

    # initializes mean for class -1
    b2_full = np.tile(Mu[1,:],(n_maj,1)).T

    b_full = np.hstack((b1_full, b2_full))

    # generates random [2,N] matrix ~ N(0,1)
    Z = np.random.randn(2,N)

    # convert ~N(0,1) random points to desired distributions 
    # via A*Z+b ~N(b, AA^T)
    X = np.dot(A,Z)+b_full

    # initialize vector with labels
    y1 = -np.ones((n_min,1))
    y2 = np.ones((n_maj,1))

    y = np.vstack((y1,y2))

    return X.T, y


def sigmoid(z):
    a = 1 + np.exp(-z)
    return 1/a


def alpha_loss_calc(X, y, theta, alpha):
    #calculate alpha loss
    d = len(y)
    X_ones = np.ones((d,1))
    X_aug = np.hstack((X_ones,X))
    # y_trans = 2*y-1

    X_T_theta = (X_aug @ theta).reshape((d,1))
    z = X_T_theta
    # z = X_T_theta * y_trans
    sigmoid_z = sigmoid(z)

    if alpha ==1:
        t1 = y * np.log(sigmoid_z)
        t2 = (1-y) * np.log(1-sigmoid_z)
        loss = -(t1 + t2)

    else:
        pow_ = (alpha-1)/alpha

        t1 = y*(np.power(sigmoid_z+1e-8, pow_))
        t2 = (1-y) * (np.power(1-sigmoid_z+1e-8, pow_))
        loss = (alpha/(alpha-1)) * (1 - t1 - t2)

    mean_loss = np.mean(loss)  
    return mean_loss


def prior_calc(x, end_1, end_2, mean, stdev):
    a = (end_1-mean)/stdev
    b = (end_2-mean)/stdev
    return np.log(stats.truncnorm.pdf(x, a, b, loc=mean, scale=stdev)+1e-6)

#import matplotlib.pyplot as plt
#fig, ax = plt.subplots(1, 1)
#r = stats.truncnorm.rvs(a, b, loc=1, scale=1, size=10000)
#ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)

def likelihood_calc(X, y, theta, alpha):
    #calculate the loglikelihood, theta with size (n_feature,) or (n_feature,1)
    d = len(y)
    X_ones = np.ones((d,1))
    X_aug = np.hstack((X_ones,X))
   

    X_T_theta = (X_aug @ theta).reshape((d,1))
    z = X_T_theta 
    sigmoid_z = sigmoid(z)

    if alpha ==1:
        t1 = y * np.log(sigmoid_z+1e-8)
        t2 = (1-y) * np.log(1-sigmoid_z+1e-8)
        loss = -(t1 + t2)
        normalzt_1 = np.exp(np.log(sigmoid_z+1e-8) )
        normalzt_0 = np.exp(np.log(1-sigmoid_z+1e-8))
        log_sum_normalzt = np.log(normalzt_0+normalzt_1)

    else:
        pow_ = (alpha-1)/alpha

        t1 = y*np.power(sigmoid_z+1e-8, pow_)
        t2 = (1-y) * np.power(1-sigmoid_z+1e-8, pow_)
        loss = (alpha/(alpha-1)) * (1 - t1 - t2)
        normalzt_1 = np.exp(-(alpha/(alpha-1)) *(1-np.power(sigmoid_z+1e-8, pow_)) )
        normalzt_0 = np.exp(-(alpha/(alpha-1)) *(1-np.power(1-sigmoid_z+1e-8, pow_)) )
        log_sum_normalzt = np.log(normalzt_0+normalzt_1+1e-8)
    
    loss_total = loss+log_sum_normalzt
    

    return -np.sum(loss_total)


# =============================================================================
# def target_draw(alpha, sigma):
#     # return np.random.normal(loc=alpha, scale=sigma)
#     return stats.norm.rvs(loc=alpha, scale=sigma)
# 
# def target_calc(alpha, alpha_cond, sigma):
#     # return stats.norm.pdf(alpha,loc=alpha_cond,scale=sigma)
#     return np.log(stats.norm.pdf(alpha,loc=alpha_cond,scale=sigma))
# 
# def target_draw_2(alpha, stdev, end_1, end_2):
#     a = (end_1-alpha)/stdev
#     b = (end_2-alpha)/stdev
#     return stats.truncnorm.rvs(a, b, loc=alpha, scale=stdev)
# =============================================================================




def alpha_loss_calc_torch(x_tensor, y_tensor, theta_1, theta_2, theta_3, alpha):

    X_tensor_1 = x_tensor[:,0]
    X_tensor_2 = x_tensor[:,1]
    X_tensor_1 = X_tensor_1.view(X_tensor_1.size(0),1)
    X_tensor_2 = X_tensor_2.view(X_tensor_2.size(0),1)

    # print('1: {}'.format(X_tensor_1.size()))
    X_T_theta = theta_1 + theta_2*X_tensor_1 + theta_3*X_tensor_2
    sig_tensor = (torch.sigmoid(X_T_theta)) #torch.special.expit
   
    if alpha == 1:
        t1 = y_tensor * torch.log(sig_tensor+1e-8)
        t2 = (1-y_tensor) * torch.log(1-sig_tensor+1e-8)
        torch_loss = -(t1 + t2)

    else:
        pow_ = 1-(1/alpha)
        sig_power_1 = torch.pow(sig_tensor+1e-8, pow_)
        sig_power_2 = torch.pow(1-sig_tensor+1e-8, pow_)               
        torch_loss = (alpha/(alpha-1))*(1 - y_tensor*sig_power_1 - (1-y_tensor)*sig_power_2)
        

    # with torch.no_grad():
    #     mean_loss = torch.mean(loss)

    mean_loss = torch.mean(torch_loss)
    return mean_loss, sig_tensor



def gradient_descent(alpha_map, theta_new, device, X_tensor, y_tensor, n_epochs=1):

    # alpha_tensor = torch.from_numpy(np.array(alpha_map)).float().to(device)

    alpha_count = 0

    theta1_vals = []
    theta2_vals = []
    theta3_vals = []
    loss_vals = []
    

    eta = 1e-3
    Tmax = n_epochs
    # Tmax = 1000
    # n_epochs = 1000

    for alpha in range(1):
        
        # print('\nStarting SGD for alpha={}'.format(alpha))

        # send current iteration's alpha to device
        alpha_tensor = torch.tensor(alpha_map, requires_grad=False, dtype=torch.float, device=device)

        # initialize starting theta
        theta_1 = torch.tensor(theta_new[0], requires_grad=True, dtype=torch.float, device=device)
        theta_2 = torch.tensor(theta_new[1], requires_grad=True, dtype=torch.float, device=device)
        theta_3 = torch.tensor(theta_new[2], requires_grad=True, dtype=torch.float, device=device)    
        # print('starting value of theta: [{},{},{}]'.format(theta_1, theta_2, theta_3))

        optimizer = optim.SGD([theta_1, theta_2, theta_3], lr=eta)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Tmax)

        for epoch in range(n_epochs):
            # y_hat = round(sigmoid(X_T_theta))
            # error = 1
            optimizer.zero_grad() 
            loss, sig_tensor = alpha_loss_calc_torch(X_tensor, y_tensor, theta_1, theta_2, theta_3, alpha_tensor)  

            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            # print('Loss for epoch #{} is{}'.format(epoch+1, loss))
            # print('Theta: {},{},{}'.format(theta_1.item(), theta_2.item(), theta_3.item()))
            # print('gradient: {},{},{}'.format(theta_1.grad, theta_2.grad, theta_3.grad))
            # if epoch%100 == 0:
            #     print('Loss for epoch #{} is: {}'.format(epoch+1, loss))
            #     print(theta_1.grad, theta_2.grad, theta_3.grad,'\n')

            #??????????????delete????????????????  gd_vals.append(theta_2.item())

        theta1_vals.append(theta_1.item())
        theta2_vals.append(theta_2.item())
        theta3_vals.append(theta_3.item())

        loss_vals.append(loss.item())

        alpha_count += 1    
    # return np.array([theta1_vals,theta2_vals, theta3_vals])
    # return np.array([theta1_vals,theta2_vals, theta3_vals]), gd_vals
    return np.array([theta_1.item(), theta_2.item(), theta_3.item()]), loss_vals



def batch_gradient_descent(alpha_map, theta_new, n_batches, device, X_tensor, y_tensor, opt, 
                           n_epochs=1, M=1000, fld_name='result/baseline', save=False):

    alpha_tensor = torch.from_numpy(np.array(alpha_map)).float().to(device)

    theta1_vals = []
    theta2_vals = []
    theta3_vals = []
    loss_vals = []


    batch_size = int(np.ceil(M/n_batches))
    eta = 1e-3
    Tmax = 50
    
   
        
    # print('\nStarting SGD for alpha={}'.format(alpha_map))

    # initialize starting theta
    theta_1 = torch.tensor(theta_new[0], requires_grad=True, dtype=torch.float, device=device)
    theta_2 = torch.tensor(theta_new[1], requires_grad=True, dtype=torch.float, device=device)
    theta_3 = torch.tensor(theta_new[2], requires_grad=True, dtype=torch.float, device=device)    

    optimizer = optim.SGD([theta_1, theta_2, theta_3], lr=eta)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Tmax)

    for epoch in range(n_epochs):
        sumloss = 0
        for batch in range(n_batches):

            X_batch = X_tensor[batch*batch_size:(1+batch)*batch_size,:]
            y_batch = y_tensor[batch*batch_size:(1+batch)*batch_size,0]

            optimizer.zero_grad() 
            loss, sig_tensor = alpha_loss_calc_torch(X_batch, y_batch, theta_1, theta_2, theta_3, alpha_tensor) 
            print('bgd, loss = {:.2f}'.format(loss.item()))
            loss.backward()
            sumloss += loss.item()
            optimizer.step()

        avgloss = sumloss/n_batches
        loss_vals.append(avgloss)
        lr_scheduler.step()
    
    if save:
        np.savetxt( os.path.join(fld_name,'loss_noise{:.0f}_alpha{:.1f}.txt'.format(opt.noisy_prob*100, alpha_map) ), loss_vals, delimiter=',')

           
    theta1_vals.append(theta_1.item())
    theta2_vals.append(theta_2.item())
    theta3_vals.append(theta_3.item())
    
    
    return np.array([theta1_vals,theta2_vals, theta3_vals]), avgloss


def accuracy_calc(X_test, y_test, theta): 
    #calculate accuracy for one weight theta(3,1)/(3,)
    d = len(y_test)
    X_ones = np.ones((d,1))
    X_aug = np.hstack((X_ones, X_test))
   

    X_T_theta = (X_aug @ theta).reshape((d,1))
    z = X_T_theta 
    sigmoid_z = sigmoid(z)
    
    y_pred = sigmoid_z>=.5
    acc = np.sum(y_pred==y_test)/d


    return acc

def accuracy_calc_thetas(X_test, y_test, theta): 
    #calculate accuracy for multiple thetas -- matrix(3*d), where d is the number of theta
    N = len(y_test)
    X_ones = np.ones((N,1))
    X_aug = np.hstack((X_ones, X_test))
   
    #(N,3)*(3,d) - > (N, d), N is the sample size
    X_T_theta = (X_aug @ theta)
    z = X_T_theta 
    sigmoid_z = sigmoid(z)
    
    y_pred = sigmoid_z>=.5
    acc = [np.sum(y_pred[:,i]==y_test[:,0])/N for i in range(theta.shape[1])]

    return acc