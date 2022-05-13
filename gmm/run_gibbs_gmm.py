#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main function to run the alternate updating for multiple noise level(0,.1,.2,.3,.4)
Created on Wed May  4 16:38:06 2022
__Modified__: Shuyi Li
__author__ =: Erika Cole
"""

import numpy as np
import random,os
from util import gmm_generator_function,prior_calc,likelihood_calc,batch_gradient_descent
from slice import slice as slice_fun
import torch
#import torch.optim as optim

def main(seed=2022):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.autograd.set_detect_anomaly(True)
    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    uniform_prior = False #mean,std = (1,1) for nonflat, (2.25,10) for flat
    fld_name = 'result/flat_prior' if uniform_prior else 'result/nonflat_prior'
    os.makedirs(fld_name, exist_ok=True)
    
    
    # setting for Alternating Updates of alpha, theta
    
    p_vals = np.array([0, 0.1, 0.2, 0.3, 0.4]) #noise levels
    
    num_chains = 1#number of repeatability for the whole process  
    epochs = 20000 #for the whole alternating update
    M = 1000 
    pmix1 = 0.5 #prob of being 0
    nmix = int(pmix1*M)
    mu1 = -1
    mu = np.array([[mu1, mu1], [-mu1, -mu1]])
    sigma = 0.5*np.eye(2)
    
    n_batches = int(10)
    N_gd = 1 #number of gd we do when update theta
    
    alpha_chain_mult = []
    theta_chain_mult = []
    
    
    # loops over all noise values
    for p_val in p_vals:
    
        alpha_fname = os.path.join(fld_name,'alpha_chains_{:.0f}.txt'.format(p_val*100) )
        theta_fname = os.path.join(fld_name,'theta_chains_{:.0f}.npy'.format(p_val*100) )
        loss_fname = os.path.join(fld_name,'loss_chains_{:.0f}.txt'.format(p_val*100) )
        # samples the chain num_chains times (for repeatability analysis)
        for n_chain in range(num_chains):
            print('Chain #', n_chain+1, ' of ', num_chains)
            # print('\n')
            print('Starting iterations for noise level = {:.0%}'.format(p_val))
        
            
            p_flip = p_val
            
            #Generate GMM Training Data
            [X,y] = gmm_generator_function(M,nmix,mu, sigma, n_chain)
            
            
            if p_flip != 0:
                n_flip = int(M*p_flip)
                idx_flip = random.sample(range(int(M)), n_flip)
                for i in range(len(idx_flip)):
                    y[idx_flip[i]] = 1 if idx_flip[i]<M/2 else -1
            y = (y+1)/2 #convert -1,1 to 0,1
            X_tensor = torch.from_numpy(X).float().to(device)
            y_tensor = torch.from_numpy(y).float().to(device)
            alpha_chain, theta_chain = np.zeros((epochs+1,1)), np.zeros((epochs+1,3))
            loss_chain = np.zeros((epochs,1))
            
            print('X: ', X[:2,])
            # initialize alpha randomly
            alpha_init = np.random.uniform(low=0.5, high=4.0)
            print('Alpha initialized to {:.2f}'.format(alpha_init))
            alpha_chain[0] = alpha_init
    
            # initialize theta randomly
            theta_init = np.random.uniform(low=-2, high=2, size=(3,))
            print('Theta initialized to {}'.format(theta_init))
            theta_chain[0,:] = theta_init
            theta_curr = theta_init
            
            def post_1(alpha, theta_curr, X=X, y=y):
                
                loglike = likelihood_calc(X, y, theta_curr, alpha)
                # print('likelihood: {}'.format(like))
            
                end1 = 0.5
                end2 = 4.0
                mean = 1 if not uniform_prior else 2.25
                stdev = 1 if not uniform_prior else 10
                logprior = prior_calc(alpha, end1, end2, mean, stdev)
                # print('prior: {}'.format(prior))
                return (loglike + logprior)
            
            post = lambda alpha: post_1(alpha, theta_curr=theta_curr)
            l_init = post(alpha_init)
    
            # run alternating update -- slice sampling, gradient descent
            for epoch in range(epochs):
                
                
                if (epoch+1)%100 == 0:
                    print('Epoch #{:.0f} of {:.0f}'.format(epoch+1, epochs))

                # sample via slice sampling
                alpha_new, l_new = slice_fun(alpha_init, l_init, post)
    
                print('sampled alpha={:.2f}, new posterior={:.2f}'.format(alpha_new,l_new))
    
                # optimize theta - N_gd epoch of mini-batch gradient descent (10 mini-batches)
                # theta_new (3,1)
                theta_new, loss_vals = batch_gradient_descent(alpha_new, theta_curr, n_batches,  device, X_tensor, y_tensor, n_epochs=N_gd, M=M)      
    
                # print('optimized theta to be {}'.format(theta_new.T))
    
                alpha_chain[epoch+1] = alpha_new
                theta_chain[epoch+1,:] = theta_new.squeeze()
                loss_chain[epoch] = loss_vals #record alpha loss for bgd
                
                alpha_init = alpha_new
                l_init = l_new
                theta_curr = theta_new.squeeze()
                post = lambda alpha: post_1(alpha, theta_curr=theta_curr)
    
    
            if (n_chain == 0):
                # print('\nFIRST CHAIN\n')
                alpha_chain_mult = alpha_chain    
                theta_chain_mult = theta_chain[np.newaxis,]
                loss_chain_mult= loss_chain
            else:
                alpha_chain_mult = np.append(alpha_chain_mult, alpha_chain, axis=1)
                theta_chain_mult = np.append(theta_chain_mult, theta_chain[np.newaxis,], axis=0)
                loss_chain_mult = np.append(loss_chain_mult, loss_chain, axis=1)
                
        np.savetxt(alpha_fname, alpha_chain_mult, delimiter=',')
        np.save(theta_fname, theta_chain_mult)
        np.savetxt(loss_fname, loss_chain_mult, delimiter=',')
            
            
if __name__ == '__main__':
    
    main()