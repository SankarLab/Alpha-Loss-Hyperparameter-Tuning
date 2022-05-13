# -*- coding: utf-8 -*-
"""
main function to run the baseline plot(fixed alpha) for multiple noise level(0,.1,.2,.3,.4)
Created on Wed May  10 16:38:06 2022
__author__ =: Shuyi Li
__credit__: Erika Cole
"""

import numpy as np
import random,os,argparse
#import seaborn as sns
from util import gmm_generator_function,likelihood_calc,batch_gradient_descent
import torch


def main(seed=2022):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_prob', type=float, default=0.1, help='noise level when flipping label')
    opt = parser.parse_args()
    
    print('Run for noise level = ', opt.noisy_prob )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.autograd.set_detect_anomaly(True)
    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    fld_name, fld_name_m = 'result/baseline', 'result/baseline_model'
    os.makedirs(fld_name, exist_ok=True)
    os.makedirs(fld_name_m, exist_ok=True)
    
    # setting for Alternating Updates of alpha, theta
    
    
    experiments = 1#number of repeatability for the whole process 
    epochs = 10000 #for the whole gradient descent
    M = 1000 
    pmix1 = 0.5 #prob of being 0
    nmix = int(pmix1*M)
    mu1 = -1
    mu = np.array([[mu1, mu1], [-mu1, -mu1]])
    sigma = 0.5*np.eye(2)
    
    [X,y] = gmm_generator_function(M, nmix, mu, sigma)
    
    #flip the label
    if opt.noisy_prob != 0:
        n_flip = int(M*opt.noisy_prob)

        idx_flip = random.sample(range(int(M)), n_flip)
        for i in range(len(idx_flip)):
            y[idx_flip[i]] = 1 if idx_flip[i]<M/2 else -1
    y = (y+1)/2
    
    X_tensor = torch.from_numpy(X).float().to(device)
    y_tensor = torch.from_numpy(y).float().to(device)  
    
    n_batches = int(10)
    alphas = np.linspace(.5,4,21)
    theta_final = np.zeros((len(alphas),3)) #final optimal theta for different alpha
    normal_loss, loss = [], []
    

    for i,alpha in enumerate(alphas):
        theta_curalpha = np.zeros((experiments,3))
        for experiment in range(experiments):
                       
            print('\nStarting SGD for alpha={}, exp={}'.format(alpha, experiment+1))
    
            # initialize starting theta
            theta_init = np.random.uniform(low=-2, high=2, size=(3,))   
            # theta_new (3,1), save intermediate loss for each alpha
            theta_new, loss_vals = batch_gradient_descent(alpha, theta_init, n_batches,  device, X_tensor, y_tensor, opt, 
                                                          n_epochs=epochs, M=M, fld_name='result/baseline', save=True)                  
            normal_loss.append(likelihood_calc(X, y, theta_new, alpha) )
            loss.append(loss_vals)
            theta_curalpha[experiment] = theta_new.squeeze()
    
            # print('Final alpha-loss for alpha={} is: {}'.format(alpha, loss.item()))
            print('Current optimal thetas for alpha={}, exp={},  are: {}, {}, {}\n'\
                .format(alpha, experiment+1, theta_new.squeeze()[0], theta_new.squeeze()[1], theta_new.squeeze()[2]))
            
        theta_final[i] = theta_curalpha.mean(axis=0)
            
    np.savetxt(os.path.join(fld_name,'loss_noise{:.0f}.txt'.format(opt.noisy_prob*100)), loss, delimiter=',')
    np.savetxt(os.path.join(fld_name,'normal_loss_noise{:.0f}.txt'.format(opt.noisy_prob*100)), normal_loss, delimiter=',')
    np.save(os.path.join(fld_name,'weight_noise{:.0f}.npy'.format(opt.noisy_prob*100)), theta_final)

        
if __name__ == '__main__':
    
    main()



