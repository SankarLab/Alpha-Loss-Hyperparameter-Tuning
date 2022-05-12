#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:58:17 2022
main function to run the baseline plot(fixed alpha) for multiple noise level(0,.1,.2,.3,.4)
Created on Wed May  4 16:38:06 2022
__Modified__: Shuyi Li
__author__ =: Erika Cole
"""

import numpy as np
import random, argparse, os
import torch
import torch.optim as optim
#from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from util import get_data,alpha_loss,Model,AverageMeter,likelihood_calc 

def main(seed=2022):

    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_prob', type=float, default=0.1, help='noise level when flipping label')
    opt = parser.parse_args()
    print('Run for noise level = ', opt.noisy_prob )
    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    bs_test,bs_train = 64,64
    max_epoch = 50
    learning_rate = float(1e-3)
    momentum = float(0.9)
    wd = float(1e-4)
    
    experiments = 1 #number of repeatability for the whole process
    num_classes = 1
    loss_chain = []
    normal_loss_chain = []
    
    
    fld_name, fld_name_m = 'result/baseline', 'result/baseline_model'
    os.makedirs(fld_name, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs(fld_name_m, exist_ok=True)
    alphas = np.linspace(.5,10,20)
    
    for alpha in alphas:
        cm_average = np.zeros((2,2))
        
        for experiment in range(experiments):
           
            print('STARTING ALPHA: ', alpha)
    
            train_loader, test_loader, train_loaderfull, _ = get_data(batch_size_train=bs_train, batch_size_test=bs_test, noisy_prob=opt.noisy_prob)
            network = Model(num_classes = num_classes)#.cuda()
    
            optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                                momentum=momentum, weight_decay=wd)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            # scheduler = MultiStepLR(optimizer, milestones=[10,30], gamma=0.1)
            lossval = []
            epoch_train_loss = AverageMeter()
            
            for epoch in range(max_epoch):
                print('\n\n======================\n ITERATION {} of {}\n======================\n'.format(epoch+1, max_epoch))

                train_loss = []
                network.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    
                    network.zero_grad()
                    data = data.cuda() if torch.cuda.device_count() else data
                    output = network(data)
                    
                    loss = alpha_loss(output,target, alpha)
                    train_loss.append(loss.cpu().item())
                    loss.backward()
                    optimizer.step()
                    
                    epoch_train_loss.update(loss.item())
                
                print('[%d/%d] Loss: %.2f'% (epoch+1, max_epoch, epoch_train_loss.avg ))
                
                lossval.append(epoch_train_loss.avg)
                epoch_train_loss.reset()
                #scheduler.step()
                    # print('\nEnd gradient descent\n')
            
            network.eval()
            for (data, target) in (train_loaderfull):
                output = network(data)
            
            normal_loss_chain.append(likelihood_calc(output.detach().numpy(), target.detach().numpy(), alpha))
            loss_chain.append(lossval[-1])
            
            # save intermediate loss for each alpha each fixed noise level
            np.savetxt( os.path.join(fld_name,'loss_noise{:.0f}_alpha{:.1f}.txt'.format(opt.noisy_prob*100, alpha) ), lossval, delimiter=',')
    
            # save model
            torch.save(network.state_dict(), os.path.join(fld_name_m,'noise{:.0f}_alpha{:.1f}'.format(opt.noisy_prob*100, alpha)) )
            
    
            # Testing at last iteration/epoch for each alpha each noise level
            test_loss = []
            test_total, test_acc_cnt = 0, 0
            predictions, actual = [], []
            network.eval()
            
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.cuda() if torch.cuda.device_count() else data
                output = network(data)
                test_pred = output>=.5   
                test_total += target.size()[0]
                predictions += test_pred.cpu().tolist()
                test_acc_cnt += (target == test_pred.cpu()).sum().item()
                t_loss = alpha_loss(output,target, alpha)
                test_loss.append(t_loss.cpu().item())
                actual += target.tolist()
            print('Test Accuracy:\t' + str((test_acc_cnt)/test_total))
            cm = confusion_matrix(actual, predictions)
            mcc = matthews_corrcoef(actual,predictions)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            class_accuracies = ','.join([str(c) for c in cm.diagonal().tolist()])
            print('Test Class Accuracies:\t' + class_accuracies)
            print('MCC:\t'+str(mcc))
            f1_micro = f1_score(actual,predictions,average='micro')                     
            f1_macro = f1_score(actual,predictions,average='macro')                     
            f1_weighted = f1_score(actual,predictions,average='weighted')               
            f1_default = f1_score(actual,predictions,average=None)                      
            print('F1 Micro:\t' + str(f1_micro))                                        
            print('F1 Macro:\t' + str(f1_macro))                                        
            print('F1 Weighted:\t' + str(f1_weighted))                                  
            print('F1 Default:\t' + str(f1_default))
            cm_average += cm
            print('Confusion Matrix:')                                                  
            print(cm)                                                                   
            print('-------')
            
        cm_average = cm_average/experiments #aggregate over all experiments for each alpha
        np.save(os.path.join(fld_name, 'mnist_cm_noise{:.0f}_alpha{:.1f}.npy'.format(opt.noisy_prob*100, alpha)), cm_average)
                  
    
            
    #save loss for all alphas at last iteration at fixed noise level
    np.savetxt(os.path.join(fld_name,'loss_noise{:.0f}.csv'.format(opt.noisy_prob*100) ), loss_chain, delimiter=',')
    np.savetxt(os.path.join(fld_name,'normal_loss_noise{:.0f}.csv'.format(opt.noisy_prob*100) ), normal_loss_chain, delimiter=',')
    
if __name__ == '__main__':
    
    main()    
    
