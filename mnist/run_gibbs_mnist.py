#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main function to run the alternate updating for multiple noise level(0,.1,.2,.3,.4)
Created on Wed May  4 16:38:06 2022
__Modified__: Shuyi Li
__author__ =: Erika Cole
"""

import numpy as np
import random,argparse,sys,os
import torch
import torch.optim as optim
#from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from util import get_data,post_1,alpha_loss,dict_noisy,dict_mse,map_fun,Model,_netDmnist,AverageMeter #prior_calc,likelihood_calc,
sys.path.append( "../" )
from slice import slice as slice_fun
#import torch.optim as optim

def main(seed=2022):
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy_prob', type=float, default=0.1, help='noise level when flipping label')
    opt = parser.parse_args()
    
    print('Run for noise level = ', opt.noisy_prob )
    
    ngpu = torch.cuda.device_count()
    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.autograd.set_detect_anomaly(True)
    
    fld_name, fld_name_m = 'result', 'result/baseline_model'
    os.makedirs(fld_name, exist_ok=True)
    bs_test = 64
    bs_train = 64
    imb_a, imb_b = 6000, 6000
    train_total = imb_a + imb_b
    model_opt = 'cnn'
    learning_rate = float(1e-3)
    momentum = float(0.9)
    wd = float(1e-4)
    
    
    
    # my_alpha = float(sys.argv[2])
    max_epoch = 20000
    experiments = 1 #number of repeatability for the whole process
    cm_average = np.zeros((2,2))             
    
    
    num_classes = 1
    # loads dictionary of optimal weights for alpha=1
    checkpoint = torch.load(os.path.join(fld_name_m,'noise{:.0f}_alpha{:.1f}'.format(opt.noisy_prob*100, 1)) )
    # set starting weights to be a perturbation of optimal weights for alpha=1
    #checkpoint_noisy = dict_noisy(checkpoint, .01)

    for experiment in range(experiments):
        alpha_chain = []
        loss_chain = []
        logposterior_chain = []
        MSE = []
        accuracy = []
        
        my_alpha = 1
        print('STARTING ALPHA: ', my_alpha)
    
        train_loader, test_loader, train_loaderfull, _ = get_data(batch_size_train=bs_train, batch_size_test=bs_test, noisy_prob=opt.noisy_prob)
        network = Model(num_classes = num_classes) if model_opt == 'cnn' else _netDmnist(ngpu, num_classes=num_classes)#.cuda()
        
        ## store MSE values for weights (compared to alpha=1)
        #network.load_state_dict(checkpoint_noisy)
        print('starting MSE: ', dict_mse(checkpoint, network.state_dict()).item())
    
        optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                            momentum=momentum, weight_decay=wd)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        # scheduler = MultiStepLR(optimizer, milestones=[10,30], gamma=0.1)

        epoch_train_loss = AverageMeter()
        for epoch in range(max_epoch):
            print('\n\n======================\n ITERATION {} of {}\n======================\n'.format(epoch+1, max_epoch))
            
            ## gradient descent part to update weights  ##

            network.train()
            
            train_acc_cnt = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                
                network.zero_grad()
                data = data.cuda() if torch.cuda.device_count() else data
                output = network(data)
                
                loss = alpha_loss(output,target, my_alpha)
                loss.backward()
                optimizer.step()
                epoch_train_loss.update(loss.item())
                
                # calculate accuracy
                train_pred = output>=.5          
                train_acc_cnt += (target == train_pred.cpu()).sum().item()
               
            network.eval()
            for (data, target) in (train_loaderfull):
                output = network(data)
            
            
            #scheduler.step()
    
    
            ##  slice sampling part to update alpha    ##
            with torch.no_grad():
                accuracy.append(train_acc_cnt/train_total)
                print('accuracy is: ', train_acc_cnt/train_total)
                print('[%d/%d] Loss: %.2f'% (epoch+1, max_epoch, epoch_train_loss.avg ))
                mse = dict_mse(network.state_dict(), checkpoint)
                print('MSE: ', mse)
                MSE.append(mse.item())
                loss_chain.append(epoch_train_loss.avg)
                epoch_train_loss.reset()
                
 
    # =============================================================================
    #             if (epoch+1) < 200 and (epoch+1)%10 != 0:
    #                 print('Pass sampling of alpha; epoch is not divisible by 10.')
    #                 continue
    #             if (epoch+1) > 200 and (epoch+1) < 400 and (epoch+1)%5 != 0:
    #                 print('Pass sampling of alpha; epoch is not divisible by 5.')
    #                 continue
    #             if (epoch+1) > 400 and (epoch+1) < 600 and (epoch+1)%4 != 0:
    #                 print('Pass sampling of alpha; epoch is not divisible by 4.')
    #                 continue
    #             if (epoch+1) > 600 and (epoch+1) < 800 and (epoch+1)%3 != 0:
    #                 print('Pass sampling of alpha; epoch is not divisible by 3.')
    #                 continue
    #             if (epoch+1) > 800 and (epoch+1) < 1000 and (epoch+1)%2 != 0:
    #                 print('Pass sampling of alpha; epoch is not divisible by 2.')
    #                 continue                    
    # =============================================================================
                
                post = lambda alpha: post_1(alpha, target=target.numpy(), output=output.numpy())
                l_init = post(my_alpha)
                my_alpha, l_init = slice_fun(my_alpha, l_init, post)
                print('Accept alpha={}'.format(my_alpha))
                alpha_chain.append(my_alpha.item())
                logposterior_chain.append(l_init)
                
    

        np.savetxt(os.path.join(fld_name,'alphachain_noise{:.0f}_exp{:d}.txt'.format(opt.noisy_prob*100, experiment+1)), alpha_chain, delimiter=',')
        np.savetxt(os.path.join(fld_name,'accchain_noise{:.0f}_exp{:d}.txt'.format(opt.noisy_prob*100, experiment+1)), accuracy, delimiter=',')
        np.savetxt(os.path.join(fld_name,'logposterior_chain_noise{:.0f}_exp{:d}.txt'.format(opt.noisy_prob*100, experiment+1)), logposterior_chain, delimiter=',')
        np.savetxt(os.path.join(fld_name,'losschain_noise{:.0f}_exp{:d}.txt'.format(opt.noisy_prob*100, experiment+1)), loss_chain, delimiter=',')
        np.savetxt(os.path.join(fld_name,'msechain_noise_{:.0f}_exp{:d}.txt'.format(opt.noisy_prob*100, experiment+1)), MSE, delimiter=',')
    
        my_alpha = map_fun(alpha_chain)
    
        print('\n\nMAP of alpha is: {}'.format(my_alpha))
        # Testing at last epoch
        test_loss = []
        test_total, test_acc = 0, 0
        predictions, actual = [], []
        network.eval()
        
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.cuda() if torch.cuda.device_count() else data
            output = network(data)
            
            test_pred = output>=.5         
            test_acc += (target == test_pred.cpu()).sum().item()
            test_total += target.size()[0]
            predictions += test_pred.cpu().tolist()
            t_loss = alpha_loss(output,target, my_alpha)
            test_loss.append(t_loss.cpu().item())
            actual += target.tolist()
            
        print('Test Accuracy:\t' + str(test_acc/test_total))
        cm = confusion_matrix(actual, predictions)
        #np.save(os.path.join(fld_name,'confusion_mtx_noise{:.0f}_exp{:d}.npy'.format(opt.noisy_prob*100, experiment+1)), cm)
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
    
        
    print('===========')
    cm_average = cm_average/experiments
    print(cm_average)
    tn, fp, fn, tp = cm_average.ravel()
    print('True Negative:\t' + str(tn))
    print('False Positive:\t' + str(fp))
    print('False Negative:\t' + str(fn))
    print('True Positive:\t' + str(tp))
    np.save(os.path.join(fld_name,'mnist_cm_noise{:.0f}.npy'.format(opt.noisy_prob*100)), cm_average)
    


            
if __name__ == '__main__':
    
    main()