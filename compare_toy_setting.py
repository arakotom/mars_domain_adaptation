#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np

def expe_setting(setting,param):   
    #print(setting)    
    if setting == 0 :
        opt = {}
        opt['n_hidden'] = 200
        opt['n_train'] = 10
        opt['n_val'] = 10
        opt['batch_size'] = 120
        opt['nb_iter'] = 20
        opt['nb_iter_alg'] = 500
        opt['balance_train']= np.array([33,33,34])
        opt['balance_val']= np.array([33,33,34])
        opt['lr'] = 0.01
        opt['centers']=np.array([ [0.25,0.0], [0,1], [1,1.5]])
        opt['corr_vec'] = [[0.002,0,0,0.005], [0.005,0,0,0.01], [0.05,0,0,0.05]] 
        opt['nb_class'] = 3
        opt['grad_scale']=0.05*param
        opt['cluster_param'] = 'ward'

        opt['translation'] = np.array([[-0.05,-0.05],[-0.05,-0.05],[-0.1,-0.0005]])*param

        return opt
    if setting == 1 :
        opt = {}
        opt = expe_setting(0,param)
        opt['balance_val']= np.array([60,20,20])
        return opt
    if setting == 2 :
        opt = {}
        opt = expe_setting(0,param)
        opt['grad_scale']=0.05*param
        opt['balance_val']= np.array([80,10,10])
        return opt
    
    #-------------------------------------------------------------------------
    # evaluate performance of model for varying imbalance ratio with respect to
    # the majority class
    #
    # setting 4 is a harder problem
    if setting == 3 :
        opt = {}
        trans = 8
        opt = expe_setting(0,trans)
        
        main = int(34 + 6.4*param)
        other = int((100 - main)//2) 
        other1 = int(100 - main - other)
        opt['balance_val']= np.array([main,other,other1])

        return opt
    if setting == 4 :
        opt = {}
        opt = expe_setting(3,param)
        opt['corr_vec'] = [[0.025,0,0,0.025], [0.025,0,0,0.05], [0.05,0,0,0.05]] 

        return opt
    if setting == 5 : # use for mean matching figure
        opt = {}
        opt = expe_setting(2,param)
        opt['corr_vec'] = [[0.025,0,0,0.025], [0.025,0,0,0.05], [0.05,0,0,0.05]] 

        return opt

    if setting == 6 :
        opt = {}
        opt = expe_setting(3,param)
        opt['corr_vec'] = [[0.035,0,0,0.035], [0.035,0,0,0.35], [0.05,0,0,0.05]] 
        
        return opt