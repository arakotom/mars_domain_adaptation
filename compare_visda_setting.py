#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import sys

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
    
from visda_shift import get_visda_shift
    
def expe_setting(setting,param=True):   
    if setting == 1:
        from compare_visda_models import DataClassifier, FeatureExtractor
        from compare_visda_models import DomainClassifier, DomainClassifierDANN

        total_sample = 6000 
        batch_size = 120
        ratio = [0.33,0.33,0.34]
        opt = {}
        

        opt['source_loader'] = get_visda_shift(train=True,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio)
        opt['target_loader'] = get_visda_shift(train=False,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio)
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractor()
        opt['data_classifier'] = DataClassifier()
        opt['domain_classifier'] = DomainClassifier()
        opt['domain_classifier_dann'] = DomainClassifierDANN()
        opt['lr'] = 0.0001
        opt['nb_iter_alg'] = 200
        opt['grad_scale'] = 0.01
        opt['wdgrl_grad_down_factor']=10
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['cluster_param']='ward'
        opt['nb_class'] = 3
        opt['cluster_every'] = 10
        opt['gamma_wdts'] = 10

        filename = f"visda-setting{setting}"

        return opt, filename
    if setting == 2:
        from compare_visda_models import DataClassifier, FeatureExtractor
        from compare_visda_models import DomainClassifier, DomainClassifierDANN

        total_sample =  6000

        batch_size = 120
        ratio = [0.4,0.2,0.4]
        ratio_2 = [0.15,0.7,0.15]

        opt = {}
        

        opt['source_loader'] = get_visda_shift(train=True,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio)
        opt['target_loader'] = get_visda_shift(train=False,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio_2)
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractor()
        opt['data_classifier'] = DataClassifier()
        opt['domain_classifier'] = DomainClassifier()
        opt['domain_classifier_dann'] = DomainClassifierDANN()
        opt['lr'] = 0.0001
        opt['nb_iter_alg'] = 200
        opt['grad_scale'] = 0.01
        opt['wdgrl_grad_down_factor']=10
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['cluster_param']='ward'
        opt['nb_class'] = 3
        opt['cluster_every'] = 10
        opt['gamma_wdts'] = 10

        filename = f"visda-setting{setting}"

        return opt, filename

    if setting == 3:
        from compare_visda_models import DataClassifier, FeatureExtractor
        from compare_visda_models import DomainClassifier, DomainClassifierDANN
        total_sample = 6000 

        batch_size = 120       
        ratio = [0.4,0.2,0.4]
        ratio_2 = [0.1,0.8,0.1]  
        opt = {}
        

        opt['source_loader'] = get_visda_shift(train=True,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio)
        opt['target_loader'] = get_visda_shift(train=False,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio_2)
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractor()
        opt['data_classifier'] = DataClassifier()
        opt['domain_classifier'] = DomainClassifier()
        opt['domain_classifier_dann'] = DomainClassifierDANN()
        opt['lr'] = 0.0001
        opt['nb_iter_alg'] = 200
        opt['grad_scale'] = 0.01
        opt['wdgrl_grad_down_factor']=10
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['cluster_param']='ward'
        opt['nb_class'] = 3
        opt['cluster_every'] = 10
        opt['gamma_wdts'] = 10

        filename = f"visda-setting{setting}"

        return opt, filename

    if setting == 4:
        from compare_visda_models import DataClassifier, FeatureExtractor
        from compare_visda_models import DomainClassifier, DomainClassifierDANN
        total_sample = 6000 
        batch_size = 120       
        ratio = [0.4,0.2,0.4]
        ratio_2 = [0.2,0.6,0.2]  
        opt = {}
        

        opt['source_loader'] = get_visda_shift(train=True,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio)
        opt['target_loader'] = get_visda_shift(train=False,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio_2)
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractor()
        opt['data_classifier'] = DataClassifier()
        opt['domain_classifier'] = DomainClassifier()
        opt['domain_classifier_dann'] = DomainClassifierDANN()
        opt['lr'] = 0.0001
        opt['nb_iter_alg'] = 200
        opt['grad_scale'] = 0.01
        opt['wdgrl_grad_down_factor']=10
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['cluster_param']='ward'
        opt['nb_class'] = 3
        opt['cluster_every'] = 10
        opt['gamma_wdts'] = 10

        filename = f"visda-setting{setting}"

        return opt, filename
    if setting == 5:
        from compare_visda_models import DataClassifier, FeatureExtractor
        from compare_visda_models import DomainClassifier, DomainClassifierDANN
        total_sample = 6000 

        batch_size = 120       
        ratio = [0.6,0.2,0.2]
        ratio_2 = [0.2,0.2,0.6]  
        opt = {}
        

        opt['source_loader'] = get_visda_shift(train=True,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio)
        opt['target_loader'] = get_visda_shift(train=False,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio_2)
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractor()
        opt['data_classifier'] = DataClassifier()
        opt['domain_classifier'] = DomainClassifier()
        opt['domain_classifier_dann'] = DomainClassifierDANN()
        opt['lr'] = 0.0001
        opt['nb_iter_alg'] = 200
        opt['grad_scale'] = 0.01
        opt['wdgrl_grad_down_factor']=10
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['cluster_param']='ward'
        opt['nb_class'] = 3
        opt['cluster_every'] = 10
        opt['gamma_wdts'] = 10

        filename = f"visda-setting{setting}"

        return opt, filename
    if setting == 6:
        from compare_visda_models import DataClassifier, FeatureExtractor
        from compare_visda_models import DomainClassifier, DomainClassifierDANN
        total_sample = 6000 

        batch_size = 120       
        ratio = [0.6,0.2,0.2]
        ratio_2 = [0.3,0.3,0.4]  
        opt = {}
        

        opt['source_loader'] = get_visda_shift(train=True,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio)
        opt['target_loader'] = get_visda_shift(train=False,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio_2)
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractor()
        opt['data_classifier'] = DataClassifier()
        opt['domain_classifier'] = DomainClassifier()
        opt['domain_classifier_dann'] = DomainClassifierDANN()
        opt['lr'] = 0.0001
        opt['nb_iter_alg'] = 200
        opt['grad_scale'] = 0.01
        opt['wdgrl_grad_down_factor']=10
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['cluster_param']='ward'
        opt['nb_class'] = 3
        opt['cluster_every'] = 10
        opt['gamma_wdts'] = 10

        filename = f"visda-setting{setting}"

        return opt, filename
    if setting == 7:
        from compare_visda_models import DataClassifier, FeatureExtractor
        from compare_visda_models import DomainClassifier, DomainClassifierDANN
        total_sample = 6000 

        batch_size = 120       
        ratio = [0.65,0.2,0.15]
        ratio_2 = [0.2,0.65,0.15]  
        opt = {}
        

        opt['source_loader'] = get_visda_shift(train=True,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio)
        opt['target_loader'] = get_visda_shift(train=False,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio_2)
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractor()
        opt['data_classifier'] = DataClassifier()
        opt['domain_classifier'] = DomainClassifier()
        opt['domain_classifier_dann'] = DomainClassifierDANN()
        opt['lr'] = 0.0001
        opt['nb_iter_alg'] = 200
        opt['grad_scale'] = 0.01
        opt['wdgrl_grad_down_factor']=10
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['cluster_param']='ward'
        opt['nb_class'] = 3
        opt['cluster_every'] = 10
        opt['gamma_wdts'] = 10

        filename = f"visda-setting{setting}"

        return opt, filename
     
    
    if setting == 10:
        from compare_visda_models import DataClassifierFull, FeatureExtractor
        from compare_visda_models import DomainClassifier, DomainClassifierDANN

        total_sample = 24000
        batch_size = 240       
        ratio = [1/12]*12
        ratio_2 = [1/12]*12 
        opt = {}
        
        if param == True:
            opt['source_loader'] = get_visda_shift(train=True,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio,
                                           classe_vec=[0,1,2,3,4,5,6,7,8,9,10,11])
            opt['target_loader'] = get_visda_shift(train=False,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio_2,
                                          classe_vec=[0,1,2,3,4,5,6,7,8,9,10,11])    
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractor()
        opt['data_classifier'] = DataClassifierFull()
        opt['domain_classifier'] = DomainClassifier()
        opt['domain_classifier_dann'] = DomainClassifierDANN()
        opt['lr'] = 0.0001
        opt['nb_iter_alg'] = 200
        opt['grad_scale'] = 0.01
        opt['wdgrl_grad_down_factor']=10
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['cluster_param']='ward'
        opt['nb_class'] = 12
        opt['cluster_every'] = 10
        opt['gamma_wdts'] = 10

        filename = f"visda-setting{setting}"
    
        return opt, filename
    
    if setting == 11:
        from compare_visda_models import DataClassifierFull, FeatureExtractor
        from compare_visda_models import DomainClassifier, DomainClassifierDANN
        total_sample = 24000

        batch_size = 240       
        ratio = [1/12]*12
        ratio_2 = [0.15,0.15,0.15,0.15,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]  
        opt = {}
        

        if param == True:
            opt['source_loader'] = get_visda_shift(train=True,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio,
                                           classe_vec=[0,1,2,3,4,5,6,7,8,9,10,11])
            opt['target_loader'] = get_visda_shift(train=False,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio_2,
                                          classe_vec=[0,1,2,3,4,5,6,7,8,9,10,11])   
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractor()
        opt['data_classifier'] = DataClassifierFull()
        opt['domain_classifier'] = DomainClassifier()
        opt['domain_classifier_dann'] = DomainClassifierDANN()
        opt['lr'] = 0.0001
        opt['nb_iter_alg'] = 200
        opt['grad_scale'] = 0.01
        opt['wdgrl_grad_down_factor']=10
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['cluster_param']='ward'
        opt['nb_class'] = 12
        opt['cluster_every'] = 10
        opt['gamma_wdts'] = 10

        filename = f"visda-setting{setting}"
    
        return opt, filename
    
    if setting == 12:
        from compare_visda_models import DataClassifierFull, FeatureExtractor
        from compare_visda_models import DomainClassifier, DomainClassifierDANN
        total_sample = 24000   #9600
        batch_size = 240       
        ratio = [1/12]*12
        ratio_2 = [0.2,0.2,0.1,0.1,0.1,0.1,0.05,0.03,0.03,0.03,0.03,0.03]  
        opt = {}
        

        if param == True:
            opt['source_loader'] = get_visda_shift(train=True,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio,
                                           classe_vec=[0,1,2,3,4,5,6,7,8,9,10,11])
            opt['target_loader'] = get_visda_shift(train=False,batch_size=batch_size,drop_last=False,total_sample=total_sample,ratio=ratio_2,
                                          classe_vec=[0,1,2,3,4,5,6,7,8,9,10,11])     
        opt['nb_iter'] = 20
        opt['feat_extract']= FeatureExtractor()
        opt['data_classifier'] = DataClassifierFull()
        opt['domain_classifier'] = DomainClassifier()
        opt['domain_classifier_dann'] = DomainClassifierDANN()
        opt['lr'] = 0.0001
        opt['nb_iter_alg'] = 200
        opt['grad_scale'] = 0.01
        opt['wdgrl_grad_down_factor']=10
        opt['iter_domain'] = 5
        opt['start_align'] = 10
        opt['batch_size'] = batch_size
        opt['cluster_param']='ward'
        opt['nb_class'] = 12
        opt['cluster_every'] = 10
        opt['gamma_wdts'] = 10

        filename = f"visda-setting{setting}"
    
        return opt, filename

if __name__ == '__main__':

    opt, filename = expe_setting(3)
    
    
    aa= opt['source_loader']
    aa.dataset.tensors[0].shape
    aa= opt['target_loader']
    aa.dataset.tensors[0].shape
    
    sum(aa.dataset.tensors[1]==0)

    sum(aa.dataset.tensors[1]==1)

    sum(aa.dataset.tensors[1]==2)
    
