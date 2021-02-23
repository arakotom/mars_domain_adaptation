#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import sys

import numpy as np
import torch
import torch.optim as optim
import copy
from utils_local import  weight_init
import getopt

setting = 2
param = 0

# run experiment on source only, WDGRL with beta = 1and our method + clustering
modeltorun='101000001'


dtype = 'torch.FloatTensor'

opts, args = getopt.getopt(sys.argv[1:], "g:t:s:")
for opt, arg in opts:
    if opt == '-s':
        setting = int(arg)
    if opt == '-g':
        print('gpu')
        gpu = int(arg)
    if opt == '-t':
        modeltorun = arg
        print(modeltorun)

path_resultat = './resultat/visda/'


from compare_visda_setting import expe_setting as expe_setting
opt, filename = expe_setting(setting)
optsave = copy.deepcopy(opt)
optsave.pop('target_loader')
optsave.pop('source_loader')
optsave.pop('feat_extract')
optsave.pop('data_classifier')
optsave.pop('domain_classifier')
optsave.pop('domain_classifier_dann')
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



filename = f"{filename}-{modeltorun}"
nb_iter = 1
batch_size = opt['batch_size']

cuda = torch.cuda.is_available()
if cuda:
    gpu_id = gpu
    torch.cuda.set_device(gpu_id)
    print('cuda is ON')



beta_vec = [1] # the different beta that can be tried (use a list [0,1,...])

bc_source = np.zeros(nb_iter)
bc_dann  = np.zeros(nb_iter)
bc_wd  = np.zeros((nb_iter,len(beta_vec)))
bc_wdts  = np.zeros(nb_iter)
bc_wdtsadv  = np.zeros(nb_iter)
bc_wdtsadv_clus  = np.zeros(nb_iter)

MAP_source = np.zeros(nb_iter)
MAP_gmm = np.zeros(nb_iter)
MAP_dann  = np.zeros(nb_iter)
MAP_wd  = np.zeros((nb_iter,len(beta_vec)))
MAP_wdts  = np.zeros(nb_iter)
MAP_wdtsadv  = np.zeros(nb_iter)
MAP_wdtsadv_clus  = np.zeros(nb_iter)

bc_source_source = np.zeros(nb_iter)
bc_dann_source  = np.zeros(nb_iter)
bc_wd_source  = np.zeros((nb_iter,len(beta_vec)))
bc_wdts_source  = np.zeros(nb_iter)
bc_wdtsadv_source  = np.zeros(nb_iter)
bc_wdtsadv_clus_source  = np.zeros(nb_iter)

MAP_source_source = np.zeros(nb_iter)
MAP_dann_source  = np.zeros(nb_iter)
MAP_wd_source  = np.zeros((nb_iter,len(beta_vec)))
MAP_wdts_source  = np.zeros(nb_iter)
MAP_wdtsadv_source  = np.zeros(nb_iter)
MAP_wdtsadv_clus_source  = np.zeros(nb_iter)


for it in range(nb_iter):
    print(it)
    np.random.seed(it)
    torch.manual_seed(it)
    torch.cuda.manual_seed(it)


    source_loader = opt['source_loader']
    target_loader = opt['target_loader']

    
    feat_extract_init = opt['feat_extract']
    data_class_init = opt['data_classifier']
    domain_class_init = opt['domain_classifier']
    domain_class_dann_init = opt['domain_classifier_dann']
    
    feat_extract_init.apply(weight_init)
    data_class_init.apply(weight_init)
    domain_class_init.apply(weight_init)
    domain_class_dann_init.apply(weight_init)
    
    

    #%%
    # ------------------------------------------------------------------------
    # Source only
    # ------------------------------------------------------------------------
    if int(modeltorun[0])== 1:
        from ClassDann import DANN

        
        feat_extract_source = copy.deepcopy(feat_extract_init)
        data_class_source = copy.deepcopy(data_class_init)
        domain_class_source = copy.deepcopy(domain_class_dann_init)

        source = DANN(feat_extract_source, data_class_source,domain_class_source, source_loader,target_loader,batch_size,
                                  cuda = cuda)
        source.set_optimizer_feat_extractor(optim.Adam(source.feat_extractor.parameters(),lr=opt['lr'],betas=(0.5, 0.999)))
        source.set_optimizer_data_classifier(optim.Adam(source.data_classifier.parameters(),lr=opt['lr'],betas=(0.5, 0.999)))
        source.set_optimizer_domain_classifier(optim.Adam(source.grl_domain_classifier.parameters(),lr=opt['lr'],betas=(0.5, 0.999)))
        source.set_nbiter(opt['nb_iter_alg'] )    

        source.set_grad_scale(opt['grad_scale'])
        source.set_epoch_to_start_align(opt['nb_iter_alg'] )
        source.set_lr_decay_epoch(-1)
        source.fit()
        
    
    
        bc_source[it],MAP_source[it] = source.evaluate_data_classifier(target_loader)
        bc_source_source[it],MAP_source_source[it] = source.evaluate_data_classifier(source_loader)


        
        from ClassWDTS_Adv import estimate_label_proportion_GMM
    
        MAP_gmm[it] = estimate_label_proportion_GMM(source_loader,target_loader,source.feat_extractor,cuda,n_clusters=opt['nb_class']
                                                ,prediction=True,n_max=5000);


    #%%
    # ------------------------------------------------------------------------
    # Domain adaptation with DANN
    # ------------------------------------------------------------------------
    if int(modeltorun[1])== 1:
        from ClassDann import DANN
        
        
        

        feat_extract_dann = copy.deepcopy(feat_extract_init)
        data_class_dann = copy.deepcopy(data_class_init)
        domain_class_dann = copy.deepcopy(domain_class_dann_init)

        dann = DANN(feat_extract_dann, data_class_dann,domain_class_dann, source_loader,target_loader,batch_size,
                                  cuda = cuda)
        dann.set_optimizer_feat_extractor(optim.Adam(dann.feat_extractor.parameters(),lr=opt['lr'],betas=(0.5, 0.999)))
        dann.set_optimizer_data_classifier(optim.Adam(dann.data_classifier.parameters(),lr=opt['lr'],betas=(0.5, 0.999)))
        dann.set_optimizer_domain_classifier(optim.Adam(dann.grl_domain_classifier.parameters(),lr=opt['lr'],betas=(0.5, 0.999)))
        dann.set_nbiter(opt['nb_iter_alg'] )    
        dann.set_grad_scale(opt['grad_scale'])
        dann.set_epoch_to_start_align(opt['start_align'])
        dann.set_lr_decay_epoch(-1)
        dann.fit()
        
    
    
        bc_dann[it],MAP_dann[it] = dann.evaluate_data_classifier(target_loader)
        bc_dann_source[it],MAP_dann_source[it] = dann.evaluate_data_classifier(source_loader)


    
    #%%
    # ------------------------------------------------------------------------
    # DA with weighted adversarial wasserstein à la Lipton ICML 2019
    # ------------------------------------------------------------------------
    print(modeltorun)
    list_run = [int(i) for i in modeltorun[2:7]]
    if  sum(list_run) >0  :
        
  
        
        from ClassWDGRL import WDGRL
        
        # create sub-networks
        for k, todo in enumerate(list_run):
     
            if todo == 1 :
                beta = beta_vec[k]
                print('beta:',beta)
                feat_extract_wd = copy.deepcopy(feat_extract_init)
                data_class_wd = copy.deepcopy(data_class_init)
                domain_class_wd = copy.deepcopy(domain_class_init)
                # compile model and fit
                wdgrl = WDGRL(feat_extract_wd, data_class_wd, domain_class_wd, source_loader,target_loader,
                                          cuda = cuda, grad_scale = 1 )
                wdgrl.set_optimizer_feat_extractor(optim.Adam(wdgrl.feat_extractor.parameters(),lr = opt['lr'],betas=(0.5, 0.999)))
                wdgrl.set_optimizer_data_classifier(optim.Adam(wdgrl.data_classifier.parameters(),lr = opt['lr'],betas=(0.5, 0.999)))
                wdgrl.set_optimizer_domain_classifier(optim.Adam(wdgrl.domain_classifier.parameters(),lr = opt['lr'],betas=(0.5, 0.999)))
                
                wdgrl.set_nbiter(opt['nb_iter_alg'] )
                wdgrl.set_epoch_to_start_align(opt['start_align'])
                wdgrl.set_iter_domain_classifier(opt['iter_domain'])                
                wdgrl.set_grad_scale(opt['grad_scale'])
                wdgrl.set_lr_decay_epoch(-1)
                wdgrl.set_beta_ratio(beta)
             
                wdgrl.fit()


                bc_wd[it,k],MAP_wd[it,k] = wdgrl.evaluate_data_classifier(target_loader)
                bc_wd_source[it],MAP_wd_source[it] = wdgrl.evaluate_data_classifier(source_loader)


#%%
    if int(modeltorun[7])== 1:
        do_wdtsadv = True
        print('adv GMM')
        from ClassWDTS_Adv import WDTS_Adv
        
        # create sub-networks
        feat_extract_wdtsadv = copy.deepcopy(feat_extract_init)
        data_class_wdtsadv =  copy.deepcopy(data_class_init)
        domain_class_wdtsadv = copy.deepcopy(domain_class_init)
    
        # compile model and fit
        wdtsadv = WDTS_Adv(feat_extract_wdtsadv, data_class_wdtsadv, domain_class_wdtsadv, source_loader,target_loader,
                                  cuda = cuda, grad_scale = 1 )
        wdtsadv.set_optimizer_feat_extractor(optim.Adam(wdtsadv.feat_extractor.parameters(),lr = opt['lr'],betas=(0.5, 0.999)))
        wdtsadv.set_optimizer_data_classifier(optim.Adam(wdtsadv.data_classifier.parameters(),lr =opt['lr'],betas=(0.5, 0.999)))
        wdtsadv.set_optimizer_domain_classifier(optim.Adam(wdtsadv.domain_classifier.parameters(),lr = opt['lr'],betas=(0.5, 0.999)))

        wdtsadv.set_n_class(opt['nb_class'])
        wdtsadv.set_compute_cluster_every(opt['cluster_every'])
        wdtsadv.set_cluster_param(opt['cluster_param'])
        wdtsadv.set_nbiter(opt['nb_iter_alg'] )
        wdtsadv.set_epoch_to_start_align(opt['start_align'])
        wdtsadv.set_proportion_method('gmm')

        wdtsadv.set_iter_domain_classifier(opt['iter_domain'])
        wdtsadv.set_grad_scale(opt['grad_scale'])
        wdtsadv.set_gamma(opt['gamma_wdts']) 
        wdtsadv.set_lr_decay_epoch(-1)
        wdtsadv.fit()
        
        bc_wdtsadv[it],MAP_wdtsadv[it] = wdtsadv.evaluate_data_classifier(target_loader)
        bc_wdtsadv_source[it],MAP_wdtsadv_source[it] = wdtsadv.evaluate_data_classifier(source_loader)
#%%
    if int(modeltorun[8])== 1:
        do_wdtsadv = True
        print('adv_clust')
        from ClassWDTS_Adv import WDTS_Adv
        
        # create sub-networks
        feat_extract_wdtsadv = copy.deepcopy(feat_extract_init)
        data_class_wdtsadv =  copy.deepcopy(data_class_init)
        domain_class_wdtsadv = copy.deepcopy(domain_class_init)
    
        # compile model and fit
        wdtsadv_clus = WDTS_Adv(feat_extract_wdtsadv, data_class_wdtsadv, domain_class_wdtsadv, source_loader,target_loader,
                                  cuda = cuda, grad_scale = 1 )
        wdtsadv_clus.set_optimizer_feat_extractor(optim.Adam(wdtsadv_clus.feat_extractor.parameters(),lr = opt['lr'],betas=(0.5, 0.999)))
        wdtsadv_clus.set_optimizer_data_classifier(optim.Adam(wdtsadv_clus.data_classifier.parameters(),lr =opt['lr'],betas=(0.5, 0.999)))
        wdtsadv_clus.set_optimizer_domain_classifier(optim.Adam(wdtsadv_clus.domain_classifier.parameters(),lr = opt['lr'],betas=(0.5, 0.999)))

        wdtsadv_clus.set_n_class(opt['nb_class'])
        wdtsadv_clus.set_compute_cluster_every(opt['cluster_every'])
        wdtsadv_clus.set_cluster_param(opt['cluster_param'])
        wdtsadv_clus.set_nbiter(opt['nb_iter_alg'] )
        wdtsadv_clus.set_epoch_to_start_align(opt['start_align'])

        wdtsadv_clus.set_iter_domain_classifier(opt['iter_domain'])
        wdtsadv_clus.set_grad_scale(opt['grad_scale'])
        wdtsadv_clus.set_gamma(opt['gamma_wdts']) 
        wdtsadv_clus.set_lr_decay_epoch(-1)
        wdtsadv_clus.fit()
        
        bc_wdtsadv_clus[it],MAP_wdtsadv_clus[it] = wdtsadv_clus.evaluate_data_classifier(target_loader)
        bc_wdtsadv_clus_source[it],MAP_wdtsadv_clus_source[it] = wdtsadv_clus.evaluate_data_classifier(source_loader)




#%%
    M_bc_wdts = bc_wdts[:it+1].mean()
    M_bc_wdtsadv = bc_wdtsadv[:it+1].mean()
    M_bc_wd = bc_wd[:it+1].mean()
    M_bc_dann = bc_dann[:it+1].mean()
    M_bc_source = bc_source[:it+1].mean()
    
    M_MAP_wdts = MAP_wdts[:it+1].mean()
    M_MAP_wdtsadv = MAP_wdtsadv[:it+1].mean()
    M_MAP_wd = MAP_wd[:it+1].mean()
    M_MAP_dann = MAP_dann[:it+1].mean()
    M_MAP_source = MAP_source[:it+1].mean()
    
    print(MAP_wdtsadv)
    print(MAP_wd)
    print(f"BC :  {M_bc_wdts:2.2f} ADV: {M_bc_wdtsadv:2.2f} wd:  {M_bc_wd:2.2f}  dann:  {M_bc_dann:2.2f} source    {M_bc_source:2.2f}")
    print(f"MAP :  {M_MAP_wdts:2.2f} ADV: {M_MAP_wdtsadv:2.2f}  wd:  {M_MAP_wd:2.2f}  dann:  {M_MAP_dann:2.2f} source    {M_MAP_source:2.2f}")

    

