
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils_local import create_data_loader
from utils_local import weight_init

import copy

dtype = 'torch.FloatTensor'
#


class FeatureExtractor(nn.Module):

    def __init__(self,dim):
        super(FeatureExtractor, self).__init__()        
        self.fc1 = nn.Linear(dim, n_hidden, bias = True)
        self.fc2 = nn.Linear(n_hidden, n_hidden, bias = True)
        self.fc3 = nn.Linear(n_hidden, n_hidden, bias = True)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x 

class DataClassifier(nn.Module):
    def __init__(self):

        super(DataClassifier, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_hidden, bias = True)
        self.fc2 = nn.Linear(n_hidden, n_hidden, bias = True)

    def forward(self, input):

        x = input.view(input.size(0), -1)
        x = F.relu(self.fc1(x))
        x = (self.fc2(x))

        return x

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()        
        self.fc1 = nn.Linear(n_hidden, n_hidden,bias = False)    
        self.fc2 = nn.Linear(n_hidden, n_hidden,bias = False)     
        self.fc3 = nn.Linear(n_hidden, 1,bias = False)     

    def forward(self, input):        
        x = input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = (self.fc3(x))
        return x


class DomainClassifierDANN(nn.Module):
    def __init__(self):
        super(DomainClassifierDANN, self).__init__()        
        self.fc1 = nn.Linear(n_hidden, n_hidden,bias = False)    
        self.fc2 = nn.Linear(n_hidden, n_hidden,bias = False)     
        self.fc3 = nn.Linear(n_hidden, 2,bias = False)     

    def forward(self, input):        
        x = input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = (self.fc3(x))
        return x


def loop_iterable(iterable):
    while True:
        yield from iterable
     

def to_one_hot(labels,num_classes,cuda = False):
    labels = labels.reshape(-1, 1)
    if cuda:
        one_hot_target = (labels == torch.arange(num_classes).float())
    else:
        one_hot_target = (labels.cpu() == torch.arange(num_classes).float())
    return one_hot_target





def make_multi_blobs(n_train,centers,corr_vec):
    X = np.zeros((0,dim))
    y = np.zeros(0)
    for i in range(len(centers)):
        cov = np.array([ [corr_vec[i][0],0],[0,corr_vec[i][1]]])
        aux = np.random.multivariate_normal(centers[i],cov,n_train[i])
        X = np.vstack((X,aux))
        y = np.hstack((y,np.ones(n_train[i])*i))
    return X,y




#%% 
"""
generate data with target shift
"""
n_hidden = 200
n_train = 10
n_val = 10
dim = 2
batch_size = 120
cuda = torch.cuda.is_available()
fixed_seed = False
if fixed_seed :
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)


from compare_toy_setting import expe_setting as expe_setting

setting = 3
param = 5
opt = expe_setting(setting,param)
print('cuda',cuda)

n_train = opt['balance_train']*n_train*3
n_val = opt['balance_val']*n_val*3

X_train,y_train = make_multi_blobs(n_train,opt['centers'],opt['corr_vec'])
centers_val  = opt['centers'] + opt['translation']
X_val,y_val = make_multi_blobs(n_val,centers_val,opt['corr_vec'])





source_loader = create_data_loader(X_train,y_train, batch_size= batch_size,drop_last=False)
target_loader = create_data_loader(X_val,y_val, batch_size= batch_size,drop_last=False)



#%%

feat_extract_init = FeatureExtractor(dim)
data_class_init = DataClassifier()
domain_class_init = DomainClassifier()
domain_class_dann_init = DomainClassifierDANN()

feat_extract_init.apply(weight_init)
data_class_init.apply(weight_init)
domain_class_init.apply(weight_init)
domain_class_dann_init.apply(weight_init)





#%%
"""
DA with dann
"""   
bc_dann,MAP_dann = 0,0
do_dann = False
if 1:
    do_dann = True
    from ClassDann import DANN


    feat_extract_dann = copy.deepcopy(feat_extract_init)
    data_class_dann = copy.deepcopy(data_class_init)
    domain_class_dann = copy.deepcopy(domain_class_dann_init)

    
    dann = DANN(feat_extract_dann, data_class_dann,domain_class_dann, source_loader,target_loader,batch_size,
                              cuda = cuda)
    dann.set_optimizer_feat_extractor(optim.SGD(dann.feat_extractor.parameters(),lr= opt['lr']))
    dann.set_optimizer_data_classifier(optim.SGD(dann.data_classifier.parameters(),lr= opt['lr']))
    dann.set_optimizer_domain_classifier(optim.SGD(dann.grl_domain_classifier.parameters(),lr= opt['lr']))
    dann.set_nbiter(opt['nb_iter_alg'])    
    dann.set_epoch_to_start_align(2000)
    dann.set_lr_decay_epoch(-1)
    dann.fit()
    
    

    bc_dann,MAP_dann = dann.evaluate_data_classifier(target_loader)


#%%
if 1:
    do_wdtsadv = True
    from ClassWDTS_Adv import WDTS_Adv
    
    # create sub-networks
    feat_extract_wdtsadv = copy.deepcopy(feat_extract_init)
    data_class_wdtsadv = copy.deepcopy(data_class_init)
    domain_class_wdtsadv = copy.deepcopy(domain_class_init)


    # compile model and fit
    wdtsadv = WDTS_Adv(feat_extract_wdtsadv, data_class_wdtsadv, domain_class_wdtsadv, source_loader,target_loader,
                              cuda = cuda, grad_scale = 1 )

    wdtsadv.set_optimizer_feat_extractor(optim.SGD(wdtsadv.feat_extractor.parameters(),lr = opt['lr']))
    wdtsadv.set_optimizer_data_classifier(optim.SGD(wdtsadv.data_classifier.parameters(),lr = opt['lr']))
    wdtsadv.set_optimizer_domain_classifier(optim.SGD(wdtsadv.domain_classifier.parameters(),lr =  opt['lr']))

    
    wdtsadv.set_n_class(3)            
    wdtsadv.set_compute_cluster_every(10)
    wdtsadv.set_cluster_param(opt['cluster_param'])
    wdtsadv.set_nbiter(opt['nb_iter_alg'] )
    wdtsadv.set_proportion_method('gmm')

    wdtsadv.set_epoch_to_start_align(100)
    wdtsadv.set_iter_domain_classifier(10)
    wdtsadv.set_grad_scale(opt['grad_scale'])

    wdtsadv.set_lr_decay_epoch(-1)
    
    
    wdtsadv.fit()
    
    bc_wdtsadv,MAP_wdtsadv = wdtsadv.evaluate_data_classifier(target_loader)

    
