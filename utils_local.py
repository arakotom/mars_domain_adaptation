#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import torch.utils.data as data_utils
from sklearn.metrics import balanced_accuracy_score
import numpy as np

def get_filename(setting,param,opt):
        
    filename = f"setting{setting:d}-param{param:d}-n_hidden{opt['n_hidden']:d}-ntrain{opt['n_train']:d}-nval{opt['n_val']:d}"
    filename += f"-train{opt['balance_train']}-val{opt['balance_val']}" 
    filename += f"-trans{opt['translation']}-nb_iter{opt['nb_iter']:d}-nb_iter_alg{opt['nb_iter_alg']:d}-gradscale{opt['grad_scale']:2.3f}"
    return filename


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def  create_data_loader(X,y, batch_size,drop_last=True):
    data = data_utils.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return data_utils.DataLoader(data, batch_size= batch_size, drop_last = drop_last,sampler = data_utils.sampler.RandomSampler(data))




#%%
def sinkhorn_torch(w1,w2,M,reg,dtype = 'torch.FloatTensor',cuda=False,nb_iter=10):
    # Compute sinkhorn iteration for OT

    
    K=torch.exp(-M/reg)
    ui = torch.ones(K.size(0))
    vi = torch.ones(K.size(1))   

    if cuda:
        K = K.cuda()
        ui = ui.cuda()
        vi = vi.cuda()
        w2 = w2.cuda()
        w1 = w1.cuda()
    else:
        K = K.cpu()
        ui = ui.cpu()
        vi = vi.cpu()
        w2 = w2.cpu()
        w1 = w1.cpu()
    for i in range(nb_iter):
        vi=w2/(K.t()@ui)
        ui=w1/(K@vi)
        
    # TODO proper expand with no memory expansion    
    G = ui.repeat(K.size(1),1).t()*K* vi.repeat(K.size(0),1)
    return G,K,ui,vi

def sinkhorn_emd(w1,w2,M,reg,dtype = 'torch.FloatTensor',cuda=False,nb_iter=10):
    # Compute EMD Unfinished
    
    M = M.detach().numpy()
    w1 = w1.detach().numpy()
    w2 = w2.detach().numpy()
    
    return

def dist_torch(x1,x2):
    x1p = x1.pow(2).sum(1).unsqueeze(1)
    x2p = x2.pow(2).sum(1).unsqueeze(1)
    prod_x1x2 = torch.mm(x1,x2.t())
    distance = x1p.expand_as(prod_x1x2) + x2p.t().expand_as(prod_x1x2) -2*prod_x1x2
    return distance #/x1.size(0)/x2.size(0) 

def unif(n):
    return torch.ones(n)/n


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



def extract_feature(data_loader, feat_extract):
        
    X = np.zeros((0,n_hidden))
    y = np.zeros(0)
    for (data,target) in data_loader:
        aux = feat_extract(data).detach().numpy()
        target = target.numpy()
        X = np.vstack((X,aux))
        y = np.hstack((y,target))
    return X, y

def extract_prototypes(X,y,n_clusters):
    n_hidden = X.shape[1]
    mean_mat = np.zeros((n_clusters,n_hidden))
    number_in_class = np.zeros(n_clusters)
    for i in range(n_clusters):
        mean_mat[i]= np.mean(X[y==i,:],axis=0)
        number_in_class[i] = np.sum(y==i)
    return mean_mat, number_in_class



def plot_data_frontier(X_train,X_test, y_train, y_test, net, method = 'pca', frontier=True,comment=''):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    dim = X_train.shape[1]       
    print(dim)
    feat_extract = net.get_feature_extractor()
    data_class = net.get_data_classifier()
    if dim == 2 and frontier:
        plt.figure(1)
        plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap= 'autumn',alpha=0.4)
        plt.scatter(X_test[:,0], X_test[:,1], c=y_test, cmap = 'winter', alpha=0.4)
    
    
        x_min, x_max = -4, 4
        y_min, y_max = -4, 4
        h = 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
        Z = np.c_[xx.ravel(), yy.ravel()]

        Z_feat = feat_extract((torch.from_numpy(np.atleast_2d(Z)).float()))
        Z_class = data_class(Z_feat)
        classe = Z_class.data.max(1)[1].numpy()
          
    
        classe = classe.reshape(xx.shape)
        plt.contour(xx, yy, classe, levels =10,  colors='r')
        plt.title(comment)        
 
        plt.show()
    
    x= torch.from_numpy(X_train).float()
    X_train_map =feat_extract(x).data.numpy()
    x = torch.from_numpy(X_test).float()
    X_test_map = feat_extract(x).data.numpy()
    
    if dim >= 2 and False:
        emb_all = np.vstack([X_train_map, X_test_map])
        if method == 'pca':
            proj = PCA(n_components=2)
        else:
            proj = TSNE(perplexity=30, n_components=2, init="pca", n_iter=3000)
        
        pca_emb = proj.fit_transform(emb_all)
        
        num = X_train.shape[0]
        plt.figure(2)
        plt.scatter(pca_emb[:num,0], pca_emb[:num,1], c=y_train, cmap='autumn', alpha=0.4)
        plt.scatter(pca_emb[num:,0], pca_emb[num:,1], c=y_test, cmap='winter', alpha=0.4)
        plt.title(comment)
