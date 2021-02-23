#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""
import numpy as np
from time import time as tick 
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim
from utils_local import weight_init, sinkhorn_torch, dist_torch,unif

from torch.autograd import grad
from sklearn.metrics import balanced_accuracy_score

def loop_iterable(iterable):
    while True:
        yield from iterable
def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad



def to_one_hot(labels,num_classes,cuda = False):
    labels = labels.reshape(-1, 1)
    if cuda:
        one_hot_target = (labels == torch.arange(num_classes).float())
    else:
        one_hot_target = (labels.cpu() == torch.arange(num_classes).float())
    return one_hot_target

def extract_feature(data_loader, feat_extract,cuda):
    with torch.no_grad():
    
        for i, (data,target) in enumerate(data_loader):
            if cuda:
                data = data.cuda()
            aux = feat_extract(data).detach().cpu().numpy()
            target = target.cpu().numpy()
            if i== 0:
                X = aux
                y = target
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

#%%
def estimate_label_proportion(source_loader,target_loader,feat_extract,cuda,n_clusters,cluster_param):
    
    """Clustering target samples in the latent space and extracting
    prototypes (means of clusters)
    """    
    feat_extract.eval()
    #n_clusters = 3
    from sklearn.cluster import AgglomerativeClustering
    
    
    X_s,y_s = extract_feature(source_loader,feat_extract,cuda) 
    X_t,y_t = extract_feature(target_loader,feat_extract,cuda) 
    
    
    
    cluster = AgglomerativeClustering(n_clusters=n_clusters,linkage=cluster_param)
    label_t = cluster.fit_predict(X_t)
    #print(np.unique(label_t))
    mean_mat_S, num_in_class_S = extract_prototypes(X_s,y_s,n_clusters)
    mean_mat_T, num_in_class_T = extract_prototypes(X_t,label_t,n_clusters)
    
    """
    We assume that prototypes of classes have been transported in some in the feature
    space 
    """
    
    import ot
    M = ot.dist(mean_mat_S, mean_mat_T)
    M /= M.max()
    
    n_1 = n_clusters
    a = np.ones((n_1,)) / n_1
    b = np.ones((n_1,)) / n_1
    
    
    gamma = ot.emd(a,b,M)
    nb_sample_S = [ np.sum(y_s==i) for i in range(n_clusters) ]
    proportion_T = num_in_class_T/np.sum(num_in_class_T)
    assignement_source_to_target = gamma.argmax(axis=1)
    
    # proportions are arranged directly per class
    proportion_T = proportion_T[assignement_source_to_target]
    print(proportion_T,assignement_source_to_target)
    

    return proportion_T,nb_sample_S, assignement_source_to_target

#%%
def estimate_label_proportion_GMM(source_loader,target_loader,feat_extract,cuda,n_clusters,prediction=False,n_max=2000):
    
    """Clustering target samples in the latent space and extracting
    prototypes (means of clusters)
    """    
    feat_extract.eval()
    
    
    X_s,y_s = extract_feature(source_loader,feat_extract,cuda) 
    X_t,y_t = extract_feature(target_loader,feat_extract,cuda) 
    
    if X_s.shape[0]>n_max:
        X_s = X_s[0:n_max,:]
        X_t = X_t[0:n_max,:]
        y_s = y_s[0:n_max]
        y_t = y_t[0:n_max]

    gmm = GaussianMixture(n_components=n_clusters,n_init=10)# means_init = mean_mat_S)
    label_t = gmm.fit_predict(X_t)  
    mean_mat_S, num_in_class_S = extract_prototypes(X_s,y_s,n_clusters)
    mean_mat_T, num_in_class_T = extract_prototypes(X_t,label_t,n_clusters)
    
    """
    We assume that prototypes of classes have been transported in some in the feature
    space 
    """
    
    import ot
    M = ot.dist(mean_mat_S, mean_mat_T)
    M /= M.max()
    
    n_1 = n_clusters
    a = np.ones((n_1,)) / n_1
    b = np.ones((n_1,)) / n_1
    
    
    gamma = ot.emd(a,b,M)
    proportion_T = num_in_class_T/np.sum(num_in_class_T)
    assignement_source_to_target = gamma.argmax(axis=1)
    
    # proportions are arranged directly per class
    proportion_T = torch.from_numpy(proportion_T[assignement_source_to_target]).float()
    print(proportion_T,assignement_source_to_target)
    
    
    
    if prediction:
        #print('predition',gamma)
        X_t,y_t = extract_feature(target_loader,feat_extract,cuda) 
        label_t = gmm.predict(X_t)  

        assignement_target_to_source = gamma.argmax(axis=0)
        for i,yi in enumerate(label_t):
            label_t[i] = assignement_target_to_source[yi]
        MAP = balanced_accuracy_score(y_t,label_t)  
        return MAP
    
    return proportion_T

#%%
#

def _gradient_penalty(critic, real_data, generated_data):
    
    ## OLD 
    
    batch_size = real_data.size()[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand_as(real_data).to(device)
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = interpolated.requires_grad_()
    
    interpolated = interpolated.to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = critic(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = grad(outputs=prob_interpolated, inputs=interpolated,
                           #grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                           grad_outputs=torch.ones_like(prob_interpolated),
                           #prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    gradient_norm = gradients.norm(2, dim=1) + 1e-12
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty


def gradient_penalty(critic, h_s, h_t,cuda):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cuda :
        device = 'cuda';
    else:
        device = 'cpu';

    
    alpha = torch.rand(h_s.size(0), 1)
    alpha = (alpha.expand(h_s.size())).to(device)
    differences = h_t - h_s
    
    interpolates = (h_s + (alpha * differences))
    interpolates = torch.cat((interpolates,h_s,h_t),dim=0).requires_grad_()


    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()

    return gradient_penalty



class WDTS_Adv(object):
    
    def __init__(self, feat_extractor,data_classifier, domain_classifier,source_data_loader, target_data_loader,
                 grad_scale = 1,cuda = False, logger_file = None, eval_data_loader = None, wgan = False, 
                 T_batches = None, S_batches = None):
        self.feat_extractor = feat_extractor
        self.data_classifier = data_classifier
        self.domain_classifier = domain_classifier
        self.source_data_loader = source_data_loader
        self.target_data_loader = target_data_loader

        self.eval_domain_data =0 # argument of list of eval_data_loader to use as domain evaluation with source
        self.eval_reference = 0
        self.source_domain_label = 1
        self.test_domain_label = 0
        self.cuda = cuda
        self.nb_iter = 1000
        self.logger = logger_file
        self.criterion = nn.CrossEntropyLoss()
        self.lr_decay_epoch = -1
        self.lr_decay_factor = 0.5
        self.wgan = wgan
        self.clamp = 0.1
        self.filesave = None
        self.save_best = True
        self.epoch_to_start_align  = 100 # start aligning distrib at this epoch
        self.iter_domain_classifier = 10
        self.T_batches = T_batches
        self.gamma = 10
        self.grad_scale_0 = grad_scale
        self.grad_scale = grad_scale
        self.compute_cluster_every= 10
        self.proportion_method = 'cluster'

        self.domain_classifier = domain_classifier
        
        # these are the default
        self.optimizer_feat_extractor = optim.SGD(self.feat_extractor.parameters(),lr = 0.001)
        self.optimizer_data_classifier = optim.SGD(self.data_classifier.parameters(),lr = 0.001)
        self.optimizer_domain_classifier = optim.SGD(self.domain_classifier.parameters(),lr = 0.01)

    def set_compute_cluster_every(self,compute):
        self.compute_cluster_every = compute
    def set_proportion_method(self,method):
        self.proportion_method = method
        print('cluster_method',method)
    
    def set_lr_decay_epoch(self,decay_epoch):
        self.lr_decay_epoch = decay_epoch
    def set_iter_domain_classifier(self,iter_domain_classifier):    
        self.iter_domain_classifier = iter_domain_classifier
    def set_epoch_to_start_align(self, epoch_to_start_align):
        self.epoch_to_start_align = epoch_to_start_align
        self.epoch_to_start_align_target = epoch_to_start_align

    def set_gamma(self,new_gamma):
        self.gamma = new_gamma
    def set_grad_scale(self,new_grad_scale):
           self.grad_scale = new_grad_scale
    def set_cluster_param(self,new_cluster_param):
           self.cluster_param = new_cluster_param
    def set_filesave(self,filesave):
           self.filesave = filesave
    def show_grad_scale(self):
        print(self.grad_scale)
        return
    def set_n_class(self,n_class):
        self.n_class = n_class
        
    def set_optimizer_data_classifier(self, optimizer):
        self.optimizer_data_classifier = optimizer
    def set_optimizer_domain_classifier(self, optimizer):
        self.optimizer_domain_classifier = optimizer
    def set_optimizer_feat_extractor(self, optimizer):
        self.optimizer_feat_extractor = optimizer
    def set_nbiter(self, nb_iter):
        self.nb_iter = nb_iter
    def set_clamp(self,clamp_val):
        self.clamp = abs(clamp_val)
    def set_save_best(self,save_best):
        self.save_best = save_best
    def build_label_domain(self,size,label):
        label_domain = torch.LongTensor(size)       
        if self.cuda:
            label_domain = label_domain.cuda()
        
        label_domain.data.resize_(size).fill_(label)
        return label_domain
        
    def evaluate_data_classifier(self,data_loader, comments = ''):
        self.feat_extractor.eval()
        self.data_classifier.eval()
        
        test_loss = 0
        correct = 0
        y_pred = torch.Tensor()
        y_true = torch.zeros((0))
        for data, target in data_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            output_feat = self.feat_extractor(data)
            output = self.data_classifier(output_feat)
            test_loss += self.criterion(output, target).item()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            y_pred = torch.cat((y_pred,pred.float().cpu()))
            y_true = torch.cat((y_true,target.float().cpu()))
        MAP = balanced_accuracy_score(y_true,y_pred)  
        test_loss = test_loss
        test_loss /= len(data_loader) # loss function already averages over batch size  
        accur = correct.item() / (len(data_loader.dataset))

        print('{} Mean Loss:  {:.4f}, Accuracy: {}/{} ({:.0f}%) MAP :{:.4f}'.format(
                comments, test_loss, correct, len(data_loader.dataset),
                100*accur,MAP))
        
        if self.logger is not None:
            self.logger.info('{} Mean Loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(comments, test_loss, correct, len(data_loader.dataset),
                accur))
        return accur,MAP
    def evaluate_domain_classifier_class(self, data_loader, domain_label):
        self.feat_extractor.eval()
        self.data_classifier.eval()
        self.grl_domain_classifier.eval()

        loss = 0
        correct = 0
        for data, _ in data_loader:
            target = self.build_label_domain(data.size(0),domain_label)
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            output_feat = self.feat_extractor(data)
            output = self.grl_domain_classifier(output_feat)
            loss += self.criterion(output, target).data[0]
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()  
        return loss, correct
    
    def evaluate_domain_classifier(self):

        self.feat_extractor.eval()
        self.data_classifier.eval()
        self.grl_domain_classifier.eval()

        test_loss,correct = 0, 0
        test_loss, correct = self.evaluate_domain_classifier_class(self.source_data_loader, self.source_domain_label)
        loss, correct_a = self.evaluate_domain_classifier_class(self.eval_data_loader[self.eval_domain_data], self.test_domain_label)
        test_loss +=loss
        correct +=correct_a
        nb_source = len(self.source_data_loader.dataset) 
        nb_target = len(self. eval_data_loader[self.eval_domain_data].dataset) 
        nb_tot = nb_source + nb_target
        print('Domain: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, ( nb_source + nb_target ),
                100. * correct / (nb_source + nb_target )))
        if self.logger is not None:
             self.logger.info('Domain: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, ( nb_tot),
                100. * correct / nb_tot ))
        return correct / nb_tot 
    
    def fit(self):
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #print(device)
        if self.cuda:
            self.feat_extractor.cuda()
            self.data_classifier.cuda()
            self.domain_classifier.cuda()     
            device = 'cuda'
        else:
            device = 'cpu'

        k_critic = self.iter_domain_classifier
        gamma = self.gamma
        wd_clf = self.grad_scale
        k_prop = 1
        for epoch in range(self.nb_iter):
            S_batches = loop_iterable(self.source_data_loader)
            
            batch_iterator = zip(S_batches, loop_iterable(self.target_data_loader))
            batch_iterator_wass = zip(S_batches, loop_iterable(self.target_data_loader))

            iterations = len(self.source_data_loader)
            total_loss = 0
            total_accuracy = 0
            tic = tick()
            
            if epoch == self.epoch_to_start_align or \
                    (epoch > self.epoch_to_start_align and (epoch % self.compute_cluster_every)==0):
                #---------------------------------------------------------
                # estimate proportion
                #---------------------------------------------------------
                if self.proportion_method == 'cluster':
                    proportion_T, nb_sample_S, assignement_source_to_target = estimate_label_proportion(
                                    self.source_data_loader,self.target_data_loader,self.feat_extractor,self.cuda,
                                    self.n_class,self.cluster_param)
                else:
                    if k_prop == 1:
                        proportion_T = estimate_label_proportion_GMM(
                                    self.source_data_loader,self.target_data_loader,self.feat_extractor,self.cuda,
                                    self.n_class)
                    else:
                        proportion_T_aux = estimate_label_proportion_GMM(
                                    self.source_data_loader,self.target_data_loader,self.feat_extractor,self.cuda,
                                    self.n_class)
                        proportion_T = proportion_T*(k_prop-1)/k_prop + proportion_T_aux/k_prop
                k_prop +=1
                print(proportion_T)
            for i in range(iterations):
                #print(i,iterations)
                (source_x, source_y), (target_x, _) = next(batch_iterator)
                # Train critic
                set_requires_grad(self.feat_extractor, requires_grad=False)
                set_requires_grad(self.domain_classifier, requires_grad=True)
            
                source_x, target_x = source_x.to(device), target_x.to(device)
                source_y = source_y.to(device)

       
                if epoch > self.epoch_to_start_align:
                    #---------------------------------------------------------
                    # compute weighted wasserstein dual
                    #---------------------------------------------------------
                    set_requires_grad(self.feat_extractor, requires_grad=False)
                    set_requires_grad(self.domain_classifier, requires_grad=True)
                    total_loss = 0
                    for kk in range(k_critic):
                        (source_x, source_y), (target_x, _) = next(batch_iterator_wass)
                        source_x, target_x = source_x.to(device), target_x.to(device)
                        source_y = source_y.to(device)

                        if source_x.shape[0] == target_x.shape[0]:
                            with torch.no_grad():
                                h_s = self.feat_extractor(source_x).data.view(source_x.shape[0], -1)
                                h_t = self.feat_extractor(target_x).data.view(target_x.shape[0], -1)
                                
                            gp = gradient_penalty(self.domain_classifier, h_s, h_t,self.cuda)
                            critic_s = self.domain_classifier(h_s)
                            critic_t = self.domain_classifier(h_t)                                
                                
                            source_weight = torch.zeros((source_y.size(0),1)).to(device)
                            for j in range(self.n_class):
                                nb_sample = source_y.eq(j).nonzero().size(0) 
                                source_weight[source_y==j] = proportion_T[j]/nb_sample
                                
                            wasserstein_distance = (critic_s*source_weight).sum() - critic_t.mean()
                      

                            critic_cost = - wasserstein_distance + gamma*gp
                    
                            self.optimizer_domain_classifier.zero_grad()
                            critic_cost.backward()
                            self.optimizer_domain_classifier.step()
                    
                            total_loss += wasserstein_distance.item()
                # ------------------------------------------------------------
                # Train classifier ad feature extractor 
                # once discriminator has been learned
                #-------------------------------------------------------------
                set_requires_grad(self.feat_extractor, requires_grad=True)
                set_requires_grad(self.domain_classifier, requires_grad=False)
                
                (source_x, source_y), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.to(device), target_x.to(device)
                source_y = source_y.to(device)
                source_features = self.feat_extractor(source_x).view(source_x.shape[0], -1)
                target_features = self.feat_extractor(target_x).view(target_x.shape[0], -1)
        
                source_preds = self.data_classifier(source_features)
                clf_loss = self.criterion(source_preds, source_y)
                if epoch > self.epoch_to_start_align:
                    source_weight = torch.zeros((source_y.size(0),1)).to(device)
                    for j in range(self.n_class):
                        nb_sample = source_y.eq(j).nonzero().size(0) 
                        source_weight[source_y==j] = proportion_T[j]/nb_sample
                        
                    wasserstein_distance = (source_weight*self.domain_classifier(source_features)).sum() - self.domain_classifier(target_features).mean()       

                    loss = clf_loss + wd_clf * wasserstein_distance
                else:
                    wasserstein_distance = torch.zeros(1)
                    loss = clf_loss
                self.optimizer_feat_extractor.zero_grad()
                self.optimizer_data_classifier.zero_grad()
                loss.backward()
                self.optimizer_feat_extractor.step()
                self.optimizer_data_classifier.step()
                total_accuracy +=clf_loss.item()
                    
                    
            toc =  tick() - tic 
            print('\nWDADV Train Epoch: {} {:2.2f}s \tLoss: {:.6f} DistLoss:{:.6f}'.format(
                        epoch, toc, total_accuracy, total_loss))
            self.evaluate_data_classifier(self.source_data_loader)
            self.evaluate_data_classifier(self.target_data_loader)
        


        
    def get_feature_extractor(self):
        return self.feat_extractor
    def get_data_classifier(self):
        return self.data_classifier
    
    def save_perf(self):
        np.savez(self.filesave + '.npz' ,accuracy_train = self.perf_source.numpy(), accuracy_evaluation = self.perf_val.numpy(),
                 accuracy_domain = self.perf_domain.numpy())
        


def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=100,lr_decay_factor=0.5):
    """Decay current learning rate by a factor of 0.5 every lr_decay_epoch epochs."""
    init_lr = optimizer.param_groups[0]['lr']
    if epoch > 0 and (epoch % lr_decay_epoch == 0):
        lr = init_lr*lr_decay_factor
        print('\n LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer

