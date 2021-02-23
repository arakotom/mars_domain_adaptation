#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim
from time import time as tick 
from sklearn.metrics import balanced_accuracy_score



def  create_data_loader(X,y, batch_size):
    data = data_utils.TensorDataset(torch.from_numpy(X).float(), y)
    return data_utils.DataLoader(data, batch_size= batch_size, sampler = data_utils.sampler.RandomSampler(data))

def  build_label_domain(size,label):
    label_domain = torch.LongTensor(size)
    label_domain.data.resize_(size).fill_(label)
    return label_domain

def loop_iterable(iterable):
    while True:
        yield from iterable

class DANN(object):
    
    def __init__(self, feat_extractor,data_classifier, domain_classifier,source_data_loader, target_data_loader,
                 grad_scale = 1,cuda = False, logger_file = None, eval_data_loader = None, wgan = False, 
                 T_batches = None):
        self.feat_extractor = feat_extractor
        self.data_classifier = data_classifier
        self.source_data_loader = source_data_loader
        self.eval_data_loader =  target_data_loader  # potentially list of data_loader on which to evaluate the model
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

        self.T_batches = loop_iterable(target_data_loader)

        self.grad_scale_0 = grad_scale
        self.grad_scale = grad_scale
        _parent_class = self


        # adding gradient reversal layer transparently to the user
        class GradReverse(torch.autograd.Function):
            @staticmethod
            def forward(self,x):  
                return x.clone()
            @staticmethod
            def backward(self,grad_output):
                return grad_output.neg()*_parent_class.grad_scale
        
        class GRLDomainClassifier(nn.Module):
            def __init__(self,domain_classifier):
                super(GRLDomainClassifier, self).__init__()                              
                self.domain_classifier = domain_classifier
            def forward(self, input):
                x = GradReverse.apply(input)
                x = self.domain_classifier.forward(x)
                return x
        self.grl_domain_classifier = GRLDomainClassifier(domain_classifier)
        
        # these are the default
        self.optimizer_feat_extractor = optim.SGD(self.feat_extractor.parameters(),lr = 0.001)
        self.optimizer_data_classifier = optim.SGD(self.data_classifier.parameters(),lr = 0.001)
        self.optimizer_domain_classifier = optim.SGD(self.grl_domain_classifier.parameters(),lr = 0.1)

    def set_lr_decay_epoch(self,decay_epoch):
        self.lr_decay_epoch = decay_epoch
#        
    def set_epoch_to_start_align(self, epoch_to_start_align):
        self.epoch_to_start_align = epoch_to_start_align
    
    def set_grad_scale(self,new_grad_scale):
           self.grad_scale = new_grad_scale
    def set_filesave(self,filesave):
           self.filesave = filesave
    def show_grad_scale(self):
        print(self.grad_scale)
        return
        
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
    def set_lr_decay_epoch(self,decay_epoch):
        self.lr_decay_epoch = decay_epoch
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
        accur = correct.item() / len(data_loader.dataset)
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
            loss += self.criterion(output, target).item()
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()  
        return loss, correct
    
    def evaluate_domain_classifier(self):

        self.feat_extractor.eval()
        self.data_classifier.eval()
        self.grl_domain_classifier.eval()

        test_loss,correct = 0, 0
        test_loss, correct = self.evaluate_domain_classifier_class(self.source_data_loader, self.source_domain_label)
        loss, correct_a = self.evaluate_domain_classifier_class(self.eval_data_loader, self.test_domain_label)
        test_loss +=loss
        correct +=correct_a
        nb_source = len(self.source_data_loader.dataset) 
        nb_target = len(self. eval_data_loader.dataset) 
        nb_tot = nb_source + nb_target
        print('Domain: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, ( nb_source + nb_target ),
                100. * correct / (nb_source + nb_target )))
        if self.logger is not None:
             self.logger.info('Domain: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, ( nb_tot),
                100. * correct / nb_tot ))
        return 1.0*correct / nb_tot 
    
    def fit(self):
        # initialization

        if self.cuda:
            self.feat_extractor.cuda()
            self.data_classifier.cuda()
            self.grl_domain_classifier.cuda()
            
        
        self.perf_source = torch.zeros(self.nb_iter)
        self.perf_val = torch.zeros(self.nb_iter,len(self.eval_data_loader))
        self.perf_domain = torch.zeros(self.nb_iter)
        
        
        for epoch in range(self.nb_iter):
            self.feat_extractor.train()
            self.data_classifier.train()
            self.grl_domain_classifier.train()
            tic = tick()
            for batch_idx, (data, target) in enumerate(self.source_data_loader):
                size_source = data.size(0)
                data_test = next(self.T_batches)[0]
                size_test = size_source 
                # set gradient to 0
                self.feat_extractor.zero_grad()
                self.data_classifier.zero_grad()
                self.grl_domain_classifier.zero_grad()
                
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                    data_test = data_test.cuda()
                #
                # Source Domain Data  : forward feature extraction + data classifier 
                #
                output_feat_source = self.feat_extractor(data)
                output_class_source = self.data_classifier(output_feat_source)
                loss = F.cross_entropy(output_class_source,target)    
        
        
                if epoch >  self.epoch_to_start_align:
                    #-----------------------------------------------------------------
                    # domain classification
                    #-----------------------------------------------------------------
                    # for source data, compute_loss, gradient on domain classifier only
                    output_domain_source = self.grl_domain_classifier(output_feat_source)     
                    label_domain = build_label_domain(size_source,self.source_domain_label)   
                    if self.cuda:
                        label_domain = label_domain.cuda()
                    error_source_data = F.cross_entropy(output_domain_source,label_domain)     
                    
                    # for test data, compute_loss, gradient on domain classifier only
            
                    output_feat_test = self.feat_extractor(data_test)
                    output_domain_test = self.grl_domain_classifier(output_feat_test)
                    label_domain = build_label_domain(size_test, self.test_domain_label)
                    if self.cuda:
                        label_domain = label_domain.cuda()
                    error_test_data = F.cross_entropy(output_domain_test,label_domain)   
                    #define loss            
                    error  = loss + (error_source_data + error_test_data)
                else:
                    error = loss
                    
                error.backward()
                
                self.optimizer_feat_extractor.step()
                self.optimizer_data_classifier. step()
                self.optimizer_domain_classifier.step()      
        
            if self.lr_decay_epoch > 0:
                exp_lr_scheduler(self.optimizer_feat_extractor,epoch,self.lr_decay_epoch,self.lr_decay_factor)
                exp_lr_scheduler(self.optimizer_data_classifier,epoch,self.lr_decay_epoch,self.lr_decay_factor)
                exp_lr_scheduler(self.optimizer_domain_classifier,epoch,self.lr_decay_epoch,self.lr_decay_factor)


        
            toc =  tick() - tic 
            print('\nDANN Train Epoch: {}/{} {:2.2f}s [{}/{} ({:.0f}%)]\tLoss: {:.6f} Error:{:.6f}'.format(
                        epoch, self.nb_iter, toc , batch_idx * len(data), len(self.source_data_loader.dataset),
                        100. * batch_idx / len(self.source_data_loader), loss.item(),error.item()))
            self.evaluate_data_classifier(self.source_data_loader)
            self.evaluate_data_classifier(self.eval_data_loader)
            self.evaluate_domain_classifier()

        
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


