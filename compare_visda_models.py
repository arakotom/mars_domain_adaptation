#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
    

#------------------------------------------------------------------------------
#                   VISDA
#------------------------------------------------------------------------------
dtype = 'torch.FloatTensor'
class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()        
        self.fc1 = nn.Linear(100, 100,bias = True)     
        self.fc2 = nn.Linear(100, 1,bias = True)     

    def forward(self, input):        
        x = input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class DomainClassifierDANN(nn.Module):
    def __init__(self):
        super(DomainClassifierDANN, self).__init__()        
        self.fc1 = nn.Linear(100, 100,bias = True)     
        self.fc2 = nn.Linear(100, 2,bias = True)     

    def forward(self, input):        
        x = input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()        
        self.fc1 = nn.Linear(2048, 100,bias = True)     
        self.fc2 = nn.Linear(100, 100)     

    def forward(self, input):        
        x = input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class DataClassifier(nn.Module):
    def __init__(self):

        super(DataClassifier, self).__init__()
        #self.fc1 = nn.Linear(n_hidden, 2, bias = True)
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 3)
    def forward(self, input):

        x = input.view(input.size(0), -1)
        #x = self.fc1(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #Fully Connected Layer/Activation
        x = self.fc2(x)
        return x

class DataClassifierFull(nn.Module):
    def __init__(self):

        super(DataClassifierFull, self).__init__()
        #self.fc1 = nn.Linear(n_hidden, 2, bias = True)
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 12)
    def forward(self, input):

        x = input.view(input.size(0), -1)
        #x = self.fc1(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #Fully Connected Layer/Activation
        x = self.fc2(x)
        return x


if __name__== '__main__':

    feat = FeatureExtractor()
    print(feat(torch.randn(15,2048)).shape)
    
    #aux = feat(torch.randn(15,3,32,32))
    #output = classifier(aux)
    
