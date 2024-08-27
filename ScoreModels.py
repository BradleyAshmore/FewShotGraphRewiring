# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 16:12:10 2024

@author: ashmo

Models for intelligent score generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_cosine_sims(data):
    cosine_sims = []       #This is parallel to the edge_index.
    for i, x in enumerate( data.edge_index.t() ):
        src, dst = x    #Break up the x varialb into the end nodes.
        src = data.x[src]
        dst = data.x[dst]
        cosine_sims.append( F.cosine_similarity( src, dst, dim = 0 ).item() )
        
    return torch.tensor(cosine_sims) 
       

###Basic MLP implementation. 
class MLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MLP, self).__init__()
        self.layer_one = nn.Linear(in_size, out_size)
        self.relu = F.relu
        self.layer_two = nn.Linear(out_size, out_size)
        self.softmax = F.softmax
        
    def forward(self, x, sigmoid_output=False):
        x_hat = self.layer_one(x)
        x_hat = self.relu(x_hat)
        x_hat = self.layer_two(x_hat)
        
        if sigmoid_output:
            return self.softmax(x_hat)
        
        #No sigmoid for test.
        return x_hat