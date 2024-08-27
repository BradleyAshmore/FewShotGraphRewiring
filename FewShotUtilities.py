# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:04:57 2024

@author: ashmo

Utitlity functions for few shot experiments.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WebKB

from FewShotModels import GCN
import numpy as np
from numpy.random import default_rng

####Misc functions
#Make fake edges
def ismember(a, b, data):
    return np.intersect1d(torch.where(data.edge_index[0] == a)[0] , torch.where(data.edge_index[1] == b)[0]).shape[0] > 0

def ismemberoflist(a, b, arr):
    if a in arr and b in arr:
        return np.intersect1d(np.where(arr ==a ), np.where(arr == b) ).shape[0] > 0
    return False

def gcn_test(x, y, edge_index, test_mask, state_dic):
    model = GCN(x.shape[1], y.max()+1)
    model.load_state_dict(state_dic)
    model.eval()
    out = model(x, edge_index)
    preds = out[test_mask].argmax(dim=1)
    correct = (preds == y[test_mask]).sum()
    acc = correct / preds.shape[0]
    del(model)
    return acc

def gcn_eval(x, y, edge_index, test_mask, train_mask, verbose = False, reps = 5):
    best_loss = 9999
    best_acc = 0.0
    acc = 0.0
    y_max =  y.max() + 1
    best_model_state = None
    if type(y_max) == torch.Tensor:
        y_max = int(y_max.item())
    
    acc_results = []
    for r in range(reps):        
        model = GCN(x.shape[1], y_max)
        # model.init()
        op = optim.Adam(model.parameters(), lr=0.001)
        loss = nn.CrossEntropyLoss()
        # train_mask = train_mask.to(torch.long)
        # test_mask = test_mask.to(torch.long)
       
        for i in range(200):
            model.train()
            op.zero_grad()
            out = model(x, edge_index)
            train_loss = loss(out[train_mask], y[train_mask])
    
            train_loss.backward(retain_graph=True)
            op.step()
            preds = out[test_mask].argmax(dim=1)
            correct = (preds == y[test_mask]).sum()
            acc = correct / preds.shape[0]
            if verbose:
                print(f'GCN Epoch {i} Training loss {train_loss.item()}, Best Acc {best_acc}, acc {acc}')
                
            if best_acc < acc and i > 5:
                best_acc = acc
                best_model_state = model.state_dict()
                # acc = correct / preds.shape[0]
                # print(f'\tTest results: Acc {acc}')
                
        acc_results.append(best_acc)
        del (loss)
        del (op)
        del (model)
        del (train_loss)
    # return (sum(acc_results)/len(acc_results), best_model_state)
    arr = np.array(acc_results)
    mean = arr.mean()
    stddev = arr.std()
    return mean, stddev, best_model_state
    

'''
Generates a new set of labes

'''
def set_few_shot_labels(data, args):
    shot_size = int(args.shot_size)
    hold = torch.tensor([]) 
    num_classes = data.y.max().item() + 1
    # torch.manual_seed(12345)
    np.random.seed(seed=1234567)
    
    for x in range(num_classes):
        first_ten = torch.where(data.y[data.train_mask] == x)[0]
        rng = default_rng(seed=12345)
        numbers = rng.choice(first_ten.shape[0], size=first_ten.shape[0], replace=False)
        r = torch.tensor(numbers)
        first_ten = first_ten[r][:shot_size]
        # first_ten = first_ten[:3]
        hold = torch.cat( (hold, first_ten))
    
    hold = hold.to(torch.long)   
    few_shot_mask = (torch.zeros_like(data.train_mask) == 1)
    few_shot_mask[hold] = True
    print(hold)
    return hold, few_shot_mask



def load_cora(data_dir=".\data"):
    dataset = Planetoid(root=data_dir, name='Cora')
    data = dataset[0]
    return data


def load_texas(data_dir=".\data"):
    dataset = WebKB(root=data_dir, name='texas')
    data = dataset[0]
    data.train_mask = data.train_mask[:,0]
    data.val_mask = data.val_mask[:,0]
    data.test_mask = data.test_mask[:,0]
    return data

def load_wisco(data_dir=".\data"):
    dataset = WebKB(root=data_dir, name='Wisconsin')
    data = dataset[0]
    data.train_mask = data.train_mask[:,0]
    data.val_mask = data.val_mask[:,0]
    data.test_mask = data.test_mask[:,0]
    return data

def load_cornell(data_dir=".\data"):
    dataset = WebKB(root=data_dir, name='Cornell')
    data = dataset[0]
    data.train_mask = data.train_mask[:,0]
    data.val_mask = data.val_mask[:,0]
    data.test_mask = data.test_mask[:,0]
    return data


def load_citeseer(data_dir=".\data"):
    dataset = Planetoid(root=data_dir, name='Citeseer')
    data = dataset[0]
    return data
    # graph = to_networkx(data, to_undirected=True)
    # return data, graph