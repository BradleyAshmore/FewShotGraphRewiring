# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:04:57 2024

@author: ashmo

Utitlity functions for few shot experiments.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch_geometric.datasets import Actor
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WebKB
import time
from FewShotModels import GCN
import numpy as np
from numpy.random import default_rng
import torch.functional as F
import matplotlib.pyplot as plt
import random
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import DataProcessor as DP
import NetworkDataset


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

def gcn_eval(data, args = None, seed_list=None, verbose = False, reps = 10, new_labels=False):
  
    y_max =  data.y.max() + 1
    best_model_state = None
    if type(y_max) == torch.Tensor:
        y_max = int(y_max.item())
    
    acc_results = []
    f_train_history = []
    # acc_res_matrix = []
    for r in range(reps): 
        training_results = []
        best_loss = 9999
        best_acc = 0.0
        acc = 0.0
        model = GCN(data.x.shape[1], y_max)
        # model.init()
        op = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        loss = nn.NLLLoss() #nn.CrossEntropyLoss()
        # train_mask = train_mask.to(torch.long)
        # test_mask = test_mask.to(torch.long)
        if new_labels:
            if seed_list == None:
                seed = None
            else:
                seed = seed_list[r]
            data.few_shot_labels, data.few_shot_mask = set_few_shot_labels(data, args, seed=seed)           
        for i in range(200):
            model.train()
            op.zero_grad()
            out = model(data.x, data.edge_index)
            train_loss = loss(out[data.few_shot_mask], data.y[data.few_shot_mask])
    
            train_loss.backward(retain_graph=True)
            op.step()
            # preds = nn.functional.softmax(out[data.test_mask]).argmax(dim=1)
            # correct = (preds == data.y[data.test_mask]).sum()
            # acc = correct / preds.shape[0]
            pred = out.argmax(dim=1).clone()
            pred_train = pred[data.few_shot_mask].detach().numpy()
            lbl_train = data.y[data.few_shot_mask].clone().detach().numpy()
            
            f_train = f1_score(lbl_train, pred_train)
            f_train_history.append(f_train)
            pred = out[data.test_mask].argmax(dim=1).clone().detach().numpy()
            lbl = data.y[data.test_mask].clone().detach().numpy()
            acc = accuracy_score(lbl, pred)
            f = f1_score(lbl, pred)
            p = precision_score(lbl, pred)
            rec = recall_score(lbl, pred)
            
            training_results.append(( acc, p, rec, f) )
            if verbose:
                print(f'GCN Epoch {i} Training loss {train_loss.item()}, Best Acc {best_acc}, acc {acc}')
                
            # if best_acc < acc:
            #     best_acc = acc
            #     best_model_state = model.state_dict()
            
            
            
            #modifying for f-1
            if best_acc < f:
                best_acc = f
                # acc = correct / preds.shape[0]
                # print(f'\tTest results: Acc {acc}')
        # plt.plot(training_results)
        # plt.title(f'GCN Eval repetition {r}')
        # plt.show()
        acc_results.append(best_acc)
        del (loss)
        del (op)
        del (model)
        del (train_loss)
    # return (sum(acc_results)/len(acc_results), best_model_state)
    # acc_res_matrix.append(acc_results)
    print(f'Results from all reps: \n\t{acc_results}\n')
    print(f'F1 history for training: \n{f_train_history}')
    arr = np.array(acc_results)
    mean = arr.mean()
    stddev = arr.std()
    return mean, stddev, acc_results, best_model_state
    # return acc_results    

'''
Generates a new set of labes

'''
def set_few_shot_labels(data, args, seed = None):
    shot_size = int(args.shot_size)
    hold = torch.tensor([]) 
    num_classes = data.y.max().item() + 1
    # torch.manual_seed(round(time.time()))
    num_shot = args.shot_size
    train_num = []
    for x in range(num_classes):
        index = (data.y[data.train_mask] == x).nonzero(as_tuple=False).view(-1).tolist() 
        random.shuffle(index)
        select_index = index[:num_shot]
        train_num.extend(select_index)


    new_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    new_train_mask[train_num] = True
    few_shot_mask = new_train_mask
    few_shot_idx = torch.where(few_shot_mask)[0]
    print(f"Number of training nodes: {data.train_mask.sum().item()}")
    return few_shot_idx, few_shot_mask
    #     # print(f'Looking for {x}')
    #     first_ten = torch.where(data.y[data.train_mask] == x)[0]    #Micro.
    #     first_ten = torch.where(data.train_mask)[0][first_ten]
    #     # print(first_ten)
    #     if not seed == None:
    #         rng = default_rng(seed=seed)
    #     else:
    #         rng = default_rng()
    #     numbers = rng.choice(first_ten.shape[0], size=first_ten.shape[0], replace=False)
    #     r = torch.tensor(numbers)
    #     first_ten = first_ten[r][:shot_size]
    #     # first_ten = first_ten[:3]
    #     hold = torch.cat( (hold, first_ten))
    
    # hold = hold.to(torch.long)   
    # few_shot_mask = (torch.zeros_like(data.train_mask) == 1)
    # few_shot_mask[hold] = True
    # print(hold)
    # return hold, few_shot_mask



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
    
def load_actor(data_dir=".\data"):
    dataset = Actor(root=data_dir)
    data = dataset[0]
    data.train_mask = data.train_mask[:,0]
    data.val_mask = data.val_mask[:,0]
    data.test_mask = data.test_mask[:,0]
    return data

def load_dataset_from_args(args):
    device = args.device
    data_loc = "E:/DataSets"
    if args.dataset.lower() == 'cora':
        # Make these 3 lines a function.
        data = load_cora(data_loc)
        data = data.to(device)
    elif args.dataset.lower() == 'texas':
        data = load_texas(data_loc)
        data = data.to(device)
        # Each iteration will have a different collection of labels.
    elif args.dataset.lower() == 'citeseer':
        data = load_citeseer(data_loc)
        data = data.to(device)
    elif args.dataset.lower() == 'wisconsin':
        data = load_wisco(data_loc)
        data = data.to(device)
    elif args.dataset.lower() == 'cornell':
        data = load_cornell(data_loc)
        data = data.to(device)
        # graph = to_networkx(data, to_undirected=True)
        # return data, graph
    elif args.dataset.lower() == 'actor':
        data = load_actor(data_loc)
        data = data.to(device)
    elif args.dataset.lower() == 'toniot':
        dataset = NetworkDataset.NetworkDataset()
        data = dataset[args.graph_num+1]
        data = data.to(device)
    #Remove self loops
    self_loops = torch.where(data.edge_index[0,:] == data.edge_index[1,:])[0]
    
    if self_loops.shape[0] > 0: #They exist
        new_edge_index = data.edge_index[:, :self_loops[0]]
        
        for c in range(1, self_loops.shape[0]):
            new_edge_index = torch.cat( (new_edge_index, data.edge_index[:, self_loops[c-1]+1:self_loops[c]]) , dim=1)
        data.edge_index = new_edge_index
        
        #Add the tail
        new_edge_index = torch.cat( (new_edge_index, data.edge_index[:, self_loops[c-1]+1:]) , dim=1)
    data.few_shot_idx, data.few_shot_mask = set_few_shot_labels(data, args)
    # data.train_idx= data.few_shot_idx
    # data.few_shot_mask = data.train_mask
    print("Data loaded....")
    return data
