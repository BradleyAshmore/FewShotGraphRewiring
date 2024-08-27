# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 17:34:32 2024

@author: ashmo

This calculates the experimental score value.

"""


#Imports

import math
import argparse
import torch.nn.functional as F
import numpy as np
import time

import torch
import sys
import os

from pathlib import Path
# from FewShotModels import FNN, NodeOnlyVariationalAutoEncoder
# from FewShotUtilities import gcn_eval, load_cora, load_texas,load_citeseer,load_cornell,load_wisco, set_few_shot_labels, gcn_test
# from FewShotTrainingFunctions import train_autoencoder, train_edge_predictor, train_edge_predictor_for_homo

from GraphAnalysis import oversample_nodes_only
import matplotlib.pyplot as plt

from ScoreModels import *

sys.path.append("..")
from FewShotUtilities import gcn_eval, load_cora, load_texas,load_citeseer,load_cornell,load_wisco, set_few_shot_labels, gcn_test







###########Functions that will need to be moved.#################


def load_dataset_from_args(args):
    if args.dataset.lower() == 'cora':
        #Make these 3 lines a function. 
        data = load_cora(".\\data")
        data = data.to(device)
    elif args.dataset.lower() == 'texas':
        data = load_texas(".\\data")
        data = data.to(device)
        #Each iteration will have a different collection of labels.
    elif args.dataset.lower() == 'citeseer':
        data = load_citeseer(".\\data")
        data = data.to(device)
    elif args.dataset.lower() == 'wisconsin':
        data = load_wisco(".\\data")
        data = data.to(device)
    elif args.dataset.lower() == 'cornell':
        data = load_cornell(".\\data")
        data = data.to(device)
        # graph = to_networkx(data, to_undirected=True)
        # return data, graph
    data.few_shot_idx, data.few_shot_mask = set_few_shot_labels(data, args)
    print("Data loaded....")
    return data


#########################################End of functions.




#Accepts a model and data set
#Returns a trained model
#Need to consider the validation set to identify the best model, reload, and return.
def train_mlp(args, model, data_train, test_mask_pointer, data_test):
    file_name = ".\\savedmodedfor" + args.dataset.lower() + "ShotSize_" + str(args.shot_size) + ".mdl"
    # file_name = ".\\savedmodelfortexas.mdl"
    if os.path.isfile(file_name) and 1 == 0:
        model.load_state_dict(torch.load(file_name))
        model.eval()
        y_bar = model(data_train.x)
        test_acc  = (F.sigmoid(y_bar[data_test.val_mask]).argmax(dim=1) == data_test.y[data.val_mask]).sum() / sum(data.val_mask)
        torch.save(model.state_dict(), file_name) 
        return model, test_acc
    #Set optimizer, etc. 
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()
    best_val_loss = 999
    saved_model = None
    #Training loop
    for epoch in range(5000+1):
        model.train()
        optim.zero_grad()
        
        start = time.time()
        y_bar = model(data_train.x)
        
        train_loss = loss(y_bar[test_mask_pointer], data_train.y[test_mask_pointer])
        train_loss.backward(retain_graph=True)
        optim.step()
        
        end = time.time()
        
        train_acc = (F.sigmoid(y_bar[test_mask_pointer]).argmax(dim=1) == data_train.y[test_mask_pointer]).sum() / sum(test_mask_pointer)
        
        if False: #args.verbose:
        #Print epoch result.
            print(f'Epoch {epoch:4d}- loss: {train_loss:4f}, training acc: {train_acc.item():4f} in {end-start:4f} seconds')
        elif epoch % 25 == 0:
            
            print(f'Epoch {epoch:4d}- loss: {train_loss:4f}, training acc: {train_acc.item():4f} in {end-start:4f} seconds')
        #Train break every 100.
        
        #Check val score.
        model.eval()
        y_bar = model(data_test.x)
        val_loss = loss(y_bar[data_test.val_mask], data_test.y[data_test.val_mask] )
        
        if best_val_loss > val_loss.item():
            #Save model and loss
            best_val_loss = val_loss.item()
            saved_model = model.state_dict()
        if epoch > 0 and epoch % 100 == 0:
            model.eval()
            test_loss = loss(y_bar[data_test.val_mask], data_test.y[data_test.val_mask])
            
            test_acc  = (F.sigmoid(y_bar[data_test.val_mask]).argmax(dim=1) == data_test.y[data_test.val_mask]).sum() / sum(data_test.val_mask)
            print(f'\tValidation Test at epoch {epoch:4d} val loss: {test_loss:4f}, val acc: {test_acc:4f}')
    
    torch.save(saved_model, file_name)
    #Reload model
    model.load_state_dict(saved_model)
    model.eval()
    y_bar = model(data_test.x)
    test_acc  = (F.sigmoid(y_bar[data_test.val_mask]).argmax(dim=1) == data_test.y[data_test.val_mask]).sum() / sum(data_test.val_mask)
    # torch.save(model.state_dict(), file_name) 
    return model, test_acc

class SimilarityScoreCalculator():
    #Sets persistant variables and trains model. 
    def __init__(self, args, data, train_mask_pointer, ae):
        self.alpha = 1.0
        self.beta = .9
        self.gamma = .3    #Using as defaults.
        
        #Create MLP
        self.model = MLP(data.x.shape[1], data.y.max()+1)
      
        #Determine what labels to use.
        if args.shot_size < 1:  #Use default training set.
            self.train_mask_pointer = data.train_mask
        else:
            self.train_mask_pointer = data.few_shot_mask

        # Deterine what groups edges belong to. Use 0 for both labeled, 1 for a single label, 2 for neigther labeled.
            # Both labeled
        self.train_idx = (train_mask_pointer == True).nonzero(as_tuple=True)[0]

        self.model, bs = train_mlp(args, self.model, data, self.train_mask_pointer, data)

    #This generates the similarity scores for all edges in the dataset provided. 
    def set_scores_from_edges(self, data):

        #These are indexes in the edge_index.
        src_in_train = []
        dst_in_train = []
        for i, x in enumerate(data.edge_index[0] ):
            if x in self.train_idx:
                src_in_train.append(i)
                
        for i, x in enumerate(data.edge_index[1]):
            if x in self.train_idx:
                dst_in_train.append(i)
        
        dual_idx = []
        single_idx = []
        for x in src_in_train:
            for i, y in enumerate(dst_in_train):
                # print(f'X: {x}, Y: {y}')
                #Check for a match.
                if x == y:
                    dual_idx.append(x)  #X or Y. Doesn't matter.
                    #Reduce dst_in_train
                    dst_in_train = dst_in_train[i+1:]
                    break              #Break inner loop.
                elif y > x:   #No need to check further.
                    #Add X here.
                    single_idx.append(x)
                    break       #This will occur until x catches up.
                elif x > y:
                    if y not in single_idx:
                        single_idx.append(y)
         
        #Set no idx.
        list_base = [x for x in range(data.edge_index[0].shape[0])]
        zero_idx = set( list_base ).difference(set( dual_idx + single_idx ))
        
        #Get soft lables
        self.model.eval()
        soft_labels = self.model(data.x).argmax(dim=1)
        
        # #Determine probabilities.
        # alpha, beta, gamma = .9, .6, .25    #Using as defaults.
        
        #Calculate cosine similarity.
        #Accepts data object.
        cosine_sims = calculate_cosine_sims(data)
        
        #Calculate softlabel matrix.
        soft_label_scores = {} #list_base.copy()
        
        for i in range(data.edge_index.shape[1] ):
            src = data.edge_index[0][i].item()
            dst = data.edge_index[1][i].item()
            ee = (src, dst)
            
            if i in dual_idx:
                #Check true labels.
                if data.y[src] == data.y[dst]:
                    soft_label_scores[ee] = self.alpha  
                else:
                    soft_label_scores[ee] = self.gamma * cosine_sims[i] 
            elif i in single_idx:
                #Find the real label
                if src in self.train_mask_pointer:
                    s_label = data.y[src]
                    d_label = soft_labels[dst]
                else:
                    s_label = soft_labels[src]
                    d_label = data.y[dst]
                    
                if s_label == d_label:
                    soft_label_scores[ee] =  self.beta * cosine_sims[i] 
                else:
                    soft_label_scores[ee] = self.gamma * cosine_sims[i]
            else:
                soft_label_scores[ee] =   self.gamma * cosine_sims[i] 
         
        # soft_label_scores = torch.tensor(soft_label_scores)
        
        #Soft scores aligns with edge index.
        # print(soft_label_scores)
        # print("END")
        return soft_label_scores
        #Uses dual_idx, single_idx, zero_idx, soft_labels

    #Src and dst are node indexes.        
    def calculate_score_for_single_edges(self, src, dst, data):
        cs = F.cosine_similarity(data.x[src], data.x[dst], dim=0)
        to_return = None
        if self.train_mask_pointer[src] and self.train_mask_pointer[dst]:    #Both in train set
            #Bot in train set
            if data.y[src] == data.y[dst]:
                #Alpha
                to_return = self.alpha
            else:
                to_return = self.gamma * cs
        elif self.train_mask_pointer[src] or self.train_mask_pointer[dst]:   #One in train set
            if self.train_mask_pointer[src]: #Psudo the dst
                dst_label = self.model(data.x[dst]).argmax()
                if dst_label == data.y[src]:
                    to_return = self.beta * cs
                else:
                    to_return = self.gamma * cs
            else:
                src_label = self.model(data.x[src]).argmax()
                if src_label == data.y[dst]:
                    to_return = self.beta * cs
                else:
                    to_return = self.gamma * cs
        else:
            to_return = self.gamma * cs
        
        return to_return
    
def generate_similarity_score(args, data, train_mask_pointer, ae) :   
    #Create MLP
    model = MLP(data.x.shape[1], data.y.max()+1)
  
    #Determine what labels to use.
    if args.shot_size < 1:  #Use default training set.
        train_mask_pointer = data.train_mask
    else:
        train_mask_pointer = data.few_shot_mask

    # Deterine what groups edges belong to. Use 0 for both labeled, 1 for a single label, 2 for neigther labeled.
        # Both labeled
    train_idx = (train_mask_pointer == True).nonzero(as_tuple=True)[0]

    model, bs = train_mlp(args, model, data, train_mask_pointer, data)

    #These are indexes in the edge_index.
    src_in_train = []
    dst_in_train = []
    for i, x in enumerate(data.edge_index[0] ):
        if x in train_idx:
            src_in_train.append(i)
            
    for i, x in enumerate(data.edge_index[1]):
        if x in train_idx:
            dst_in_train.append(i)
    
    dual_idx = []
    single_idx = []
    for x in src_in_train:
        for i, y in enumerate(dst_in_train):
            # print(f'X: {x}, Y: {y}')
            #Check for a match.
            if x == y:
                dual_idx.append(x)  #X or Y. Doesn't matter.
                #Reduce dst_in_train
                dst_in_train = dst_in_train[i+1:]
                break              #Break inner loop.
            elif y > x:   #No need to check further.
                #Add X here.
                single_idx.append(x)
                break       #This will occur until x catches up.
            elif x > y:
                if y not in single_idx:
                    single_idx.append(y)
     
    #Set no idx.
    list_base = [x for x in range(data.edge_index[0].shape[0])]
    zero_idx = set( list_base ).difference(set( dual_idx + single_idx ))
    
    #Get soft lables
    model.eval()
    soft_labels = model(data.x).argmax(dim=1)
    
    #Determine probabilities.
    alpha, beta, gamma = .9, .6, .25    #Using as defaults.
    
    #Calculate cosine similarity.
    #Accepts data object.
    cosine_sims = calculate_cosine_sims(data)
    
    #Calculate softlabel matrix.
    soft_label_scores = {} #list_base.copy()
    
    for i in range(len(soft_label_scores) ):
        src = data.edge_index[0][i]
        dst = data.edge_index[1][i]
        ee = (src, dst)
        
        if i in dual_idx:
            #Check true labels.
            if data.y[src] == data.y[dst]:
                soft_label_scores[ee] = alpha
            else:
                soft_label_scores[ee] = gamma
        elif i in single_idx:
            #Find the real label
            if src in train_mask_pointer:
                s_label = data.y[src]
                d_label = soft_labels[dst]
            else:
                s_label = soft_labels[src]
                d_label = data.y[dst]
                
            if s_label == d_label:
                soft_label_scores[ee] = beta
            else:
                soft_label_scores[ee] = gamma
        else:
            soft_label_scores[ee] = gamma
     
    # soft_label_scores = torch.tensor(soft_label_scores)
    
    #Soft scores aligns with edge index.
    print(soft_label_scores)
    print("END")
    return soft_label_scores, model
    #Uses dual_idx, single_idx, zero_idx, soft_labels
    
    