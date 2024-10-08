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
import random

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


def block_data(data):
    usable_data = data.x[data.few_shot_idx]
    pairs = zip(usable_data, data.y[data.few_shot_idx])
    to_add_x = torch.tensor(usable_data[0]).unsqueeze(dim=0)
    to_add_y = torch.tensor(data.y[data.few_shot_idx][0].clone().detach()).unsqueeze(dim=0)
    for x, y in pairs:
        #Another loop for the number of bits.
        its = int(sum(x).item())
        decoder = torch.where(x == 1)[0]
        for i in range(its):
            hold = x.clone()  
            hold[decoder[i]] = 0 #Remove a bit.
            to_add_x = torch.cat( (to_add_x, hold.unsqueeze(dim=0)), dim=0 )
            to_add_y = torch.cat( (to_add_y, y.unsqueeze(dim=0)) )
    
    idxes = [i for i in range(len(to_add_x))]
    random.shuffle(idxes)
    sz = min(len(to_add_x), int(data.x.shape[0]/2) )
    idxes = torch.tensor(idxes[:sz])   #Only take a subset.
    
    to_return = data.clone()
    to_return.x = torch.cat( (to_return.x.clone().detach(), torch.tensor(to_add_x)[idxes] ) )
    to_return.y = torch.cat( (to_return.y.clone().detach(), torch.tensor(to_add_y)[idxes] ) )
    
    extra_mask = torch.tensor([True for i in range(sz) ])
    to_return.few_shot_mask = torch.cat( (to_return.few_shot_mask.clone().detach(), torch.tensor(extra_mask)) )
    
    extra_mask = extra_mask == False
    to_return.val_mask = torch.cat( (to_return.val_mask.clone().detach(), torch.tensor(extra_mask)) )
    to_return.test_mask = torch.cat( (to_return.test_mask.clone().detach(), torch.tensor(extra_mask)) )
    
    return to_return


#Accepts a model and data set
#Returns a trained model
#Need to consider the validation set to identify the best model, reload, and return.
def train_mlp(args, model, data_in, test_mask_pointer, data_test, epochs):
    data_train = data_in.clone()
    # data_train = block_data(data_train)
    test_mask_pointer = data_train.few_shot_mask
    
    #For debuggin purposes.
    test_acc_progress = []
    
    # file_name = ".\\savedmodedfor" + args.dataset.lower() + "ShotSize_" + str(args.shot_size) + ".mdl"
    # # file_name = ".\\savedmodelfortexas.mdl"
    # if os.path.isfile(file_name) and 1 == 0:
    #     model.load_state_dict(torch.load(file_name))
    #     model.eval()
    #     y_bar = model(data_train.x)
    #     # test_acc  = (F.sigmoid(y_bar[data_test.val_mask]).argmax(dim=1) == data_test.y[data.val_mask]).sum() / sum(data.val_mask)
    #     test_acc  = (F.sigmoid(y_bar[data_test.test_mask]).argmax(dim=1) == data_test.y[data.test_mask]).sum() / sum(data.test_mask)
    #     torch.save(model.state_dict(), file_name) 
    #     return model, test_acc
    #Set optimizer, etc. 
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()
    best_val_loss = 999
    saved_model = None
    
    #Training loop
    for epoch in range(epochs):
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
        # val_loss = loss(y_bar[data_test.val_mask], data_test.y[data_test.val_mask] )
        val_loss = loss(y_bar[data_test.test_mask], data_test.y[data_test.test_mask] )
        #Predictions
        pred = F.sigmoid(y_bar[data_test.test_mask]).argmax(dim=1)
        ta = ( (pred == data_test.y[data_test.test_mask]).sum() / pred.shape[0]).item()
        test_acc_progress.append(ta)
        if best_val_loss > val_loss.item():
            #Save model and loss
            best_val_loss = val_loss.item()
            saved_model = model.state_dict()
        if epoch > 0 and epoch % 100 == 0:
            model.eval()
            # test_loss = loss(y_bar[data_test.val_mask], data_test.y[data_test.val_mask])
            # test_acc  = (F.sigmoid(y_bar[data_test.val_mask]).argmax(dim=1) == data_test.y[data_test.val_mask]).sum() / sum(data_test.val_mask)
            test_loss = loss(y_bar[data_test.test_mask], data_test.y[data_test.test_mask])

            test_acc  = (F.sigmoid(y_bar[data_test.test_mask]).argmax(dim=1) == data_test.y[data_test.test_mask]).sum() / sum(data_test.test_mask)
            print(f'\tValidation Test at epoch {epoch:4d} val loss: {test_loss:4f}, val acc: {test_acc:4f}')
    
    # torch.save(saved_model, file_name)
    #Reload model
    # model.load_state_dict(saved_model)
    model.eval()
    y_bar = model(data_test.x)
    test_acc  = (F.sigmoid(y_bar[data_test.test_mask]).argmax(dim=1) == data_test.y[data_test.test_mask]).sum() / sum(data_test.test_mask)
    # torch.save(model.state_dict(), file_name)
    # plt.plot(test_acc_progress)
    # plt.title('Test accuracy for predictor MLP')
    # plt.show()
    return model, test_acc

class SimilarityScoreCalculator():
    #Sets persistant variables and trains model. 
    def __init__(self, args, data, train_mask_pointer, epochs=2000, alpha = 1.0, beta = .9, delta = .8, gamma = .3):
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma   #Using as defaults.
        self.dual_counter = 0
        self.single_match = 0
        self.single_mis = 0
        self.no_label = 0
        self.no_label_matches = 0
        #Create MLP
        self.model = MLP(data.x.shape[1], data.y.max()+1)
        self.model.to(args.device)
        #Determine what labels to use.
        if args.shot_size < 1:  #Use default training set.
            self.train_mask_pointer = data.train_mask
        else:
            self.train_mask_pointer = data.few_shot_mask

        # Deterine what groups edges belong to. Use 0 for both labeled, 1 for a single label, 2 for neigther labeled.
            # Both labeled
        self.train_idx = (train_mask_pointer == True).nonzero(as_tuple=True)[0]

        self.model, bs = train_mlp(args, self.model, data, self.train_mask_pointer, data, epochs)

    def print_match_stats(self, ratios = False):
        if ratios:
            total = self.dual_counter + self.single_match + self.single_mis + self.no_label
            # print(f'Dual labels: {self.dual_counter/total:4f}, single_match: {self.single_match/total:4f}, single mismatch {self.single_mis/total:4f}, gammas: {self.no_label/total:4f}, soft matches: {self.no_label_matches/total:4f}')    
        print(f'Dual labels: {self.dual_counter}, single_match: {self.single_match}, single mismatch {self.single_mis}, gammas: {self.no_label}, soft matches: {self.no_label_matches}')
    
    #This function performs the score calculation. 
    #src, dst are node lables
    #data is the dataset
    #soft_labels is the output of the pre-trained model.
    #i is an index. What the index is is managed by the calling function. 
    def internal_score_calculation(self, src, dst, data, dual_idx, single_idx, cosine_sims, i, on_edge=True):
        #Get soft lables
        self.model.eval()
        soft_labels = self.model(data.x).argmax(dim=1)

        if i in dual_idx:
            #Check true labels.
            if data.y[src] == data.y[dst]:
                
                self.dual_counter += 1
                return self.alpha 
            else:
                return self.gamma * cosine_sims[i] 
            
        elif i in single_idx:
            #Find the real label
            if src in self.train_mask_pointer:
                s_label = data.y[src]
                d_label = soft_labels[dst]
            else:
                s_label = soft_labels[src]
                d_label = data.y[dst]
                
            if s_label == d_label:
                self.single_match += 1
                return self.beta * cosine_sims[i] 
                
            else:
                self.single_mis += 1
                return self.gamma * cosine_sims[i]

        else:
            
            self.no_label += 1
            s_label = soft_labels[src]
            d_label = soft_labels[dst]
            if s_label == d_label:
                self.no_label_matches += 1
                return self.delta * cosine_sims[i] 
                
            return self.gamma * cosine_sims[i]     
    
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
        

        # #Determine probabilities.
        # alpha, beta, gamma = .9, .6, .25    #Using as defaults.
        
        #Calculate cosine similarity.
        #Accepts data object.
        
        
        #Calculate softlabel matrix.
        soft_label_scores = {} #list_base.copy()
        cosine_sims = calculate_cosine_sims(data)
        
        for i in range(data.edge_index.shape[1] ):
            src = data.edge_index[0][i].item()
            dst = data.edge_index[1][i].item()
            soft_label_scores[ (src, dst) ] = self.internal_score_calculation(src, dst, data, dual_idx, single_idx ,cosine_sims, i)
           
                
        # soft_label_scores = torch.tensor(soft_label_scores)
        #Soft scores aligns with edge index.
        # print(soft_label_scores)
        # print("END")
        return soft_label_scores
        #Uses dual_idx, single_idx, zero_idx, soft_labels

    #Exhaustivly sets the scores.
    def set_scores_about_a_node(self, node, data):
        #These are pairs in the training index
        src_in_train = []
        dst_in_train = []
        if data.train_mask[node]:
            src_in_train = True
        else:
            src_in_train = False
         
        if src_in_train:
            dual_idx = torch.where(data.train_mask)[0].tolist()
        else:
            dual_idx = []
            
        single_idx = []
        if src_in_train:
            single_idx = torch.where(data.train_mask == False)[0].tolist()
        else:
            single_idx = torch.where(data.train_mask)[0].tolist()
            
        soft_label_scores = {}
        cosine_sims = None #This needs to be calculated.
        
        src_tensor = torch.stack( (data.x[node], data.x[node] ) )
        for qq in range(src_tensor.shape[0], data.x.shape[0]):
            src_tensor = torch.cat( (src_tensor, data.x[node].unsqueeze(dim=0)), dim=0)
        
        # src_tensor = torch.tensor( [data.x[node] for qq in range(data.x.shape[0]) ] )
        cosine_sims = F.cosine_similarity(src_tensor, data.x, dim=1).tolist()
        for dst in range(data.x.shape[0]):
            if not node == dst:
                soft_label_scores[ (node, dst) ] = self.internal_score_calculation(node, dst, data, dual_idx, single_idx, cosine_sims, dst)
                
        return soft_label_scores
    
    def calculate_score_for_all_pairs(self, data, node_scores):
        #Get all cosine sim.
        row_size = data.x.shape[0]
        vector_size = data.x.shape[1]
        x_row, x_col = data.x[None,:,:], data.x[None, :, :]
        x_row, x_col = x_row.expand(row_size, row_size, vector_size), x_col.expand(row_size, row_size, vector_size)
        # self.cs = F.cosine_similarity(x_row, x_col, dim=-1)
        
        #Set truth matrix
        self.truth_mat = torch.eq(data.y[None,:], data.y[:,None])
        
        #Get label predictions.
        psuedo_labels = self.model(data.x).argmax(dim=1)
        self.pred_mat = torch.eq(psuedo_labels[None, :], psuedo_labels[:, None])
        
        #in the for real X psuedo--  COL              ROW
        self.hybrid_mat = torch.eq(data.y[None, :], psuedo_labels[:, None])
        
        #Monster for loop
        for x in range(data.x.shape[0]):
            if x % 100 == 0:
                print(f'x: {x}')
            self.cs = F.cosine_similarity(data.x[x], data.x)
            for y in range(data.y.shape[0]):
                if x == y:
                    node_scores.add_score((x, y), 0.0)
                elif x > y:
                    node_scores.add_score((x, y), node_scores.get_score( (y, x) ) )
                
                else:
                    node_scores.add_score((x,y), self.score_logic(x, y, data))
        return node_scores
   
    def score_logic(self, src, dst, data):
        # return cs
        cs = self.cs[dst]
        to_return = None
        if data.train_mask[src] and data.train_mask[dst]:    #Both in train set
            if self.truth_mat[src][dst]    :
                return self.alpha
            else:
                return self.gamma * cs
           
        elif data.train_mask[src] or data.train_mask[dst]:   #One in train set
            if data.train_mask[src]:    #Needed to ensure order is correct
                comp = self.hybrid_mat[dst][src]
            else:
                comp = self.hybrid_mat[src][dst]
            if comp: #Psudo the dst
             
                return  self.beta * cs
            else:
                return self.gamma * cs
          
        else:   #None in train set
            if self.pred_mat[src][dst]:
                return self.delta * cs
            else:
                return self.gamma * cs
        
        
    
        
        
        
    #Src and dst are node indexes.        
    def calculate_score_for_single_edges(self, src, dst, data):
        # raise Excpetion("Debug this.")
        cs = F.cosine_similarity(data.x[src], data.x[dst], dim=0)
        # return cs
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
            s_label, d_label = self.model(data.x[src]).argmax(), self.model(data.x[dst]).argmax()
            if s_label.item() == d_label.item():
                to_return = .75 * cs
            else:
                to_return = self.gamma * cs
        
        return to_return
    
def generate_similarity_score(args, data, train_mask_pointer) :   
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
    
    