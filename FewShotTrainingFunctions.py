# -*- coding: utf-8 -*-
"""
Created on Fri May 31 01:10:53 2024

@author: ashmo

Training functions for autoencoder and edge predictor.
"""

import time
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from FewShotModels import NodeOnlyVariationalAutoEncoder, FNN, NodeAndStructureVariationalAutoEncoder


#Loss functions

def kl_loss(predictions, targets, mean, log_var, KDL_factor=-0.1):
    x1 = predictions  # predictions.squeeze().cpu().detach()
    x2 = targets  # targets.squeeze().cpu().detach()

    repoduction_loss = F.mse_loss(x1, x2, reduction='mean')
    KDL = KDL_factor * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    KL_loss = repoduction_loss + KDL

    return KL_loss
# Training
#This saves the autoencoder to the disk to be read later.
def train_autoencoder(data, device, save_file_name, args = None):
    in_dim = data.x.shape[1]
    
    if args == None:
        ae = NodeOnlyVariationalAutoEncoder(in_dim, 256, 14)
    elif args.oversample_method == 'ae_structure':
        ae = NodeAndStructureVariationalAutoEncoder(in_dim, data.x.shape[0], 32, 2)
        
    optimizer = optim.Adam(ae.parameters(),
                           lr=0.001)
    # feature_loss_func = nn.MSELoss()
    feature_loss_func = kl_loss #nn.KLDivLoss()

    # save_file_name = ".\\ExperimentalVAE.mdl"
    best_loss = 100000
    best_model = None
    ae_loss = []
    perfect_loss = []
    best_state = None
    
    ae_mask = data.val_mask + data.train_mask
    for i in range(30000):  # tqdm(range(5)):
        
        # Setups
        optimizer.zero_grad()
        ae.train()
        t = time.time()

        X_hat, mu, sig = ae(data.x, data.edge_index)
        
        feature_loss = feature_loss_func(X_hat[ae_mask], data.x[ae_mask], mu, sig, KDL_factor=-0.001)
        loss_train =  feature_loss #.75 * feature_loss + .25 * degree_loss
        loss_train.backward()
        optimizer.step()
        print(" Epoch: ", i, " Training Loss: ", loss_train.item(), "Feature Loss: ", feature_loss, " Time: ", time.time() - t)

        if loss_train.item() <= best_loss:
            # Save model
            best_loss = loss_train.item()
            print("Saving...")
            best_state = ae.state_dict()
            # torch.save(ae.state_dict(), save_file_name)

        # loss_values.append(loss_train.item())
        if i % 100 == 0 and i > 1:    #Run on test case
            ae.eval()
            # X_hat, mu, sig = ae(data.x[data.test_mask], data.edge_index)
            X_hat, mu, sig = ae(data.x, data.edge_index)
            # X_hat = F.normalize(X_hat)
            # , mu[idx_train], sig[idx_train])
            test_loss = feature_loss_func(X_hat[data.test_mask], data.x[data.test_mask], mu, sig)
            ae_loss.append(test_loss.item())
            
            # X_hat, mu, sig = ae(data.x[data.test_mask], data.edge_index, reparamiterize=True)
            # # X_hat = F.normalize(X_hat)
            # # , mu[idx_train], sig[idx_train])
            # test_loss2 = feature_loss_func(X_hat, data.x[data.test_mask], mu, sig)
            # perfect_loss.append(test_loss2.item())
    # Load best model
    # ae.load_state_dict(torch.load(save_file_name))

    # Evaluate
    # plt.plot(ae_loss)
    # plt.title('Test Loss for Variational autoencoder')
    # plt.show()
    torch.save(best_state, save_file_name)
    plt.plot(perfect_loss)
    plt.title('Test Loss for Basic Autoencoder')
    plt.savefig('.\\AutoencoderLoss')

#####Edge predictor####

def prepare_edges(args, data):
    ####Edge predictions
    def ismember(a, b):
        return np.intersect1d(torch.where(data.edge_index[0] == a)[0] , torch.where(data.edge_index[1] == b)[0]).shape[0] > 0

    def ismemberoflist(a, b, arr):
        if a in arr and b in arr:
            return np.intersect1d(np.where(arr ==a ), np.where(arr == b) ).shape[0] > 0
        return False

    #Pair real edges 
    real_edges = torch.cat( (data.x[data.edge_index[0]], data.x[data.edge_index[1]]), 1 )
    edge_save_file = ".\\FullEdgesFor" + args.dataset.lower() + "Predictor.csv"  #Check if this is present before training.
    #Make fake edges
 

    edges_needed = data.edge_index.shape[1] #real_edges.shape[0]
    edges_created = 0
    test_edges_false = []
    first_pass_counter = 0
    while edges_created < edges_needed:
        if first_pass_counter < 10:
            print("Looping")
        #     idx_ia = np.random.randint(0, data.edge_index[0].max(), 2708)
        #     idx_ja = np.random.randint(0, data.edge_index[0].max(), 2708)
            idx_ia = torch.randperm(data.edge_index[0].max() + 1)[: data.edge_index[0].max()]
            idx_ja = torch.randperm(data.edge_index[0].max() + 1)[: data.edge_index[0].max()]
            for idx_i, idx_j in zip(idx_ia, idx_ja):
                idx_i, idx_j = idx_i.item(), idx_j.item()
                if edges_created == edges_needed:
                    break;
                if idx_i == idx_j:
                    continue
                if ismember(idx_i, idx_j):
                    continue
                if test_edges_false:
                      if ismemberoflist(idx_j, idx_i, np.array(test_edges_false)):
                          continue
                      if ismemberoflist(idx_i, idx_j, np.array(test_edges_false)):
                          continue
                test_edges_false.append([idx_i, idx_j])
                edges_created += 1
                print(f'{edges_created} of {edges_needed}')
                
        else:
         
            print("Brute forcing....")
            idx_ia = torch.randperm(data.edge_index[0].max() + 1)[: 500]
            idx_ij = torch.randperm(data.edge_index[0].max() + 1)[: data.edge_index[0].max()]
            for idx_i in idx_ia:
                idx_i.item()
                
                for idx_j in idx_ja:
                    idx_j.item()
                  
                if edges_created == edges_needed:
                    break;
                if idx_i == idx_j:
                    continue
                if ismember(idx_i, idx_j):
                    continue
                if test_edges_false:
                      if ismemberoflist(idx_j, idx_i, np.array(test_edges_false)):
                          continue
                      if ismemberoflist(idx_i, idx_j, np.array(test_edges_false)):
                          continue
                test_edges_false.append([idx_i, idx_j])
                edges_created += 1
                print(f'{edges_created} of {edges_needed}')
        first_pass_counter += 1   
                    
    #Create self loops
    #Add self loops
    ids = [x for x in range(data.x.shape[0])]
    selfs = []
    for x in ids:
            selfs.append([x, x])
            
    test_edges_false_idx = test_edges_false[:edges_needed]
    reals =  torch.cat( (data.edge_index[:, :edges_needed], torch.tensor(selfs).t()), dim = 1) 
    all_edges = torch.cat( (reals,  torch.tensor(test_edges_false).t() ), dim = 1 )
    
    torch.save(all_edges, edge_save_file)

    all_edges = torch.load(edge_save_file)
 

    edge_labels = torch.cat( (torch.ones_like(all_edges[0][:(edges_needed + data.x.shape[0])]), torch.zeros_like(data.edge_index[0][:edges_needed])) )  

 
    num_edges = edge_labels.shape[0]
    
    #Assume an 80/20 train test split
    train_edges = math.floor(num_edges * .8)
    test_edges = num_edges - train_edges
    
    idx = np.arange(num_edges)
    np.random.shuffle(idx)
    edge_train_idx = torch.tensor(idx[:train_edges] )
    edge_test_idx = torch.tensor( idx[train_edges:] )
    print("Edges Done")

    return all_edges, edge_labels, train_edges, test_edges, edge_train_idx, edge_test_idx
    
def train_edge_predictor(args, data, save_file_name):
    all_edges, edge_labels, train_edges, test_edges, edge_train_idx, edge_test_idx = prepare_edges(args, data)
    in_dim = data.x.shape[1]
    edge_predictor = FNN(in_dim * 2, math.floor(in_dim/ 6), 2, 2) 

    print(edge_predictor)
    
    edge_optimizer = optim.Adam(edge_predictor.parameters(), lr=0.0001)
    class_weight = [1 if x == 1 else 1 for x in edge_labels]
    class_weight = torch.FloatTensor(class_weight)

    loss_function = torch.nn.BCELoss(weight=class_weight[edge_train_idx])
    test_loss = torch.nn.BCELoss(weight=class_weight[edge_test_idx])

    frm = data.x[all_edges[0][edge_train_idx]]
    to = data.x[all_edges[1][edge_train_idx]]
    train_set = frm + to # torch.cat( (frm, to), dim = 1)
    
    #Train edge  
    best_loss = 999999999999
    lbls = edge_labels[edge_train_idx].to(torch.long)
    # lbls.requires_grad = True
     #Load best
    frm = data.x[all_edges[0][edge_test_idx]]
    to = data.x[all_edges[1][edge_test_idx]]
    test_set =  frm + to #torch.cat( (frm, to), dim = 1)
    test_lbls = edge_labels[edge_test_idx].to(torch.long)
    num_real = sum(edge_labels[edge_train_idx] == 1 )
    num_fake = sum(edge_labels[edge_train_idx] == 0 )
    test_reals = sum(edge_labels[edge_test_idx] == 1)
    test_fakes = sum(edge_labels[edge_test_idx] == 0)
    test_dic = {'correct':[],
                'reals':[],
                'fakes':[]}
    loss_history = []

    EP_name = save_file_name
    best_acc = 0.0
    decline_counter = 0

    for i in range(5000):  # tqdm(range(5)):
        edge_predictor.train()
        # Setups
        edge_optimizer.zero_grad()
        t = time.time()
       
        #Concat nodes
        
        preds = F.softmax(edge_predictor(train_set)).to(torch.float)
        # preds.requires_grad = True
        # preds = preds.argmax(dim = 1).to(torch.float)
        # preds.require_grad = True
        loss = loss_function(preds[:, 1], lbls.float())
        preds = preds.argmax(dim = 1).to(torch.float)
        num_correct = sum(preds == edge_labels[edge_train_idx])
        real_correct = sum((preds == 1) & (edge_labels[edge_train_idx] ==  1)).item()
        fakes = sum( (preds == 0) & (edge_labels[edge_train_idx] == 0))
    #   neighbor_loss = neighbor_loss_func(neighbors[idx_train], ae.last_samples[idx_train], ae.sampled_mean[idx_train], ae.sampled_sigma[idx_train])
        # loss_train = .4 * feature_loss + .2 * degree_loss   + .4 * n_loss
        # loss_train = feature_loss
        # loss_train = loss_func(X_bar[idx_train], data.x[idx_train])
        loss.backward()
        edge_optimizer.step()
        # print("Dataset Name:  Edges, Epoch: ", i, " Training Loss: ", loss.item(),  "Correct: ", num_correct, "Acc: ", (num_correct/ edge_labels[edge_train_idx].shape[0]).item(), " \n\tFakes IDed: ", fakes.item(),
        #       " Fake Acc: ", fakes/num_fake, " Real Acc: ", real_correct/ num_real, " Time: ", time.time() - t)
    
        if i % 100 == 0:
            loss_history.append(loss.item())
            #Train edge predictor
            
           
            t = time.time()
               
            #Concat nodes
            edge_predictor.eval()
            preds = F.softmax(edge_predictor(test_set)).to(torch.float)
            # preds.requires_grad = True
            # preds = preds.argmax(dim = 1).to(torch.float)
            # preds.require_grad = True
            loss = test_loss(preds[:, 1], test_lbls.float())
            preds = preds.argmax(dim = 1).to(torch.float)
            num_correct = sum(preds == edge_labels[edge_test_idx]).item()
            correct_reals = sum((preds == 1) & (edge_labels[edge_test_idx] == 1) ).item()
            correct_fakes = sum((preds == 0) & (edge_labels[edge_test_idx] == 0) ).item()
            test_dic['correct'].append(num_correct / edge_test_idx.shape[0] ) 
            test_dic['reals'].append( (correct_reals / sum(edge_labels[edge_test_idx] == 1)).item())
            test_dic['fakes'].append( (correct_fakes/ sum(edge_labels[edge_test_idx] == 0) ).item())
            acc = (num_correct/ edge_labels[edge_test_idx].shape[0])
            print("\nTest Result:  Edges, Epoch: ", i, " Training Loss: ", loss.item(),  "Correct: ", num_correct, "Acc: ", (num_correct/ edge_labels[edge_test_idx].shape[0]), 
                  "\n\tReal Correct: ", correct_reals, "Real acc: ", correct_reals / test_reals, " Correct fakes: ", correct_fakes, " Fake Acc: ", correct_fakes/ test_fakes, " Time: ", time.time() - t)
            
            if best_acc < acc:
                best_acc = acc
                print("Saving...")
                torch.save(edge_predictor.state_dict(), EP_name)
                decline_counter = 0
            else:
                decline_counter += 1
                if decline_counter == 15:
                    pass
            # if   correct_fakes/ test_fakes > .85:
            #     # best_loss = loss
            #     # print("Saving...")
            #     if torch.save(edge_predictor.state_dict(), EP_name):
            #         break
    
        # Save for reporting
    
    for key, val in test_dic.items():
        plt.plot(val, label = key)
        
    plt.legend() 
    plt.title("Accuracy of Predictions")   
    plt.savefig(args.dataset + "-EdgePredictorTrainingAcc.png")
    # plt.show()
    
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.savefig(args.dataset + "-EdgePredictorTrainingLoss.png")
    # plt.show()
    #Load best
    edge_predictor.load_state_dict(torch.load(EP_name))
    frm = data.x[all_edges[0][edge_test_idx]]
    to = data.x[all_edges[1][edge_test_idx]]
    test_set = torch.cat( (frm, to), dim = 1)
    
    #Train edge predictor
    
    test_lbls = edge_labels[edge_test_idx].to(torch.long)
    t = time.time()
       
    #Concat nodes
    edge_predictor.eval()
    preds = F.softmax(edge_predictor(test_set)).to(torch.float)
    # preds.requires_grad = True
    # preds = preds.argmax(dim = 1).to(torch.float)
    # preds.require_grad = True
    loss = test_loss(preds[:, 1], test_lbls.float())
    preds = preds.argmax(dim = 1).to(torch.float)
    num_correct = sum(preds == edge_labels[edge_test_idx])
    
    print("\n\n\tTest Result:  Edges, Epoch: ", i, " Training Loss: ", loss.item(),  "Correct: ", num_correct.item(), "Acc: ", (num_correct/ edge_labels[edge_test_idx].shape[0]).item(),
          "\n\tReal Correct: ", correct_reals, "Real Acc: ", correct_reals/ test_reals, " Correct fakes: ", correct_fakes, 
          " Fake Acc: ", correct_fakes/test_fakes,  " Time: ", time.time() - t)
    print("\n\n")
    # Save for reporting

def prepare_edges_homo(args, data, ae):
    #Pair real edges 
    real_edges = torch.cat( (data.x[data.edge_index[0]], data.x[data.edge_index[1]]), 1 )
    edge_save_file = ".\\FullEdgesFor" + args.dataset.lower() + "Predictor.csv"  #Check if this is present before training.
    #Make fake edges
    def ismember(a, b):
        return np.intersect1d(torch.where(data.edge_index[0] == a)[0] , torch.where(data.edge_index[1] == b)[0]).shape[0] > 0

    def ismemberoflist(a, b, arr):
        if a in arr and b in arr:
            return np.intersect1d(np.where(arr ==a ), np.where(arr == b) ).shape[0] > 0
        return False


    edges_needed = data.edge_index.shape[1] #real_edges.shape[0]
    edges_created = 0
    test_edges_false = []
    first_pass_counter = 0
    while edges_created < edges_needed:
        if first_pass_counter < 10:
            print("Looping")
        #     idx_ia = np.random.randint(0, data.edge_index[0].max(), 2708)
        #     idx_ja = np.random.randint(0, data.edge_index[0].max(), 2708)
            idx_ia = torch.randperm(data.edge_index[0].max() + 1)[: data.edge_index[0].max()]
            idx_ja = torch.randperm(data.edge_index[0].max() + 1)[: data.edge_index[0].max()]
            for idx_i, idx_j in zip(idx_ia, idx_ja):
                idx_i, idx_j = idx_i.item(), idx_j.item()
                if edges_created == edges_needed:
                    break;
                if idx_i == idx_j:
                    continue
                if ismember(idx_i, idx_j):
                    continue
                if test_edges_false:
                      if ismemberoflist(idx_j, idx_i, np.array(test_edges_false)):
                          continue
                      if ismemberoflist(idx_i, idx_j, np.array(test_edges_false)):
                          continue
                test_edges_false.append([idx_i, idx_j])
                edges_created += 1
                print(f'{edges_created} of {edges_needed}')
                
        else:
         
            print("Brute forcing....")
            idx_ia = torch.randperm(data.edge_index[0].max() + 1)[: 500]
            idx_ij = torch.randperm(data.edge_index[0].max() + 1)[: data.edge_index[0].max()]
            for idx_i in idx_ia:
                idx_i.item()
                
                for idx_j in idx_ja:
                    idx_j.item()
                  
                if edges_created == edges_needed:
                    break;
                if idx_i == idx_j:
                    continue
                if ismember(idx_i, idx_j):
                    continue
                if test_edges_false:
                      if ismemberoflist(idx_j, idx_i, np.array(test_edges_false)):
                          continue
                      if ismemberoflist(idx_i, idx_j, np.array(test_edges_false)):
                          continue
                test_edges_false.append([idx_i, idx_j])
                edges_created += 1
                print(f'{edges_created} of {edges_needed}')
        first_pass_counter += 1   
                    
    #Create self loops
    #Add self loops
    ids = [x for x in range(data.x.shape[0])]
    selfs = []
    for x in ids:
            selfs.append([x, x])
            
    test_edges_false_idx = test_edges_false[:edges_needed]
    reals =  torch.cat( (data.edge_index[:, :edges_needed], torch.tensor(selfs).t()), dim = 1) 
    all_edges = torch.cat( (reals,  torch.tensor(test_edges_false).t() ), dim = 1 )
    
    torch.save(all_edges, edge_save_file)

    all_edges = torch.load(edge_save_file)
 

    edge_labels = torch.cat( (torch.ones_like(all_edges[0][:(edges_needed + data.x.shape[0])]), torch.zeros_like(data.edge_index[0][:edges_needed])) )  

 
    num_edges = edge_labels.shape[0]
    
    #Assume an 80/20 train test split
    train_edges = math.floor(num_edges * .8)
    test_edges = num_edges - train_edges
    
    idx = np.arange(num_edges)
    np.random.shuffle(idx)
    edge_train_idx = torch.tensor(idx[:train_edges] )
    edge_test_idx = torch.tensor( idx[train_edges:] )
    print("Edges Done")

    return all_edges, edge_labels, train_edges, test_edges, edge_train_idx, edge_test_idx

def prepare_edges_quick(args, data):
    #Pair real edges 
    # real_edges = torch.cat( (data.x[data.edge_index[0]], data.x[data.edge_index[1]]), 1 )
    edge_save_file = ".\\FullEdgesFor" + args.dataset.lower() + "Predictor.csv"  #Check if this is present before training.
    
    edges_needed = data.edge_index.shape[1] #real_edges.shape[0]
    all_edges = torch.load(edge_save_file)
 

    edge_labels = torch.cat( (torch.ones_like(all_edges[0][:(edges_needed + data.x.shape[0])]), torch.zeros_like(data.edge_index[0][:edges_needed])) )  

 
    num_edges = edge_labels.shape[0]
    
    #Assume an 80/20 train test split
    train_edges = math.floor(num_edges * .8)
    test_edges = num_edges - train_edges
    
    idx = np.arange(num_edges)
    np.random.shuffle(idx)
    edge_train_idx = torch.tensor(idx[:train_edges] )
    edge_test_idx = torch.tensor( idx[train_edges:] )
    print("Edges Done")

    return all_edges, edge_labels, train_edges, test_edges, edge_train_idx, edge_test_idx
        
def train_edge_predictor_for_homo(args, data, save_file_name, ae_name, device):
    #Get real and fake edges
    all_edges, edge_labels, train_edges, test_edges, edge_train_idx, edge_test_idx = prepare_edges_quick(args, data)
    
    #Get other numbers
    train_ratio = edge_train_idx.shape[0] / (edge_train_idx.shape[0] + edge_test_idx.shape[0])
    #Augment with homophilic edges. 
    num_nodes_generated = 50   #This will generate n*(n-1)/2 edges. 50 --> 1225
    labeled_nodes = data.x[data.few_shot_idx]
    label_of_interest = data.y[data.few_shot_idx]
    
    num_nodes = data.x.shape[0]
    X_train = data.x
    
    #Load autoencoder
    ae = NodeOnlyVariationalAutoEncoder(data.x.shape[1], 64, 64, 14, None, device)
    ae.load_state_dict(torch.load(ae_name))
    for l in data.y.unique():
        #Get indexes needed to select Xs of the current class.
        labeled_indexes = torch.where( data.y[data.few_shot_idx] == l )
        
        to_seed = data.x[labeled_indexes].clone()
        while(to_seed.shape[0] < num_nodes_generated):
            to_seed = torch.cat( (to_seed, data.x[labeled_indexes].clone()), dim = 0)   
        
        #Reduce to the proper size
        to_seed = to_seed[:num_nodes_generated]
    
        #Generate the new nodes.
        out = ae(to_seed, None)[0]   #All of these are the same labels.
                            #Out needs to be appended to X somewhere.
        
        #Add generated nodes to the list of nodes.
        X_train = torch.cat( (X_train, out), dim = 0)
        
        edge_adds = [[],[]]
        
        #Generate a fully connected subgraph. 
        for new_node_idx in range(num_nodes_generated): #For every index
            for remainder_idx in range(new_node_idx, num_nodes_generated):
                edge_adds[0].append(num_nodes + new_node_idx)
                edge_adds[1].append(num_nodes + remainder_idx)
                
                #And in reverse
                edge_adds[1].append(num_nodes + new_node_idx)
                edge_adds[0].append(num_nodes + remainder_idx)
    
        #Prepare with details about existing edges
        previous_edge_count = all_edges.shape[1]
        
        #Add to train/test splits.
        num_idxes = len(edge_adds[1])       #####May need to adjust so that edges in both directions are not included in both sets.
        idx = np.arange(num_idxes)
        np.random.shuffle(idx)
        
        #Shift indexes to the end of the list
        idx +- previous_edge_count
        divider = math.floor(train_ratio * idx.shape[0])
        train_add = torch.tensor(idx[:divider] )
        edge_train_idx = torch.cat( (edge_train_idx, train_add)) 
       
        test_add = torch.tensor( idx[divider:] )
        edge_test_idx = torch.cat( (edge_test_idx, test_add))
        
        
        #Add edges to the tensor. 
        all_edges = torch.stack( (torch.cat( (all_edges[0], torch.tensor(edge_adds[0])) ),
                                torch.cat( (all_edges[1], torch.tensor(edge_adds[1])) ) ), dim = 1).t()
        #Append edge labels.
        edge_labels = torch.cat( (edge_labels, torch.ones(len(edge_adds[0]) ) ) )
    
       #Debug statement
        print(f'IDX len is {edge_train_idx.shape[0] + edge_test_idx.shape[0]} comapre to {all_edges.shape}')
        num_nodes = X_train.shape[0]
        
        ####This should be correct.
    in_dim = data.x.shape[1]
    edge_predictor = FNN(in_dim, math.floor(in_dim/ 6), 2, 2) 

    print(edge_predictor)
    
    edge_optimizer = optim.Adam(edge_predictor.parameters(), lr=0.0001)
    class_weight = [1 if x == 1 else 1 for x in edge_labels]
    class_weight = torch.FloatTensor(class_weight)

    loss_function = torch.nn.BCELoss(weight=class_weight[edge_train_idx])
    test_loss = torch.nn.BCELoss(weight=class_weight[edge_test_idx])

    frm = data.x[all_edges[0][edge_train_idx]]
    to = data.x[all_edges[1][edge_train_idx]]
    train_set = frm + to #torch.cat( (frm, to), dim = 1)
    
    #Train edge  
    best_loss = 999999999999
    lbls = edge_labels[edge_train_idx].to(torch.long)
    # lbls.requires_grad = True
     #Load best
    frm = data.x[all_edges[0][edge_test_idx]]
    to = data.x[all_edges[1][edge_test_idx]]
    test_set = to + frm #torch.cat( (frm, to), dim = 1)
    test_lbls = edge_labels[edge_test_idx].to(torch.long)
    num_real = sum(edge_labels[edge_train_idx] == 1 )
    num_fake = sum(edge_labels[edge_train_idx] == 0 )
    test_reals = sum(edge_labels[edge_test_idx] == 1)
    test_fakes = sum(edge_labels[edge_test_idx] == 0)
    test_dic = {'correct':[],
                'reals':[],
                'fakes':[]}
    loss_history = []

    EP_name = save_file_name
    best_acc = 0.0
    decline_counter = 0
    # edge_predictor.load_state_dict(torch.load(EP_name))
    for i in range(10000):  # tqdm(range(5)):
        edge_predictor.train()
        # Setups
        edge_optimizer.zero_grad()
        t = time.time()
       
        #Concat nodes
        
        preds = F.softmax(edge_predictor(train_set)).to(torch.float)
        num_preds = F.relu(preds[:,1].sum() - preds[:,0].sum())
        # preds.requires_grad = True
        # preds = preds.argmax(dim = 1).to(torch.float)
        # preds.require_grad = True
        loss = loss_function(preds[:, 1], lbls.float()) + num_preds
        preds = preds.argmax(dim = 1).to(torch.float)
        num_correct = sum(preds == edge_labels[edge_train_idx])
        real_correct = sum((preds == 1) & (edge_labels[edge_train_idx] ==  1)).item()
        fakes = sum( (preds == 0) & (edge_labels[edge_train_idx] == 0))
    #   neighbor_loss = neighbor_loss_func(neighbors[idx_train], ae.last_samples[idx_train], ae.sampled_mean[idx_train], ae.sampled_sigma[idx_train])
        # loss_train = .4 * feature_loss + .2 * degree_loss   + .4 * n_loss
        # loss_train = feature_loss
        # loss_train = loss_func(X_bar[idx_train], data.x[idx_train])
        loss.backward()
        edge_optimizer.step()
        # print("Dataset Name:  Edges, Epoch: ", i, " Training Loss: ", loss.item(),  "Correct: ", num_correct, "Acc: ", (num_correct/ edge_labels[edge_train_idx].shape[0]).item(), " \n\tFakes IDed: ", fakes.item(),
        #       " Fake Acc: ", fakes/num_fake, " Real Acc: ", real_correct/ num_real, " Time: ", time.time() - t)
    
        if i % 100 == 0:
            loss_history.append(loss.item())
            #Train edge predictor
            
           
            t = time.time()
               
            #Concat nodes
            edge_predictor.eval()
            preds = F.softmax(edge_predictor(test_set)).to(torch.float)
            # preds.requires_grad = True
            # preds = preds.argmax(dim = 1).to(torch.float)
            # preds.require_grad = True
            loss = test_loss(preds[:, 1], test_lbls.float())
            preds = preds.argmax(dim = 1).to(torch.float)
            num_correct = sum(preds == edge_labels[edge_test_idx]).item()
            correct_reals = sum((preds == 1) & (edge_labels[edge_test_idx] == 1) ).item()
            correct_fakes = sum((preds == 0) & (edge_labels[edge_test_idx] == 0) ).item()
            test_dic['correct'].append(num_correct / edge_test_idx.shape[0] ) 
            test_dic['reals'].append( (correct_reals / sum(edge_labels[edge_test_idx] == 1)).item())
            test_dic['fakes'].append( (correct_fakes/ sum(edge_labels[edge_test_idx] == 0) ).item())
            acc = (num_correct/ edge_labels[edge_test_idx].shape[0])
            print("\nTest Result:  Edges, Epoch: ", i, " Training Loss: ", loss.item(),  "Correct: ", num_correct, "Acc: ", (num_correct/ edge_labels[edge_test_idx].shape[0]), 
                  "\n\tReal Correct: ", correct_reals, "Real acc: ", correct_reals / test_reals, " Correct fakes: ", correct_fakes, " Fake Acc: ", correct_fakes/ test_fakes, " Time: ", time.time() - t)
            
            if best_acc < acc:
                best_acc = acc
                print("Saving...")
                torch.save(edge_predictor.state_dict(), EP_name)
                decline_counter = 0
            else:
                decline_counter += 1
                if decline_counter == 15:
                    pass
            # if   correct_fakes/ test_fakes > .85:
            #     # best_loss = loss
            #     # print("Saving...")
            #     if torch.save(edge_predictor.state_dict(), EP_name):
            #         break
    
        # Save for reporting
    
    for key, val in test_dic.items():
        plt.plot(val, label = key)
        
    plt.legend() 
    plt.title("Accuracy of Predictions")   
    plt.savefig(args.dataset + "-EdgePredictorTrainingAcc.png")
    # plt.show()
    
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.savefig(args.dataset + "-EdgePredictorTrainingLoss.png")
    # plt.show()
    #Load best
    edge_predictor.load_state_dict(torch.load(EP_name))
    frm = data.x[all_edges[0][edge_test_idx]]
    to = data.x[all_edges[1][edge_test_idx]]
    test_set = frm + to # torch.cat( (frm, to), dim = 1)
    
    #Train edge predictor
    
    test_lbls = edge_labels[edge_test_idx].to(torch.long)
    t = time.time()
       
    #Concat nodes
    edge_predictor.eval()
    preds = F.softmax(edge_predictor(test_set)).to(torch.float)
    # preds.requires_grad = True
    # preds = preds.argmax(dim = 1).to(torch.float)
    # preds.require_grad = True
    loss = test_loss(preds[:, 1], test_lbls.float())
    preds = preds.argmax(dim = 1).to(torch.float)
    num_correct = sum(preds == edge_labels[edge_test_idx])
    
    print("\n\n\tTest Result:  Edges, Epoch: ", i, " Training Loss: ", loss.item(),  "Correct: ", num_correct.item(), "Acc: ", (num_correct/ edge_labels[edge_test_idx].shape[0]).item(),
          "\n\tReal Correct: ", correct_reals, "Real Acc: ", correct_reals/ test_reals, " Correct fakes: ", correct_fakes, 
          " Fake Acc: ", correct_fakes/test_fakes,  " Time: ", time.time() - t)
    print("\n\n")
    # Save for reporting
        