# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:00:54 2024

@author: ashmo

VAE Testing
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
import torch.optim as optim




from pathlib import Path
from FewShotModels import FNN, NodeOnlyVariationalAutoEncoder, NodeAndStructureVariationalAutoEncoder
from FewShotUtilities import gcn_eval, load_cora, load_texas,load_citeseer,load_cornell,load_wisco, set_few_shot_labels, gcn_test
from FewShotTrainingFunctions import train_autoencoder, train_edge_predictor, train_edge_predictor_for_homo


from ExperamentalScores import generate_similarity_score, train_mlp
from ScoreModels import MLP

from GraphAnalysis import *
import matplotlib.pyplot as plt


def kl_loss(predictions, targets, mean, log_var, KDL_factor=-0.1):
    x1 = predictions  # predictions.squeeze().cpu().detach()
    x2 = targets  # targets.squeeze().cpu().detach()

    repoduction_loss = F.mse_loss(x1, x2, reduction='mean')
    KDL = KDL_factor * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    KL_loss = repoduction_loss + KDL

    return KL_loss

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

def clean_up(t1):
    t = t1.clone()
    problems = t > 1
    t[problems] = 1
    return t

def threshold(t):
    thresh = torch.nn.Threshold(0.15, 0)
    return thresh(t)

# Training
#This saves the autoencoder to the disk to be read later.
def train_autoencoder(data, device, save_file_name):
    in_dim = data.x.shape[1]
    
    ae = NodeAndStructureVariationalAutoEncoder(in_dim, data.x.shape[0], 32, 4)
    # ae=  NodeOnlyVariationalAutoEncoder(in_dim, 256, 14)
    optimizer = optim.Adam(ae.parameters(),
                           lr=0.001)
    # feature_loss_func = nn.MSELoss()
    feature_loss_func = kl_loss #nn.KLDivLoss()

    # save_file_name = ".\\ExperimentalVAE.mdl"
    best_loss = 100000
    best_model = None
    ae_loss = []
    perfect_loss = []
    
    ae_mask = data.val_mask + data.train_mask
    best_ae = None
    for i in range(12000):  # tqdm(range(5)):
        
        # Setups
        optimizer.zero_grad()
        ae.train()
        t = time.time()

        # X_hat, mu, sig = ae(data.x[ae_mask], data.edge_index)
        X_hat, mu, sig = ae(data.x, data.edge_index)
        feature_loss = feature_loss_func(X_hat[ae_mask], data.x[ae_mask], mu, sig, KDL_factor=-0.5)
        loss_train =  feature_loss #.75 * feature_loss + .25 * degree_loss
        loss_train.backward()
        optimizer.step()
        print(" Epoch: ", i, " Training Loss: ", loss_train.item(), "Feature Loss: ", feature_loss, " Time: ", time.time() - t)

        if loss_train.item() <= best_loss:
            # Save model
            best_loss = loss_train.item()
            print("Saving...")
            # torch.save(ae.state_dict(), save_file_name)
            best_ae = ae.state_dict()

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
    
    torch.save(best_ae, save_file_name)
    plt.plot(ae_loss)
    plt.title('Test Loss for Node and Structure Autoencoder')
    plt.savefig('.\\AutoencoderLossFromTestScript.png')


#Set args
#######################################################
#Parser prep

####Options needed 



######To Do
    #  XXX Return adds and subtracts from the rewiring epoch
    #Homophily study
    #Plot curvature
    #  XXX Add functionality for iterations.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('-f')
parser.add_argument('--dataset', type=str, default="texas")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--edge_delta', type=float, default=.5)
# neighbor reconstruction loss weight
parser.add_argument('--oversample_method', default = "none")
parser.add_argument('--oversample_multiplier', default=5)
parser.add_argument('--add_edges_to', default='sample')
parser.add_argument('--pair_to_synthetic', default=True)
parser.add_argument('--pair_to_labeled', default=True)
parser.add_argument('--repetitions', default=1)
parser.add_argument('--shot_size', default=5)
parser.add_argument('--threshold', default=2.0)
parser.add_argument('--verbose', default=False)
parser.add_argument('--homo_focus', default=True)

args = parser.parse_args()

#Set args as appropriate datatypes
args.oversample_multiplier = int(args.oversample_multiplier)
args.pair_to_synthetic = args.pair_to_synthetic == "True"
args.pair_to_labeled = args.pair_to_labeled == "True"
args.repetitions = int(args.repetitions)
args.shot_size = int(args.shot_size)
args.threshold = float(args.threshold)
args.edge_delta = float(args.edge_delta)




# fn = make_filename(args)
        

for rep in range(args.repetitions):
   
    #Autoencoder things.
        #Load pretrained models, ae --> autoencoder ep --> edge predictor
    # ae_name, ep_name = get_pretria.ned_model_names(args)
    # ae_path = Path(ae_name)
    # ep_path = Path(ep_name)
    ae_name = f'.\\NodeAndStuructureVAEfor-{args.dataset}.mdl'
        #Train model
    data = load_dataset_from_args(args)
    
    
    train_autoencoder(data, device, ae_name)

    #Load autoencoder
    ae = NodeAndStructureVariationalAutoEncoder(data.x.shape[1], data.x.shape[0], 32, 4)
    ae.load_state_dict(torch.load(ae_name))
    
    #Genearte data and check differences
    x_out, mean, std = ae(data.x, data.edge_index)
    rounded = x_out.round()
    high = x_out.ceil()
    floo = x_out.floor()
    rawdiff = (data.x - x_out).sum(dim=1).mean()
    # rdiff = (data.x - rounded).sum(dim=1).mean()
    # hdiff = (data.x - high).sum(dim=1).mean()
    # fdiff = (data.x - floo).sum(dim=1).mean()
    cleandiff = (data.x - clean_up(x_out)).sum(dim=1).mean()
    crdiff = (data.x - clean_up(x_out).round()).sum(dim=1).mean() 
    ct = (data.x - threshold(clean_up(x_out))).sum(dim=1).mean()
    tresh = (data.x - threshold(x_out)).sum(dim=1).mean()
    print(f'raw {rawdiff},  cleaned {cleandiff}, clean and round {crdiff}, clean and thresh {ct}, tresh {tresh}')
    print("hold")