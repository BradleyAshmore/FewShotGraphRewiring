# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:41:12 2024

@author: ashmo

This is the models used for the fewshot experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor, matmul
import math

#####Models

class NodeEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim,  latent_dim=2):
        super(NodeEncoder, self).__init__()

        # Networks for encoding.
        mid = math.floor( (in_dim - hidden_dim) / 2) + hidden_dim
        # self.graphconv1 = GCNConv(in_dim, hidden_dim)
        self.graphconv1 = nn.Linear(in_dim, hidden_dim)
        self.graphconv2 = nn.Linear(hidden_dim, latent_dim)
        self.graphconv_mean = nn.Linear(latent_dim, latent_dim)
        self.graphconv_std = nn.Linear(latent_dim, latent_dim)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, edge_index):
        X = self.leakyrelu(self.graphconv1(X)) #, edge_index))  # , edge_index))
        X = self.leakyrelu(self.graphconv2(X))
        mean = self.graphconv_mean(X)  # , edge_index)
        var = self.graphconv_std(X) #, edge_index)

        return mean, var, X


class NodeDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim=2):
        super(NodeDecoder, self).__init__()
        mid = math.floor( (in_dim - hidden_dim) / 2) + hidden_dim
        # Attempt as Graphs
        self.hidden_conv1 = nn.Linear(latent_dim, hidden_dim)
        self.hidden_conv2 = nn.Linear(hidden_dim, in_dim)
        # self.hidden_conv2 = nn.Linear(hidden_dim, hidden_dim)
        # self.out_conv = GCNConv(hidden_dim, in_dim) # nn.Linear(mid, in_dim)
        # self.out_conv = nn.Linear(hidden_dim, in_dim)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm1d(in_dim, in_dim)
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, in_dim)
        self.relu = nn.ReLU()
    def forward(self, Z, edge_index):
        # , edge_index) ) #, edge_index) )
        # Z = self.leakyrelu(self.hidden_conv1(Z))
        # Z = F.relu(self.hidden_conv2(Z))#, edge_index) )
        # # X_prime = self.out_conv(Z) #, edge_index)  # , edge_index)
        # Z = self.bn(Z)
        
        
        Z = self.linear1(Z)
        Z = self.leakyrelu(Z)
        
        Z = self.linear2(Z)
        Z = self.relu(Z)
        return Z


class MLP_generator(nn.Module):
    def __init__(self, input_dim, output_dim, sample_size):
        super(MLP_generator, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.linear3 = nn.Linear(output_dim, output_dim)
        self.linear4 = nn.Linear(output_dim, output_dim)

    def forward(self, embedding, device):
        neighbor_embedding = F.relu(self.linear(embedding))
        neighbor_embedding = F.relu(self.linear2(neighbor_embedding))
        neighbor_embedding = F.relu(self.linear3(neighbor_embedding))
        neighbor_embedding = self.linear4(neighbor_embedding)
        return neighbor_embedding



class NodeAndStructureEncoder(nn.Module):
    def __init__(self, feature_in_dim, num_nodes, hidden_dim, latent_dim):
        super(NodeAndStructureEncoder, self).__init__()
        
        self.layerX = MLP(1,feature_in_dim, hidden_dim, hidden_dim)
        self.layerA = MLP(1, num_nodes, hidden_dim, hidden_dim)
        self.layerCombine = MLP(1, hidden_dim * 2, latent_dim, latent_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim,latent_dim)
        self.mean_layer = MLP(1, latent_dim, latent_dim, latent_dim)
        self.std_layer = MLP(1, latent_dim, latent_dim, latent_dim)
        
        self.leakyrelu = F.leaky_relu
        self.num_nodes = num_nodes
        self.GCN1 = GCN(feature_in_dim, hidden_dim)
        self.GCN2 = GCN(hidden_dim, latent_dim)
        
    def forward(self, X, A):
        m = self.num_nodes #X.shape[0]
        X_hat = self.layerX(X)
        X_hat = self.leakyrelu(X_hat)
        row, col = A
        A_hat = SparseTensor(row=row, col=col, sparse_sizes=(m, m) ).to_torch_sparse_coo_tensor()

        A_hat = self.layerA(A_hat)
        A_hat = self.leakyrelu(A_hat)
        
        concatenated_data = torch.cat( (X_hat, A_hat), dim=1 )
        concatenated_data = self.layerCombine(concatenated_data)
        # X_hat = self.GCN1(X,A)
        # X_hat - self.leakyrelu(X_hat)
        # concatenated_data = self.GCN2(X_hat, A)
        concatenated_data = self.leakyrelu(concatenated_data)
        # concatenated_data = self.bn2(concatenated_data)
        
        mean = self.mean_layer(concatenated_data)
        std = self.std_layer(concatenated_data)
        
        return concatenated_data, mean, std
        
#This is an autoencoder that accepts both node information and structural information.
#Node features and strucutre data is fed to independent MLPs before being combined
#to allow for heterophily considerations. 
#Only node freatres are reconstructed.
class NodeAndStructureVariationalAutoEncoder(nn.Module):
    def __init__(self, features_in_dim, num_nodes, hidden_dim, latent_dim = 2):
        super(NodeAndStructureVariationalAutoEncoder, self).__init__() 
        self.encoder = NodeAndStructureEncoder(features_in_dim, num_nodes, hidden_dim, latent_dim)
        
        self.decoder = NodeDecoder(features_in_dim, hidden_dim, latent_dim=latent_dim)
        
        self.out_dim = features_in_dim

    def reparamiterize(self, mean, var):
        epsilon = torch.randn_like(var)
        Z = mean + var * epsilon
        return Z

    def forward(self, X, edge_index, reparamiterize = True):
      
        # Encode into a distribution.
        # mu, sig, X_enc  = self.encoder(X_0, edge_index)
        mu, sig, X_enc = self.encoder(X, edge_index)
  
        # Reparameterize distribution.
        if reparamiterize:
            Z = self.reparamiterize(mu, sig)
        else:
            Z = X_enc

        # Decode nodes
        X_1 = self.decoder(Z, edge_index)

        # Saving X_1 incase it is needed for debugging in the future.
        X_1 = F.relu(X_1)

        return X_1, mu, sig  #, neighbors, n_loss
        
class NodeOnlyVariationalAutoEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim,  latent_dim=2):

        super(NodeOnlyVariationalAutoEncoder, self).__init__()

        self.encoder = NodeEncoder(
            in_dim,  hidden_dim,  latent_dim=latent_dim)

        self.decoder = NodeDecoder(
            in_dim, hidden_dim,  latent_dim=latent_dim)

        self.out_dim = in_dim
    def reparamiterize(self, mean, var):
        epsilon = torch.randn_like(var)
        Z = mean + var * epsilon
        return Z

    def forward(self, X, edge_index, reparamiterize = True):
        # Reduce dimension.
        # X_0 = self.mlp0(X)

        # Encode into a distribution.
        # mu, sig, X_enc  = self.encoder(X_0, edge_index)
        mu, sig, X_enc = self.encoder(X, edge_index)
  
        # Reparameterize distribution.
        if reparamiterize:
            Z = self.reparamiterize(mu, sig)
        else:
            Z = X_enc

        # Decode nodes
        X_1 = self.decoder(Z, edge_index)

        # Saving X_1 incase it is needed for debugging in the future.
        X_1 = F.relu(X_1)

        # Back to full vector size
        # X_hat = self.mlp_back(X_1)
        # X_hat = F.relu(X_hat)
        # neighbors, n_loss = self.decode(Z, X_enc)
        return X_1, mu, sig  #, neighbors, n_loss
 

       

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


# FNN
class FNN(nn.Module):
    def __init__(self, in_features, hidden, out_features, layer_num):
        super(FNN, self).__init__()
        self.linear1 = MLP(1, in_features, hidden, hidden)
        self.linear2 = nn.Linear(hidden, out_features)

    def forward(self, embedding):
        x = self.linear1(embedding)
        x = self.linear2(F.relu(x))
        return x

class GCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, 24)
        # torch.nn.init.normal_(self.conv1.lin.weight, mean=0, std=0.3)
        # torch.nn.init.normal_(self.conv1.lin.bias, mean=0, std=0.01)
        self.conv2 = GCNConv(24, out_features)

    def forward(self, x, adj):
        x = self.conv1(x, adj)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, adj)
        return x

###############################End of models
