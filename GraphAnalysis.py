# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 23:47:03 2024

@author: ashmo

This file houses graph analysis functions.
"""

#Imports
import torch
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.utils import degree
from torch_geometric.transforms import RootedEgoNets
import matplotlib.pyplot as plt 
import math
import copy
import numpy as np
import random as r
import time 

'''
This class record the manipulations applied to an egograph.
Node additions are not explicitly recorded. This class relies on the functionality
of Networkx's Graph object to handle node additions.
'''
class GraphModifier():
    def __init__(self, ego_id, node_limit, allow_node_additions=False):
        #Added and removed are list of tuples containing two elements.
        self.ego_node = ego_id
        self.added_edges = []
        self.removed_edges = []
        self.last_original_node = node_limit
        self.added_nodes = []
        self.allow_adds = allow_node_additions
        
    '''
    Checks and edge to make sure it is a tuple with two elements.
    An exception is thrown if either check fails.
    '''
    def edge_safety_check(self, edge):
        if type(edge) != tuple:
            raise Exception("Edge is not a tuple.")
            
        if len(edge) != 2:
            raise Exception("Edge must contain exactly two nodes.")
      
    
    def has_a_new_node(self, edge):
        if edge[0] > self.last_original_node or edge[1] > self.last_original_node:
            if not edge[0] in self.added_nodes and not edge[1] in self.added_nodes:
                return True
        
        return False
        
    '''
    Records that an edge was added
    '''
    def add_edge(self, edge):
        self.edge_safety_check(edge)
        
        if self.allow_adds:
            if self.has_a_new_node(edge):
                #Add
                if edge[0] > self.last_original_node:
                    self.added_nodes.append(edge[0])
                    
                if edge[1] > self.last_original_node:
                    self.added_nodes.append(edge[1])
                    
        self.added_edges.append(edge)  
        
    '''
    Records that an edge was removed from the graph.
    '''            
    def remove_edge(self, edge):
        self.edge_safety_check(edge)
        self.removed_edges.append(edge)
        
    def num_edges_added(self):
        return len(self.added_edges)
    
    def num_edges_removed(self):
        return len(self.added_edges)
        
    def num_nodes_added(self):
        return len(self.added_nodes)
    
  
###
# This function breaks a graph down into a set of components. 
# If no label is provided all labels will be analized else only the desired label
# is addressed.

###
def get_component_stats(data, label=None):
    #Convert to a networkx graph
    g = to_networkx(data, to_undirected=True)
    
    #Get num components 
    num_con = nx.number_connected_components(g)
    
    ###There needs to be an if-statement here to account for the label option...
    #Get all components
    components = nx.connected_components(g)
    
    #Record size of the components
    componenent_size = []
    for x in components:
        componenent_size.append(len(x))
    
    # print(componenent_size)
    return num_con, componenent_size


def better_version_of_calculate_curve_scores(data, node_list = None, starter_product=None, rad=4):
    #Convert to a networkx graph
    g = to_networkx(data, to_undirected=True)
       
    #to_return is for ego-graphs
    to_return = []
    sim_list = []
    final_product = EdgeScoreOrganizer()
    final_product.merge(starter_product)
    
    if node_list == None:
        #This returns an EdgeScoreOrganizer
        eso = better_calcuate_curvature(g, final_product)
        # eso = torch_calculate_curvature(data)
        return eso
    
    else:   #There is a node list
    #curve_matrix probably needs to be a list of dictionaries. Index by centernode, keyed on pairs.
    #Build matrix for all ego graphs.
   
        if type(node_list) == torch.Tensor:
            node_list = node_list.tolist()
        
        
        #Find every edge that contains a node in node_list
        to_score = []          
        for x in node_list:
            to_score + list(g.edges(x))
            
        for e in to_score:
            final_product.merge(better_calcuate_curvature(g,final_product, e, force_recalculate=True) )
        
        
        
#Returns a series of networkx graphs 
##########This is the current test case!!!
#Node_list is the list of nodes that will be evaluated. 
def  calculate_curve_scores(data, node_list=None, rad=4):
    #Convert to a networkx graph
    g = to_networkx(data, to_undirected=True)
       
    #to_return is for ego-graphs
    to_return = []
    sim_list = []
    
    #curve_matrix probably needs to be a list of dictionaries. Index by centernode, keyed on pairs.
    curve_matrix = []
    #Build matrix for all ego graphs.
    ego_graph_storage = [] 
    curve_sums = []
    
    if node_list == None:   #Use entire graph
        print('c1urve_vectoring')
        curve_vect = calculate_curvature(g)        ###The return value of calc_curv needs to change.
        edge_scores = EdgeScoreOrganizer(curve_vect)
        curve_matrix.append(edge_scores)
        curve_sums.append(edge_scores.calc_sum())
        return curve_vect, curve_sums
    #Check parameters
    if type(node_list) == torch.Tensor:
        node_list = node_list.tolist()
        

    

    for x in node_list: ###For every node in the list.
        print(f'Working on {x}')
        #Get the ego-graph
        tt = nx.ego_graph(g, x, radius=rad, undirected=True)
        
        #Save for latter
        ego_graph_storage.append(tt)
       
        ###Will move this elsewhere.############
        # visulize_graph(tt, ego_val=x, data=data)
        # plt.show()
        ########################################
        to_return.append(tt)
        sim_list.append(get_ego_graph_stats(tt, data,x))
        curve_vect = calculate_curvature(tt)        ###The return value of calc_curv needs to change.
        edge_scores = EdgeScoreOrganizer(curve_vect)
        curve_matrix.append(edge_scores)
        curve_sums.append(edge_scores.calc_sum())
        
    return curve_matrix, ego_graph_storage, curve_sums


def only_add_nodes(data, ae, multiplier =5):
    new_node_number = data.x.shape[0]
    tensor_list = []
    label_list = []
    for n in data.few_shot_idx:
        for qqq in range(multiplier):
            tensor_list.append( ae(data.x[n], data.edge_index)[0] )
            label_list.append( data.y[n] )
            new_node_number += 1
            
    
    data.x = torch.cat( (data.x, torch.stack( tuple(tensor_list) ) ) )
    data.y = torch.cat( (data.y, torch.tensor(label_list) ) )
    
    data.few_shot_idx = torch.cat( (data.few_shot_idx, torch.tensor(label_list)))
    ss = torch.tensor( [True for ssss in label_list])
    data.few_shot_mask = torch.cat( (data.few_shot_mask, ss) )
    
    ss = torch.tensor( [False for ssss in label_list])
    data.test_mask = torch.cat( (data.test_mask, ss) )
    return data

def oversample_nodes_only(node_list, ae, data, multiplier=5):
    # new_node_number = data.x.shape[0]
    labeled_nodes = data.x[node_list]
    working_lables = data.y[node_list]
    
    multiples_needed = int(data.x.shape[0] / labeled_nodes.shape[0])    #This will be the number of iterations..
    
    for i in range(multiples_needed):
        hold = ae(data.x, data.edge_index)[0][node_list]
        labeled_nodes = torch.cat( (labeled_nodes, hold) )
        working_lables = torch.cat( (working_lables, data.y[node_list]) )
        print(f'{labeled_nodes.shape[0]}, {working_lables.shape[0]}')
        
    train_data = data.clone() 
    train_data.x = labeled_nodes
    train_data.y = working_lables
     
    # for c in range(multiplier):
    # #Generate a new set of nodes
        
    
    #     #For each node in list get ego-graph.
    #     for n in node_list:
    #         n = n.item()

    #         labels.append(data.y[c])
            
    #         new_nodes.append(generated[n])
    #         new_node_number += 1
        
        
    # #Add things back to data. 
    # data.x = torch.cat( (data.x, torch.stack( tuple(new_nodes) ) ) )
    # data.y = torch.cat( (data.y, torch.tensor(labels) ) )      
    
  
    
    # data.few_shot_idx = torch.cat( (data.few_shot_idx, torch.tensor(node_numbers_added)))
    # ss = torch.tensor( [True for ssss in labels])
    # data.few_shot_mask = torch.cat( (data.few_shot_mask, ss) )
    
    # ss = torch.tensor( [False for ssss in labels])
    # data.test_mask = torch.cat( (data.test_mask, ss) )
    # data.val_mask = torch.cat( (data.val_mask, ss) )        
    
    return train_data

def better_add_oversampled_nodes(curve_scores, node_list, ae, data, multiplier = 5):
    new_node_number = data.x.shape[0]
    g = to_networkx(data)
    
    new_edges = [[],[]]
    new_nodes = []
    node_numbers_added = []
    labels = []
    
    for c in range(multiplier):
    #Generate a new set of nodes
        generated = ae(data.x, data.edge_index)[0]
    
        #For each node in list get ego-graph.
        for n in node_list:
            n = n.item()
            ego_G = nx.ego_graph(g, n, radius=5)

            if ego_G.size() > 0:           
                edge, worst_curve = curve_scores.get_worst_score_from_set(list(ego_G.edges() ))
                
                #Two edges could be added
                candidate_edge_one = (edge[0], new_node_number)
                candidate_edge_two = (edge[1], new_node_number)
            
            else:
                candidate_edge_one = (n, new_node_number)
                candidate_edge_two = (n, new_node_number)
                
            if g.has_edge(candidate_edge_one[0], candidate_edge_one[1]):
                raise Exception(f'{candidate_edge_one}')
                
            if g.add_edge(candidate_edge_two[0], candidate_edge_two[1]):
                raise Exception(f'{candidate_edge_two}')
            #Add edge to the graph.
            g.add_edge(candidate_edge_one[0], candidate_edge_one[1])
            g.add_edge(candidate_edge_two[0], candidate_edge_two[1])
            
            
            new_edges[0].append(candidate_edge_one[0])
            new_edges[1].append(candidate_edge_one[1])
            
            new_edges[0].append(candidate_edge_two[0])
            new_edges[1].append(candidate_edge_two[1])
            # new_edges.append(candidate_edge_two)
            
            #Score new edges.
            
            curve_scores.merge( better_calcuate_curvature( g, focus_edge=candidate_edge_one,  force_recalculate=True ) )
            curve_scores.merge( better_calcuate_curvature( g, focus_edge=candidate_edge_two,  force_recalculate=True ) )
            
            node_numbers_added.append(new_node_number)
            
            labels.append(data.y[c])
            
            new_nodes.append(generated[n])
            new_node_number += 1
        
        
    #Add things back to data. 
    data.x = torch.cat( (data.x, torch.stack( tuple(new_nodes) ) ) )
    data.y = torch.cat( (data.y, torch.tensor(labels) ) )      
    
    temp = [[],[]]
    temp[0] = torch.cat( (data.edge_index[0], torch.tensor(new_edges[0])) )
    temp[1] = torch.cat( (data.edge_index[1], torch.tensor(new_edges[1])) )
      
    for s,d in zip(new_edges[0], new_edges[1]):
        g.add_edge(s, d)
        
    data.edge_index = from_networkx(g).edge_index
    
    # data.edge_index[0] = torch.tensor(temp[0])
    # data.edge_index[1] = torch.tensor(temp[1])
    
    data.few_shot_idx = torch.cat( (data.few_shot_idx, torch.tensor(node_numbers_added)))
    ss = torch.tensor( [True for ssss in labels])
    data.few_shot_mask = torch.cat( (data.few_shot_mask, ss) )
    
    ss = torch.tensor( [False for ssss in labels])
    data.test_mask = torch.cat( (data.test_mask, ss) )
    data.val_mask = torch.cat( (data.val_mask, ss) )        
    
    return new_edges, data

###
# Returns new ego_graph_list, and list of graph manipulations.
###
def add_oversampled_nodes(curve_matrix, ego_graph_list, node_list, ae, data, multiplier = 5):
    imp = 0
    new_node_number = data.x.shape[0]
    manipulations_list = []
    node_decoder = {}
    #Modify the current egographx
    for i, x in enumerate(curve_matrix):    #Curve_Matrix is a list of edgeorganizer objects.
        ego_G = ego_graph_list[i]
        manip = GraphModifier(node_list[i], data.x.shape[0]-1, allow_node_additions=True)

        for qqq in range(multiplier):
            ###Step 1: Add to worst score.
            #Get the index of the smallest edge
            # idx = torch.tensor(x).argmin()
            edge, worst_curve = x.get_worst_score()
            #Save score
            # worst_curve = x[idx]
            
            #Convert index to an edge
            # edge = list(ego_G.edges)[idx]
            
            #Two edges could be added
            candidate_edge_one = (edge[0], new_node_number)
            candidate_edge_two = (edge[1], new_node_number)
            
            #Add edges to the most bottlenecked node.
            ego_G.add_edge(candidate_edge_one[0], candidate_edge_one[1])
            ego_G.add_edge(candidate_edge_two[0], candidate_edge_two[1])
            
            manip.add_edge( (candidate_edge_one[0], candidate_edge_one[1]) )
            manip.add_edge( (candidate_edge_two[0], candidate_edge_two[1]) )
           
            manipulations_list.append(manip)
            
            #Modify scores
            # x[idx] += 1
            x.add_score(edge, calculate_curvature(ego_G, focus_edge=edge))
        
       #Get lastest score ---- This needs to be a full recalculationl
        # x.append(calculate_curvature(ego_G, focus_edge = candidate_edge_one))
        # x.append(calculate_curvature(ego_G, focus_edge = candidate_edge_two))
       
        #Add to data object
        # data.y = torch.cat( (data.y, data.y[node_list[i]].unsqueeze(dim=0))  )#Add a label that is the same as the ego node.
        # data.y = torch.cat( (data.y, torch.tensor(7).unsqueeze(dim=0)) )
        # visulize_graph( ego_G, ego_val=node_list[i], data=data)
        # plt.show()
        #Generate a node and add it to the decoder
            # node_decoder[new_node_number] = (ae(data.x[node_list[i]], data.edge_index)[0], data.y[node_list[i]])
            node_decoder[new_node_number] = (ae(data.x, data.edge_index)[0][node_list[i]], data.y[node_list[i]])
            # node_decoder[new_node_number] = (node_decoder[new_node_number][0][node_list[i], node_decoder[new_node_number][1]
                                             
            new_node_number += 1
    
    return ego_graph_list, manipulations_list, node_decoder
       # #Get updated score
       #  new_score_one = calculate_curvature_experiment(nx.ego_graph(), candidate_edge_one, edge)
       #  new_score_two = calculate_curvature_experiment(nx.ego_graph(), candidate_edge_two, edge)
        
       #  #Check for improvement
       #  if new_score_one > worst_curve or new_score_two > worst_curve:
       #      new_score = max(new_score_one, new_score_two)
            
       #      if new_score_one > new_score_two:
       #          edge = (edge[0], new_node_number)
       #      else:
       #          edge = (edge[1], new_node_number)
                
       #      #Record improvement
       #      imp += new_score - worst_curve
       #      ###The stochastic element needs to be added here....
            
       #      #Update score
       #      curve_matrix[i][idx] = new_score    
            
       #      ego_graph_storage[i].add_node
            
       #      #Need a new score at the end...
       #  ####Step 2: Remove best score.
        
       #  break
    #Using form_networkx creates a graph using the index of the nodes in the nx nodeview. This needs to be 
    #considered for reconstruction.  

    # print( sim_list)
    # print("Final average")
    # print(torch.tensor(sim_list).mean().item())
    # return to_return
###
# Determines homophily percentage for the graph. 
###
def get_ego_graph_stats(G, data, ego_val, label=None):
    idxes = list(G.nodes)
    labels = data.y[idxes]
    homo_base = data.y[ego_val].item()
    
    stat_dic = {}
    
    for q in labels:
        if q in stat_dic.keys():
            stat_dic[q] += 1
        else:
            stat_dic[q] = 1
    
    in_tot = 0
    out_tot = 0
    for k in stat_dic.keys():
        if k == homo_base:
            in_tot = stat_dic[k]
        else:
            out_tot += stat_dic[k]
            
    return in_tot/ (in_tot + out_tot)

#G uses the true node IDs.
def visulize_graph(G,ego_val = None, data = None):
    if data != None:    #Use label data.
        #Get labels for each node
        idxes = list(G.nodes)
        labels = data.y[idxes]
        edge_map = []
        color_decoder = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'lightgreen']
        color_map = []
        for q, i in zip(labels, idxes):
            if i == ego_val:
                edge_map.append(300)
            else:
                edge_map.append(100)
                
            color_map.append(color_decoder[q])
    
        nx.draw(G, node_color=color_map, node_size=edge_map)
    else:
        nx.draw(G)     
 
"""
G is a graph object
new_edge is a tuple-pair of integers which represent node IDs.
"""
def calculate_curvature_experiment(G, new_edge, eval_at_edge, remove_edge=False):
    # G_hat = G.copy()    #Make sure this doesn't need to be a deep copy.
    
    #Add new edge
    G.add_edge(new_edge[0], new_edge[1])
    
    improvement = calculate_curvature(G, eval_at_edge)
    
    G.remove_edge(new_edge[0], new_edge[1])
    return improvement
    
def sub_function_multi(G, edge)    :
      print(f'Subfunction with {edge}')
      src, dst = edge
      
      #Check degrees.
      ds = max(G.degree(src), 1)
      dd = max(G.degree(dst), 1)
      if min(ds, dd) == 1:    #is this correct???
          return 0
      else:
          #Extract and merge ego_graphs to keep the runtime reasonable
          ego1 = nx.ego_graph(G, edge[0], radius=4)
          ego2 = nx.ego_graph(G, edge[1], radius=4)
          combo = nx.compose(ego1, ego2)
          #Get triangles
          #Get the cycles from the whole graph.
          
          # cycles1 = find_all_cycles(combo, source=edge[0], cycle_length_limit=4)
          # cycles2 = find_all_cycles(combo, source=edge[1], cycle_length_limit=4)
          cycles1_gen = nx.simple_cycles(combo, length_bound=4)
          cycles2_gen = nx.simple_cycles(combo, length_bound=4)
          
          #Check for duplicates and reduce to cycles of 3 and 4 nodes.
          to_delete = [] 
          cycles1 = []
          for i, x in enumerate(cycles1_gen):
              l = len(x)
              if l == 3 or l == 4:
                  cycles1.append(sorted(x))
            
          # #Reverse the list
          # to_delete = sorted(to_delete, reverse=True)                    
          # for x in to_delete:
          #     del(cycles1[x])
           
          cycles2 = []
          to_delete = [] 
          for i, x in enumerate(cycles2_gen):
              l = len(x)
              if l == 3 or l == 4:
                  hold =  sorted(x) 
                  if not hold in cycles1:
                      cycles2.append(hold)
              
          # to_delete = sorted(to_delete, reverse=True)                    
          # for x in to_delete:
          #     del(cycles2[x])
              
              
          tri_count = 0
          
          four_src = 0
          four_dst = 0
          four_common = 0
          for cyc in cycles1 + cycles2:
              if len(cyc) == 3:
                  #Get number of triangles.
                  if src in cyc and dst in cyc:
                      tri_count += 1
      
              if len(cyc) == 4:
                  if src in cyc and dst in cyc:
                      four_common += 1
                      
                  if src in cyc:
                      four_src += 1
                      
                  if dst in cyc:
                      four_dst += 1
          if be_verbose:                        
              print(f'SrcDegree {ds}, DstDegree: {dd}, Tricount: {tri_count}, Sq Common: {four_common}, Sq Src: {four_src}, Sq Dst: {four_dst}')        
          return ( (edge, 2/ds + 2/dd - 2 + tri_count/ max(ds, dd) + tri_count/ min(ds, dd) + four_common/max(ds,dd)*(four_src + four_dst) ) )
     
def calculate_curvature_from_edges(data, pickle_file):
    #Check for pickle file
    
    cm = CurveMatrix()
    #If exists read into object
        #Check if done. If so return
        
    #If not create a new obejct.
    
    #Determine the number of edges used.
    
    
    #Convert to a networkx graph
    g = to_networkx(data, to_undirected=True)
       
    #to_return is for ego-graphs
    to_return = []
    sim_list = []
    
    #curve_matrix probably needs to be a list of dictionaries. Index by centernode, keyed on pairs.
    curve_matrix = []
    #Build matrix for all ego graphs.
    ego_graph_storage = [] 
    curve_sums = []
    
    if node_list == None:   #Use entire graph
        print('c1urve_vectoring')
        curve_vect = calculate_curvature(g)        ###The return value of calc_curv needs to change.
        edge_scores = EdgeScoreOrganizer(curve_vect)
        curve_matrix.append(edge_scores)
        curve_sums.append(edge_scores.calc_sum())
        return curve_vect, curve_sums
    #Check parameters
    if type(node_list) == torch.Tensor:
        node_list = node_list.tolist()

def extract_ego_graph(edge_index, ego, hops):
    to_return = [[],[]]
    i = 0
    hop_stack = [ego]
    search_stack = []
    visited = []
    while i < hops:
        search_stack.extend(hop_stack)
        hop_stack = []
        while(len(search_stack) > 0):
            topic = search_stack[0]
            search_stack.pop(0)
            
            idxes = torch.cat( (torch.where(edge_index[0] == topic)[0], torch.where(edge_index[1] == topic)[0]) ).unique()
            neighbors = edge_index[:, idxes].unique()
            visited.append(topic)
            
            for x in neighbors:
                x = x.item()
                if not x in visited:
                    hop_stack.append(x)
                    
        # hop_stack = neighbors
        i += 1
    return torch.tensor(to_return)
    
def torch_calculate_curvature(G ):
    edge_list = G.edge_index
    degrees = degree(G.edge_index[0])   #Double check this.
    # ego_finder = RootedEgoNets(4)
    # egos = ego_finder.extract(G)
    
    for edge in zip(edge_list.t()):
        edge = edge[0]
        src, dst = edge[0].item(), edge[1].item()
        
        ds = degrees[src]
        dd = degrees[dst]
    
        if min(ds,dd).item() == 1:
            return 0
        else:
            ego = extract_ego_graph(G.edge_index, src, 4)  
            pass
    
##This version will alway returns an EdgeScoreOrganizer object. 
def better_calcuate_curvature(G, previous_scores = None, focus_edge=None, be_verbose=False, force_recalculate=False):
    def sub_function(G, edge):
        src, dst = edge
        
        #Check degrees.
        ds = max(G.degree(src), 1)
        dd = max(G.degree(dst), 1)
        if min(ds, dd) == 1:    #is this correct???
            return 0
        else:
            #Extract and merge ego_graphs to keep the runtime reasonable
            ego1 = nx.ego_graph(G, edge[0], radius=4)
            ego2 = nx.ego_graph(G, edge[1], radius=4)
            combo = nx.compose(ego1, ego2)
            #Get triangles
            #Get the cycles from the whole graph.
            
            # cycles1 = find_all_cycles(combo, source=edge[0], cycle_length_limit=4)
            # cycles2 = find_all_cycles(combo, source=edge[1], cycle_length_limit=4)
            cycles1_gen = nx.simple_cycles(combo, length_bound=4)
            # cycles2_gen = nx.simple_cycles(combo, length_bound=4)
            
            #Check for duplicates and reduce to cycles of 3 and 4 nodes.
            to_delete = [] 
            cycles1 = []
            for i, x in enumerate(cycles1_gen):
                l = len(x)
                if l == 3 or l == 4:
                    cycles1.append(sorted(x))
              
            # #Reverse the list
            # to_delete = sorted(to_delete, reverse=True)                    
            # for x in to_delete:
            #     del(cycles1[x])
             
            # cycles2 = []
            # to_delete = [] 
            # for i, x in enumerate(cycles2_gen):
            #     l = len(x)
            #     if l == 3 or l == 4:
            #         hold =  sorted(x) 
            #         if not hold in cycles1:
            #             cycles2.append(hold)
                
            # to_delete = sorted(to_delete, reverse=True)                    
            # for x in to_delete:
            #     del(cycles2[x])
                
                
            tri_count = 0
            
            four_src = 0
            four_dst = 0
            four_common = 0
            for cyc in cycles1:
                if len(cyc) == 3:
                    #Get number of triangles.
                    if src in cyc and dst in cyc:
                        tri_count += 1
        
                if len(cyc) == 4:
                    if src in cyc and dst in cyc:
                        four_common += 1
                        
                    if src in cyc:
                        four_src += 1
                        
                    if dst in cyc:
                        four_dst += 1
            if be_verbose:                        
                print(f'SrcDegree {ds}, DstDegree: {dd}, Tricount: {tri_count}, Sq Common: {four_common}, Sq Src: {four_src}, Sq Dst: {four_dst}')        
            return 2/ds + 2/dd - 2 + tri_count/ max(ds, dd) + tri_count/ min(ds, dd) + four_common/max(ds,dd)*(four_src + four_dst) 
        
   
    score_book = None
    score_book = EdgeScoreOrganizer()
    if not previous_scores == None:
        score_book.merge(previous_scores)    
    
    if focus_edge == None: 
        edge_list = list(G.edges)   
        for i, edge in enumerate(edge_list):
            print(f'Working on edge {edge} - {i} of {len(edge_list)}')
            if force_recalculate:
                score_book.add_score(edge, sub_function(G, edge))
            #Check for score
            if not score_book.edge_has_score(edge): 
                #Calculate new score. 
                
                # to_return.append( (edge, sub_function(G, edge)) )
                score_book.add_score(edge, sub_function(G, edge))
        
        
    else:
        if force_recalculate or not score_book.edge_has_score(focus_edge):
            score_book.add_score(focus_edge, sub_function(G, focus_edge))
        
    return score_book
        
#This needs to return a list of curvature values
def calculate_curvature(G, focus_edge=None, be_verbose=False):
    def sub_function(G, edge):
        src, dst = edge
        
        #Check degrees.
        ds = max(G.degree(src), 1)
        dd = max(G.degree(dst), 1)
        if min(ds, dd) == 1:    #is this correct???
            return 0
        else:
            #Extract and merge ego_graphs to keep the runtime reasonable
            ego1 = nx.ego_graph(G, edge[0], radius=4)
            ego2 = nx.ego_graph(G, edge[1], radius=4)
            combo = nx.compose(ego1, ego2)
            #Get triangles
            #Get the cycles from the whole graph.
            
            # cycles1 = find_all_cycles(combo, source=edge[0], cycle_length_limit=4)
            # cycles2 = find_all_cycles(combo, source=edge[1], cycle_length_limit=4)
            cycles1_gen = nx.simple_cycles(combo, length_bound=4)
            # cycles2_gen = nx.simple_cycles(combo, length_bound=4)
            
            #Check for duplicates and reduce to cycles of 3 and 4 nodes.
            to_delete = [] 
            cycles1 = []
            for i, x in enumerate(cycles1_gen):
                l = len(x)
                if l == 3 or l == 4:
                    cycles1.append(sorted(x))
              
            # #Reverse the list
            # to_delete = sorted(to_delete, reverse=True)                    
            # for x in to_delete:
            #     del(cycles1[x])
             
            # cycles2 = []
            # to_delete = [] 
            # for i, x in enumerate(cycles2_gen):
            #     l = len(x)
            #     if l == 3 or l == 4:
            #         hold =  sorted(x) 
            #         if not hold in cycles1:
            #             cycles2.append(hold)
                
            # to_delete = sorted(to_delete, reverse=True)                    
            # for x in to_delete:
            #     del(cycles2[x])
                
                
            tri_count = 0
            
            four_src = 0
            four_dst = 0
            four_common = 0
            for cyc in cycles1:
                if len(cyc) == 3:
                    #Get number of triangles.
                    if src in cyc and dst in cyc:
                        tri_count += 1
        
                if len(cyc) == 4:
                    if src in cyc and dst in cyc:
                        four_common += 1
                        
                    if src in cyc:
                        four_src += 1
                        
                    if dst in cyc:
                        four_dst += 1
            if be_verbose:                        
                print(f'SrcDegree {ds}, DstDegree: {dd}, Tricount: {tri_count}, Sq Common: {four_common}, Sq Src: {four_src}, Sq Dst: {four_dst}')        
            return 2/ds + 2/dd - 2 + tri_count/ max(ds, dd) + tri_count/ min(ds, dd) + four_common/max(ds,dd)*(four_src + four_dst) 
        
   
    # import multiprocessing
    # from multiprocessing import Pool, get_context, Process
    # from itertools import repeat
    
    # import threading.Thread as Thread
    if focus_edge == None: 
        
        to_return = []  #This will be the length of edge_list
        edge_list = list(G.edges)   
        end = len(edge_list)
        count = 0
        interval = int(end / 4)
        spacing = 4
        if end > 20:
            spacing = 20
            interval = int(end / 20)    
                
        for edge in edge_list:
            
      
            # if interval !=0 and count % interval == 0:
            #     print(f'{int(count/interval)} of {spacing}.')
            to_return.append( (edge, sub_function(G, edge)) )
            count += 1
        return to_return
        
        
    else:
    #Get prereq data
    # edge_list = list(G.edges)   #List of tuples
    # triangles = nx.triangles(G) #All trinagles indexed by node
    # cycles = nx.cycle_basis(G)  #All cycles in the graph.
    
    # tri_cycles = []
    # four_cycles = []
    #Reduce cycles to 4-cycles
    # for c in cycles:
    #     if len(c) == 3:
    #         tri_cycles.append(c)
    #     if len(c) == 4:
    #         four_cycles.append(c)
    
    # if focus_edge != None:
        return sub_function(G, focus_edge)            
   




    
#This needs to return a list of curvature values
def calculate_curvature_original(G, focus_edge=None):
    def sub_function(G, edge):
        src, dst = edge
        
        #Check degrees.
        ds = max(G.degree(src), 1)
        dd = max(G.degree(dst), 1)
        if min(ds, dd) == 1:    #is this correct???
            return 0
        else:
        #Get number of triangles.
            tri_count = 0 
            for cyc in tri_cycles:
                if src in cyc and dst in cyc:
                    tri_count += 1
                    
            four_src = 0
            four_dst = 0
            four_common = 0
            for cyc in four_cycles:
                if src in cyc and dst in cyc:
                    four_common += 1
                    
                if src in cyc:
                    four_src += 1
                    
                if dst in cyc:
                    four_dst += 1
            print(tri_count, four_common, four_src, four_dst)        
            return 2/ds + 2/dd - 2 + tri_count/ max(ds, dd) + tri_count/ min(ds, dd) + four_common/max(ds,dd)*(four_src + four_dst) 
    
    
    #Get prereq data
    edge_list = list(G.edges)   #List of tuples
    triangles = nx.triangles(G) #All trinagles indexed by node
    cycles = nx.cycle_basis(G)  #All cycles in the graph.
    
    tri_cycles = []
    four_cycles = []
    #Reduce cycles to 4-cycles
    for c in cycles:
        if len(c) == 3:
            tri_cycles.append(c)
        if len(c) == 4:
            four_cycles.append(c)
    
    if focus_edge != None:
        return sub_function(G, focus_edge)            
    to_return = []  #This will be the length of edge_list
    for edge in edge_list:
        # print(to_return)        
        to_return.append( (edge, sub_function(G, edge)) )
    
    return to_return

def modify_edges_in_dataset(data, ego, manipulations):
    full_graph_manipulation(data, manipulations.added_edges, manipulations.removed_edges)
       
#Making the assumption that something else can figure out the final organization.
#Returning a dictionary of node-ids and node representations.
def add_nodes_and_modify_edges_in_dataset(data, ego, manipulations, ego_center, ae):
    #Determine the number of nodes to synthesize. Compare largest node values to 
    original_node_count = data.x.shape[0] #Number of nodes
    current_nodes = list(sorted( ego.nodes() ), reverse=True )
    
    to_return = {}
    
    for x in current_nodes:
        if x < original_node_count:
            #No more nodes to add
            break
      
        #Otherwise generate an new node and put in the dictionary
        to_return[ego_center] = ae(data.x[ego_center])
        
    return to_return

'''
Accepts a list of manipuation objects, reduces duplications and removes any conflicts
'''
def combine_manipulations(manipulation_list):
    removals = []
    additions = []
    
    for m in manipulation_list:
        re = m.removed_edges
        
        d = {}
        for r in re:
            d[r] = 0
            
        removals += list(d.keys())
        
        ad = m.added_edges
        d = {}
        for a in ad:
            d[a] = 0
            
        additions += list(d.keys())
        
        #Now nodes
    #If in added and removed the remove.
    for i, r in enumerate(removals):
        if r in additions:
            #Remove from removals
            removals = removals[:i] + removals[i+1:]
            i_a = additions.index(r)
            additions = additions[:i_a] + additions[i_a+1:]
            
    return (removals, additions)

def full_graph_manipulation(data, additions, removals, node_dic):
    # print(f'\tREmoveal {len(removals)}')
    for removed in removals:
        
        #Find the index of removed
        #One direction
        src = (data.edge_index[0] == removed[0]).nonzero(as_tuple=True)[0]  #indexes where src is in 0th row.
        dst = (data.edge_index[1] == removed[1]).nonzero(as_tuple=True)[0]  #indexes where dst is in 1th row.
        
        #Find where they are the same.
        idx = None
        for s in src:
            if s in dst:
                idx = s.item()
                break
        # idx = (src == dst).nonzero(as_tuple=True)[0]
        # if idx.shape[0] > 1 or src == dst:
            # raise Exception("Shits broke.")
        # if idx.shape == 0:
            # raise Exception("Removed edge not found!!!!")
        
        #If there is a duplicate edge this will blow up.
        if not idx == None:
            data.edge_index = torch.stack( (torch.cat( (data.edge_index[0][:idx], data.edge_index[0][idx+1:]) ),
                                        torch.cat( (data.edge_index[1][:idx], data.edge_index[1][idx+1:]) ) )
                                      )
        #Then the other
        src = (data.edge_index[0] == removed[1]).nonzero(as_tuple=True)[0]
        dst = (data.edge_index[1] == removed[0]).nonzero(as_tuple=True)[0]
        
        #Find where they are the same.
        
        idx = None
        for s in src:
            if s in dst:
                idx = s.item()
                break
        if not idx == None:
            data.edge_index = torch.stack( (torch.cat( (data.edge_index[0][:idx], data.edge_index[0][idx+1:]) ),
                                            torch.cat( (data.edge_index[1][:idx], data.edge_index[1][idx+1:]) )) 
                                         )
    for added in additions:
        # print(f'Adding {added}, {added[1],added[0]}')
        data.edge_index = torch.stack( (torch.cat( (data.edge_index[0], torch.tensor(added)) ),
                                        torch.cat( (data.edge_index[1], torch.tensor( (added[1], added[0]) )) ))
                                     )
    #Add nodes
    if not node_dic == None:        #Node_dic is a dictionary of tuples --> ( <Node features>, <Class label> )
        keys = list(node_dic.keys())
        if not data.x.shape[0] == min(keys):
            raise Exception("Nodes are broke.") 
      
        tensor_list = []
        label_list = []
        for k in sorted(keys):
            # data.x = torch.cat(data.x, node_dic[k])
            tensor_list.append(node_dic[k][0])
            label_list.append(node_dic[k][1])
        data.x = torch.cat( (data.x, torch.stack( tuple(tensor_list) ) ) )
        data.y = torch.cat( (data.y, torch.tensor(label_list) ) )      
        
        data.few_shot_idx = torch.cat( (data.few_shot_idx, torch.tensor(keys)))
        ss = torch.tensor( [True for ssss in label_list])
        data.few_shot_mask = torch.cat( (data.few_shot_mask, ss) )
        
        ss = torch.tensor( [False for ssss in label_list])
        data.test_mask = torch.cat( (data.test_mask, ss) )
        data.val_mask = torch.cat( (data.val_mask, ss) )
    return data

def graph_analysis_func(data, data_as_graph, curve_scores, node_organizer, one_hop_list, psuedo_labeler, thresh = 2, visualize = False, homo_stats=None, can_add=True, can_remove=True):
    #Divide edges based on curvature
    add_list = []
    remove_list = []
    edges_added = 0
    edges_removed = 0
    decoder = {'added':[], 'removed':[]}
    value_stats = []
    
    #Find curve limit
    for value in curve_scores.score_dic.values():
        if value > 1:
            value_stats.append(value)
   
    value_stats = np.array(value_stats)
     
    mv = value_stats.mean() #sum(value_stats) / len(value_stats)
    std = value_stats.std()
    print(f'Mean curvature is {mv} + {std}')
     
    for key, value in curve_scores.score_dic.items():
        if value < 5:
            add_list.append(key)
        elif value > mv:
            # value_stats.append(value)
            remove_list.append(key)
        #Iterate through curve_scores.
   
    # exit()
    
    #Find homophily ratio around bottlenecks and curved edges.
    
    def local_homo_calc(lst):
        curve_homo_ratio = {}    
        for edge in lst:
            #Get 1 hop neighborhood from graph object.
            for nd in edge:
                if not nd in curve_homo_ratio: 
                    nhood = nx.ego_graph(data_as_graph, nd, 2)
                    
                    #Determine local homophily ratio.
                    target = data.y[nd].item()  #This is the truth for homophily.
                    homos = 0 
                    counter = 0
                    for nber in list(nhood.nodes()):
                        if data.y[nber].item() == target:
                            homos += 1
                        counter += 1
                    
                    curve_homo_ratio[nd] = (homos, homos / counter)     #Homo in 1-hop, ratio in 1-hop
        return curve_homo_ratio
    
    def edge_homo_calc(lst):
        edge_homo = []
        for edge in lst:
            src, dst = edge
            if data.y[src].item() == data.y[dst].item():
                edge_homo.append(1)
            else:
                edge_homo.append(0)
                
                
        homos = sum(edge_homo)
        count = len(edge_homo)
        del(edge_homo)
        return homos, homos/count
    
    add_homo_dic = local_homo_calc(add_list)
    remove_homo_dic = local_homo_calc(remove_list)
    to_add = []
    for k, v in add_homo_dic.items():
        to_add.append(v[1])
        
    to_rem = []
    for k, v in remove_homo_dic.items():
        to_rem.append(v[1])
        
       
    plt.hist(to_add)
    plt.title("Homo ratio or curved sections of the graph")
    plt.show()
    
    plt.hist(to_rem)
    plt.title("Homo ratio or bottleneced sections of the graph")
    plt.show()
    
    add_homo = edge_homo_calc(add_list)
    rem_homo = edge_homo_calc(remove_list)
    
    print(f'Add ratio {add_homo}, remove ratio {rem_homo}')
    exit
    mean_sim = node_organizer.get_median_score()
    stddev = node_organizer.get_stddev()
    sim_thresh = mean_sim
    endpos = len(add_list) + len(remove_list)
    compressed_list = []
    
def run_rewire(args, data, data_as_graph, curve_scores, node_organizer, one_hop_list, psuedo_labeler, thresh = 2, visualize = False, homo_stats=None, can_add=True, can_remove=True):
    #Divide edges based on curvature
    add_list = []
    remove_list = []
    edges_added = 0
    edges_removed = 0
    decoder = {'added':[], 'removed':[]}
    value_stats = []
    
    #Find curve limit
    for value in curve_scores.score_dic.values():
        if value > 1:
            value_stats.append(value)
   
    value_stats = np.array(value_stats)
     
    mv = value_stats.mean() #sum(value_stats) / len(value_stats)
    std = value_stats.std()
    print(f'Mean curvature is {mv} + {std}')
     
    for key, value in curve_scores.score_dic.items():
        if value < 5:
            add_list.append(key)
        elif value > mv:
            # value_stats.append(value)
            remove_list.append(key)
        #Iterate through curve_scores.
   
    # exit()
    
    if args.thresh_type == 'median':
        mean_sim = node_organizer.get_median_score()
    elif args.thresh_type == 'mean':
        mean_sim = node_organizer.get_mean_score()
        
    stddev = node_organizer.get_stddev()
    sim_thresh = mean_sim
    endpos = len(add_list) + len(remove_list)
    compressed_list = []
    for c in add_list:
        x, y = c
        compressed_list.append(x)
        compressed_list.append(y)
        
    compressed_list = list(set(compressed_list))
    # for ttt, e in enumerate(compressed_list):   #Add list needs to be reduced 
    #     if math.floor( ttt/endpos * 100) % 10 == 0:
    #         print(f'{ttt} of {endpos}, addlist is {len(add_list)} node is {e}')
    #     # mean_sim = node_organizer.get_median_score() + 2 * stddev
    #     # print(edge)
    #     #Will need this
    #     score_dic = psuedo_labeler.set_scores_about_a_node(e, data)
    #     for k , v in score_dic.items():
    #         # if  182 in k:
    #             # print("182 right here????")
    #         node_organizer.add_score( k,v )
    #     # for yyy, e in enumerate(edge):
    #     #     #Check organizer to see if the calculation has already been done.
    #     #     if not node_organizer.has_node_been_explored(e):
                
    #     #         # for n in range(data.x.shape[0]):
    #     #             # if not n == e:
    #     #                 #Calc cosinesim, add to dictionary. This is edge indexed.
    #     #         score_dic = psuedo_labeler.set_scores_about_a_node(e, data)
    #     #         for k , v in score_dic.items():
    #     #             node_organizer.add_score( k,v )
    #     #                 # node_organizer.add_score( (e, n), torch.nn.functional.cosine_similarity(data.x[e], data.x[n], dim=0).item())
    #     #                 # sim_scores[(e,n)] = torch.nn.functional.cosine_similarity(data.x[e], data.x[n], dim=0).item()
    #     #     ###End score update    
    #     #     #Need most similar edge.
            
    #     #     elif not node_organizer.has_node_been_full_explored(e, data):
    #     #         score_dic = psuedo_labeler.set_scores_about_a_node(e, data)
    #     #         for k , v in score_dic.items():
    #     #             node_organizer.add_score( k,v )    
    #     ##########
    #     # e1 = node_organizer.get_most_similar_node_to(edge[0])
    #     # e2 = node_organizer.get_most_similar_node_to(edge[1])
        
    #     # #Compare values
    #     # if e1[1] > e2[1]:
    #     #     candi`date_edge = (edge[0], e1[0])
    #     # else:
    #     #     candidate_edge = (edge[1], e2[0])
    #     # # candidate_edge = edge
        #############
    bn_breaks = 0
    ender = len(add_list)
    for ttt, edge in enumerate(add_list):
        if ttt % 100 == 0:
            print(f'{ttt} of {ender} in 2nd go.')
        if not node_organizer.has_node_been_explored(edge[0]):
            print(f'Exploring {edge[0]}')
            score_dic = psuedo_labeler.set_scores_about_a_node(edge[0], data)
            for k , v in score_dic.items():
                node_organizer.add_score( k,v )
        e1 = node_organizer.get_most_similar_node_to(edge[0])[0]
        
        if not node_organizer.has_node_been_explored(edge[1]):
            print(f'Exploring {edge[1]}')
            score_dic = psuedo_labeler.set_scores_about_a_node(edge[1], data)
            for k , v in score_dic.items():
                node_organizer.add_score( k,v )
        e2 = node_organizer.get_most_similar_node_to(edge[1])[0]
        try:
            bn_score = node_organizer.get_score(edge)
        except:
            print(f'{edge} not present in node score organizer.!!!!')        
            bn_score = psuedo_labeler.calculate_score_for_single_edges(edge[0], edge[1], data)
        # candidate_edge = (edge[0], e1[0])
        # #Compare values
        # if e1[1] > e2[1]:
        #     candidate_edges = [(edge[0], e1[0])]
        # else:
        #     candidate_edges = [(edge[1], e2[0])]
        should_break = False
        new_edge = False
        score_pair = []
        for candidate_edge in [(edge[0], e1), (edge[1], e2)]:
            # sim_thresh = r.uniform(mean_sim, 1)
            sim_thresh = mean_sim
            sim_measure = psuedo_labeler.calculate_score_for_single_edges(candidate_edge[0], candidate_edge[1], data)
            # score_pair.append(sim_measure)
            # print(f'Sim thresh {sim_thresh}')
            if bn_score < sim_measure:
                score_pair.append(True)
            else:
                score_pair.append(False)
            if sim_measure > sim_thresh:   #Curvature improvment. Edge is added.
                
                #This is the new curvature score.
                # improvement = calculate_curvature_experiment(data_as_graph, candidate_edge, edge)
    
                #Counter.
                # edges_added += 1
    
                if data_as_graph.has_edge(candidate_edge[0], candidate_edge[1]) or data_as_graph.has_edge(candidate_edge[1], candidate_edge[0]):
                    # raise Exception("Double edge add....")
                    continue
                #Add edge to the graph. 
                new_edge = True
                edges_added += 1
                data_as_graph.add_edge(candidate_edge[0], candidate_edge[1])
                decoder['added'].append( (candidate_edge[0], candidate_edge[1]) )
                #Add edge to the curve score 
                # curve_scores.add_score(candidate_edge, improvement)
                    
                # print(f'-----------Safety check Num edges currently::: {curve_scores.size()} ')
                # delta += abs(improvement - worst_score )
                if homo_stats != None:
                    #Check if homo
                    if data.y[candidate_edge[0]] == data.y[candidate_edge[1]]:
                        homo_stats.append(1)
                    else:
                        homo_stats.append(0)
             #Check for break
            
                 
        if score_pair[0] and score_pair[1]:
             #Remove bottleneck
             if data_as_graph.has_edge(edge[0], edge[1]):
                 if bn_score < sim_thresh:
                     data_as_graph.remove_edge(edge[0], edge[1])
                     bn_breaks += 1
        # edge, best_score = curve_scores.get_best_score()
    # pass
    # print("\t\t\tCannot Add Edges!!!!!!!!!!!!!!!!!!!!1") 
    print(f'Bottlenecks broken {bn_breaks}')
    bump = len(add_list)
    for ttt, edge in enumerate(remove_list):
        
        if math.floor( (ttt + bump)/endpos * 100) % 10 == 0:
            print(f'{ttt + bump} of {endpos}')
        #Will need this
        for yyy, e in enumerate(edge):
            #Check organizer to see if the calculation has already been done.
            if not node_organizer.has_node_been_explored(e):
                
                for n in range(data.x.shape[0]):
                    if not n == e:
                        #Calc cosinesim, add to dictionary. This is edge indexed.
                        node_organizer.add_score( (e, n), psuedo_labeler.calculate_score_for_single_edges(e, n, data) )
                        # node_organizer.add_score( (e, n), torch.nn.functional.cosine_similarity(data.x[e], data.x[n], dim=0).item())
                        # sim_scores[(e,n)] = torch.nn.functional.cosine_similarity(data.x[e], data.x[n], dim=0).item()
            ###End score update    
        
        #Do remove thing.
        if can_remove:
            sim_measure = psuedo_labeler.calculate_score_for_single_edges(edge[0], edge[1], data)
            # sim_thresh = r.uniform(0, .5  )
            
            sim_thresh = mean_sim #+ .2 * stddev
            # print(f'Removing sim measure {sim_measure}, threshold {sim_thresh}')
            if sim_measure < sim_thresh:
                # print(f'\t\t\tSim is reached {sim_thresh}. Removing....')
                if data_as_graph.degree(edge[0]) > 1 and data_as_graph.degree(edge[1]) > 1:
                    if data_as_graph.has_edge(edge[0], edge[1]):
                            
                            
                        data_as_graph.remove_edge(edge[0], edge[1])            
                        edges_removed += 1
                        decoder['removed'].append( (edge[0], edge[1]) )
                            # print('\t\tEdge removed!!')
                        #Remove Edge
                    # curve_scores.remove_edge_score(edge)
                    # delta += best_score
                else:
                        # print("$$$$$$$$$$$$$$EDGE NOT HERE$$$$$$$$")
                        #Remove Edge
                        # curve_scores.remove_edge_score(edge)
                        pass
                        # print("CANNOT REMOVE ++++ DEGREES ARE == 1")
            else:
                pass    
                #Adjust score.
                # print(f'\t\t\tCannot remove. Sim score too great.{sim_measure}, median { mean_sim} Decreasin SCORFE VALUE!!!!')
                    # curve_scores.add_score(edge, best_score /2)
        else:
                # print("REMOVAL LESS THAN THRESHOLD>>>>>")
            # pass
            print(f'-----------Safety check on removal Num edges currently::: {curve_scores.size()} ')
        if visualize:
            nx.draw(data_as_graph, node_size=20)
            plt.show()
        # print(len(curve_scores.score_dic), data_as_graph.size())
        # if len(curve_scores.score_dic) != data_as_graph.size():
            # Exception("Too many removed")
        
        done = False #curve_scores.get_worst_score()[1] > 0
        # return delta, edges_added, edges_removed,done, worst_score
    return edges_added, edges_removed , decoder, bn_breaks
        #Calculate change in curvature.            
        # return  manipulations_list, delta
    
#Change to accept the organizer object
#Changes data to graph object
def run_epoch(data, data_as_graph, curve_scores, node_organizer, one_hop_list, psuedo_labeler, thresh = 2, visualize = False, homo_stats=None, can_add=True, can_remove=True):
 
    #OBE - organizer object
    # curve_scores, ego_graph_list, curve_sums = calculate_curve_scores(data, data.few_shot_idx)
    # simplified_score = sum(curve_sums)
    delta = 0
    imp = 0
    edges_added = 0
    edges_removed = 0
    manipulations_list = []
    #Modify the current egographx
    edge, worst_score = curve_scores.get_worst_score()
    # edge, worst_score = curve_scores.get_range_of_worst_scores(num_options=2)
    # print(f'-----Worst edge {edge}, Score: {worst_score}')
    #Got the worst score and the edge associated.
    #For each node. Check similarity with all other nodes. 
    for yyy, e in enumerate(edge):
        #Check organizer to see if the calculation has already been done.
        if not node_organizer.has_node_been_explored(e):
            
            for n in range(data.x.shape[0]):
                if not n == e:
                    #Calc cosinesim, add to dictionary. This is edge indexed.
                    node_organizer.add_score( (e, n), psuedo_labeler.calculate_score_for_single_edges(e, n, data) )
                    # node_organizer.add_score( (e, n), torch.nn.functional.cosine_similarity(data.x[e], data.x[n], dim=0).item())
                    # sim_scores[(e,n)] = torch.nn.functional.cosine_similarity(data.x[e], data.x[n], dim=0).item()
    #Get the largets degree....
    largest_idx, other_idx = (0, 1) if edge[0] == min(data_as_graph.degree(edge[0]), data_as_graph.degree(edge[1])) else (1, 0)
    e = edge[largest_idx]
   
    def find_candidate_edge(G, node):
        
            #Get the most similar node
        neighbor_list = list(nx.ego_graph(G, node,  radius=1  ).nodes())
        
        #Remove 'e' from the list
        if e in neighbor_list:
            h = neighbor_list.index(e)
            neighbor_list = neighbor_list[:h] + neighbor_list[h+1:]
        #Check similarity from neighbor of other node.        
            
        # dst, sim_measure = node_organizer.get_most_similar_node_to(e)    #######THis needs to return the score
        if len(neighbor_list) == 0:
            # print("\n  No neighbors.")

            dst = None
        else:
            # print("((((List exists")
            #Need to make sure everything in the neighbor list is present in node organizer. 
            for q in neighbor_list:
                if not node_organizer.has_edge_been_explored( (e,q) ):
                    #Set a new score
                    scr = psuedo_labeler.calculate_score_for_single_edges(e, q, data)
                    node_organizer.add_score( (e,q), scr)
                    
            dst, sim_measure = node_organizer.get_most_similar_node_to_from_list(e, neighbor_list)
            while(  data_as_graph.has_edge(e, dst)):
            #Can't add an edge that already exists. Safety check here...
                if len(neighbor_list) == 0:
                    dst = None
                    print("###################################################LIST EXHAUSTED##############")
                    
 
                h = neighbor_list.index(dst)
                 
                del(neighbor_list[h])

                if len(neighbor_list) == 0:
                    # print("(((((      Empty list!!!!!!!!!!!!!")
                    new_dst = None
                    sim_measure = 0
                else:
                    # print("((((     Third condition")

                    new_dst, sim_measure = node_organizer.get_most_similar_node_to_from_list(e,  neighbor_list )
                if new_dst == dst:
                    raise Exception("Shits broke")
                dst = new_dst
        if dst == None:
            # print("((((                      No dest exists")

            candidate_edge = None
            sim_measure = 0.0
        else:
            candidate_edge = (e, dst)
        # print(f'Returning with a list of size {len(neighbor_list)}, dst is {candidate_edge}\n')
        return candidate_edge, sim_measure
          
    mean_sim = node_organizer.get_median_score()
    max_sim = node_organizer.get_max_similarity()
    if can_add:
        #This will return None if no edge is suitable.
        candidate_edge, sim_measure =  find_candidate_edge(data_as_graph, edge[other_idx])
        # print(f'\tSim mesaure {sim_measure}')
        #Check for failed condition.
        
        # sim_thresh = r.uniform(mean_sim, max_sim)
        sim_thresh = mean_sim
        if candidate_edge == None:  #This is a failure case...
            #Increase score
            # print("INCREASING SCORFE VALUE!!!!")
            curve_scores.add_score(edge, worst_score + 1)
        elif sim_measure < sim_thresh:  #Not adding an edge due to binomial distribution.
            # print(f'Not adding. Must be less than atleast {sim_measure}, got {sim_thresh}.... Increasing Score')
            # pass
            print("INCREASING SCORFE VALUE Thresh too low.!!!!")
            curve_scores.add_score(edge, worst_score + .2)
            
        else:   #Curvature improvment. Edge is added.
        
                #This is the new curvature score.
                improvement = calculate_curvature_experiment(data_as_graph, candidate_edge, edge)

                #Counter.
                edges_added += 1

                if data_as_graph.has_edge(candidate_edge[0], candidate_edge[1]) or data_as_graph.has_edge(candidate_edge[1], candidate_edge[0]):
                    raise Exception("Double edge add....")
                #Add edge to the graph.            
                data_as_graph.add_edge(candidate_edge[0], candidate_edge[1])
                    
                #Add edge to the curve score 
                curve_scores.add_score(candidate_edge, improvement)
                    
                print(f'-----------Safety check Num edges currently::: {curve_scores.size()} ')
                delta += abs(improvement - worst_score )
                if homo_stats != None:
                    #Check if homo
                    if data.y[candidate_edge[0]] == data.y[candidate_edge[1]]:
                        homo_stats.append(1)
                    else:
                        homo_stats.append(0)
             
        # edge, best_score = curve_scores.get_best_score()
    else:
        # pass
        print("\t\t\tCannot Add Edges!!!!!!!!!!!!!!!!!!!!1")
    if can_remove:
        edge, best_score = curve_scores.get_best_score()
        # print(f'Best score {best_score}')
        # edge, best_score = curve_scores.get_range_of_best_scores(num_options=1)  #Adds a stochastic element.
        if best_score > thresh: # and edges_added > 0:
            # print("\t\tAbove threshold. ")
            #Get cos sime for "edge"
            # sim_measure = torch.nn.functional.cosine_similarity(data.x[edge[0]], data.x[edge[1]], dim=0).item()
            sim_measure = psuedo_labeler.calculate_score_for_single_edges(edge[0], edge[1], data)
            #Random element
            # ran = r.random()
            # if sim_measure < ran:
            sim_thresh = r.uniform(0, mean_sim  )
            if sim_measure < sim_thresh:
                # print(f'\t\t\tSim is reached {sim_thresh}. Removing....')
                if data_as_graph.degree(edge[0]) > 1 and data_as_graph.degree(edge[1]) > 1:
                    if data_as_graph.has_edge(edge[0], edge[1]):
                        
                        
                        data_as_graph.remove_edge(edge[0], edge[1])            
                        edges_removed += 1
                        # print('\t\tEdge removed!!')
                    #Remove Edge
                    curve_scores.remove_edge_score(edge)
                    delta += best_score
                else:
                    print("$$$$$$$$$$$$$$EDGE NOT HERE$$$$$$$$")
                    #Remove Edge
                    curve_scores.remove_edge_score(edge)
                    pass
                    # print("CANNOT REMOVE ++++ DEGREES ARE == 1")
            else:
                #Adjust score.
                print(f'\t\t\tCannot remove. Sim score too great.{sim_measure}, median { mean_sim} Decreasin SCORFE VALUE!!!!')
                curve_scores.add_score(edge, best_score /2)
        else:
            # print("REMOVAL LESS THAN THRESHOLD>>>>>")
            pass
        print(f'-----------Safety check on removal Num edges currently::: {curve_scores.size()} ')
    else:
        pass
        # print("\t\t%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Cannot remove due to flag.") 
    if visualize:
        nx.draw(data_as_graph, node_size=20)
        plt.show()
    print(len(curve_scores.score_dic), data_as_graph.size())
    if len(curve_scores.score_dic) != data_as_graph.size():
        Exception("Too many removed")
    
    done = False #curve_scores.get_worst_score()[1] > 0
    return delta, edges_added, edges_removed,done, worst_score
   
    #Calculate change in curvature.            
    return  manipulations_list, delta


class OrganizerBase():
    def __init__(self):
        self.score_dic = {}
    
    ###
    # Places the components of a given edge in asscending order. This function 
    # performs some checks against invalid edge formats.
    # @Param: edge-Tuple of nodes IDs
    ###
    @staticmethod
    def order_edge(edge):
        if type(edge) != tuple:
            raise Exception("Edge not a tuple - EdgeScoreOrganizer-->order_edge")
        if len(edge) > 2:
            raise Exception("Edge tuple is too large - EdgeScoreOrganizer-->order_edge")
        if len(edge) < 2:
            raise Exception("Edge tuple is too small - EdgeScoreOrganizer-->order_edge")
            
        return ( min(edge), max(edge))
    
    #Prepares dictionary for json. Specifically converts keys to strings.
    def prep_for_JSON(self):
        to_return = {}
        for key, val in self.score_dic.items():
            to_return[ str(key) ] = val
            
        return to_return
    
    #This corrects the string-key issue used as a work-around for writing to JSON.
    def add_from_JSON_file(self, dic_from_JSON):
        # print(dic_from_JSON)
        for key, val in dic_from_JSON.items():
            self.score_dic[eval(key)] = val
        
class CurveMatrix():
    def __init__(self):
        self.score_list = []
        self.is_done = False
        self.last_node = -1 
        
    def __getitem__(self, arg):        
        return self.score_list[arg]
    
    def set_last_node(self):
        self.last_node = self.score_list[-1][0]
        
    def set_finished(self):
        self.is_done = True
        
    def num_edges_contained(self):
        return len(self.score_list)
    
    def check_for_edges(self, edge_list):
        not_present = []
        for e in edge_list:
            found = False
            for t, s in self.score_list:
                if e == t:
                    found = True
                    break
            if not found:
                not_present.append(e)
            
        return not_present
    
            
###
#This is a convenience class for keeping a collection of curvature scores organized.
###
class EdgeScoreOrganizer_orig(OrganizerBase):
    ###
    # Constructor.
    # @Param: initial_data- List of tuples containing and edge-score pair.
    def __init__(self, initial_data=None):
        super().__init__()
        if initial_data != None:
            for edge, score in initial_data:
                self.add_score(edge, score)

            
    def size(self):
        return len(self.score_dic)
    ###
    # Combines the current object with another EdgeScoreOrganizer
    ###
    def merge(self, outsider):
        if outsider != None:
            for edge in outsider:
                score = outsider.get_score(edge)
                self.add_score(edge, score)
            
    ###
    # Returns True if an edge exists in the object.
    ###
    def edge_has_score(self, edge):
        e = self.order_edge(edge)
        return e in self.score_dic.keys()
    
    ### 
    # Returns the score for a given edge. 
    # There is no protection against an edge not existing.
    # Edge is a tuple
    ###
    def get_score(self, edge):            
        #Put edge in proper form
        e = self.order_edge(edge)
        return self.score_dic[e]
    
    ###
    # Calculates the sum of the scores recorded.
    ###
    def calc_sum(self):
       sm = sum(self.score_dic.values())
       return sm
   
    ###
    # Returns the best score and the associated edge in the form (Edge, score)
    ###
    def get_best_score(self):
        best = max(self.score_dic.values())
        ret = [key for key in self.score_dic if self.score_dic[key] == best]
        return ( ret[0], self.score_dic[ret[0]] )
    
    ###
    # Returns the lowest score and the associated edge in the form (Edge, score)
    ###
    def get_worst_score(self, length = 1):
        best = min(self.score_dic.values())
        ret = [key for key in self.score_dic if self.score_dic[key] == best]
        return ( ret[0], self.score_dic[ret[0]] )
   
    def get_worst_score_from_set(self, node_set):
        subset = {}
        for x in node_set:
            x = self.order_edge(x)
            subset[x] = self.score_dic[x]
          
        best = min(subset.values())
        ret = [key for key in subset if subset[key] == best]
        return (ret[0], subset[ret[0]])
        
    def get_range_of_worst_scores(self, num_options=3):
        vals = sorted(self.score_dic.values(), reverse=False)[:num_options]
        ret_options = [key for key in self.score_dic if self.score_dic[key] in vals]
        selected_idx = r.randint(0, num_options-1)
        return (ret_options[selected_idx], self.score_dic[ret_options[selected_idx]])
      
    def get_range_of_best_scores(self, num_options=10):
        vals = sorted(self.score_dic.values(), reverse=True)[:num_options]
        ret_options = [key for key in self.score_dic if self.score_dic[key] in vals]
        selected_idx = r.randint(0, num_options-1)
        return (ret_options[selected_idx], self.score_dic[ret_options[selected_idx]])
      
        
    #Iterable.
    def __iter__(self):
        return iter(self.score_dic.keys())
    
    #Conveninece functions
    def size(self):
        return len(self.score_dic)
    
    ###
    # Adds an edge and a score to the dictionary. Edges may be reformated to prevent 
    # duplication.
    # This is a "destructive add". There is no check for existing data before assignment.
    # @Param: edge-Tuple of nodes IDs
    # @Param: score- Curvature score as a float.
    ###
    def add_score(self, edge, score):
        e = self.order_edge(edge)
        self.score_dic[e] = score

    def remove_edge_score(self, edge):
        e = self.order_edge(edge)
        del(self.score_dic[e])
        
    ###This method takes a list of objects and generates a single object/
    @staticmethod
    def merge_list_of_GraphModifiers(gm_list):
        to_return = EdgeScoreOrganizer()
        for x in gm_list:
            print(f'Status: total size {to_return.size()}, adding {x.size()}')
            for edge in x:
                #Check for duplicates i
                if edge in to_return.score_dic:
                    #Min is immaterial if scores are the same. 
                    to_return.add_score(edge, min(x.get_score(edge) , to_return.get_score(edge)))
                    if x.get_score(edge) != to_return.get_score(edge):
                        print(f'Problem with edge {edge}')
                        print(f'{x.get_score(edge)} V {to_return.get_score(edge)}')
                        # raise Exception("Edges Score Mismatch!!!")
                else:
                    to_return.add_score(edge, x.get_score(edge))
                
        return to_return
        
 

class NodeSimilarityOrganizer(OrganizerBase):
    def __init__(self):
        super().__init__()
        self.similar_index = 0 
        self.total_score = 0.0
        self.counter = 0
        
    def get_score(self, edge):
        if edge[0] in self.score_dic.keys():
            if edge[1] in self.score_dic[edge[0]].keys():
                
                return self.score_dic[edge[0]][edge[1]]
    
        raise Exception(f"Missing key in {edge}")            
        
    def has_node_been_explored(self, node):
        if node in self.score_dic.keys():
            return True
        
        return False
    
    def has_node_been_full_explored(self, node, data):
        dst_list = list(self.score_dic[node])
        
        if data.x.shape[0] - 1 > len(dst_list):
            return False
        else:
            return True
        
    def has_edge_been_explored(self, edge):
        # e = OrganizerBase.order_edge(edge)
        e = edge
        if self.has_node_been_explored(e[0]):
            if e[1] in self.score_dic[e[0]].keys():
                return True
        #Occurs if either if above fails. 
        return False
    
    #This returns the average similarity for homophilic edges and heterophilic edges.
    def calculate_homo_hetero_similarity(self, data):
        homo_list = []
        hetero_list = []
        for key in self.score_dic.keys():
            for subkey, val in self.score_dic[key].items():
                if data.y[key] == data.y[subkey]:
                    homo_list.append(val)
                else:
                    hetero_list.append(val)
                    
                    
        return homo_list, hetero_list
                    
                
    def is_score_present(self, src, dst):
        e = OrganizerBase.order_edge( (src, dst) )
        
        if self.has_node_been_explored(e[0]):
            active_dic = super().score_dic[e[0]]
            
            if e[1] in active_dic:
                return True
            
        return False
    
    ###
    # Adds an edge and a score to the dictionary. Edges may be reformated to prevent 
    # duplication.
    # @Param: edge-Tuple of nodes IDs
    # @Param: score- Curvature score as a float.
    ###
    def add_score(self, edge, score):
        # e = self.order_edge(edge)
        e = edge
        if not e[0] in self.score_dic:
            self.score_dic[e[0]] = {}
            
        self.score_dic[ e[0] ][e[1]] = score
        self.total_score += score
        self.counter += 1 
        
    def get_mean_score(self):
        return self.total_score / self.counter
        #Old approach
        total, counter = 0.0, 0
        for v in self.score_dic.values():
            for subv in v.values(): 
                total += subv
                counter += 1
                
        return total / counter
    #Returns a list of tuples (edge, score) ordered by score value.
    #Param: edge_list - list of edges, assumed to be real edges.
    def get_all_scores_ordered(self, edge_list=None):
        temp_dic = {}
        as_list = []

        #Flatten dictionary into one level.
        for k, v in self.score_dic.items():
         
                for subk, score in v.items():
                    if edge_list == None or (k, subk) in edge_list:
                        if k > subk:
                            continue
                        if type(score) == torch.tensor:
                            score = score.item()
                        temp_dic[(k, subk)] = score
        
        #Sort by value.
        sorted_list = [( k,v ) for k, v in sorted(temp_dic.items(), key=lambda item:item[1])]
        
        #Check against edge_list
        return sorted_list
    
    def get_median_score(self):
        as_list = []
        for v in self.score_dic.values():
            for subv in v.values(): 
                as_list.append(subv)
                
        as_list.sort()
        l = int(len(as_list ) * .3)
        return as_list[int( len(as_list)/2 ) + l]
    
    def get_stddev(self):
        mean = self.get_mean_score()
        running = 0.0
        counter = 0
        for v in self.score_dic.values():
            for subv in v.values(): 
                temp = subv - mean
                running += temp * temp
                counter += 1
                
        running = running / (counter - 1)
        return math.sqrt(running)
                
                
        
    def get_max_similarity(self):
        best = 0.0 
        for v in self.score_dic.values():
            if max(v.values())  > best:
                best = max(v.values()) 
        if type(best) == torch.tensor:
            best = best.item()
        return best
    
    def get_similarity_scores_from_a_list(self, base, list_of_nodes):
        if not base in self.score_dic.keys():
            raise Exception("Node issue in NodeSimilarityOrganizer")

        to_return = []
        for x in list_of_nodes:
            to_return.append( self.score_dic[ base ][x])

        return to_return 
    
    def get_most_similar_node_to(self, node, exception_list=None):
        if not node in self.score_dic.keys():
            raise Exception("Node issue in NodeSimilarityOrganizer")
        
        #Reset index
        self.similar_index = 0            
        active = self.score_dic[node]   #This is a dictionary.
        
        if exception_list != None:
            exception_nodes = []
            #This could be a list of nodes or a list of edges.
            data_type = type(exception_list[0])

            if data_type == int:
                #This is fine.
                exception_nodes = exception_list
            elif data_type == tuple:
                for tpl in exception_list:
                    exception_nodes.append(tpl[0])
                    exception_nodes.append(tpl[1])
                    
                exception_nodes = list(set(exception_nodes))
                
            elif data_type == torch.tensor:
                for t in exception_nodes:
                    exception_nodes.append(t.item())
                    
                exception_nodes = list(set(exception_nodes))
                
            score_copy = copy.deepcopy(active)
            
            for n in exception_nodes:
                del(score_copy[n])  
             
            active = score_copy
        
        best = max(active.values())
           
        ret = [key for key in active if active[key] == best]
        return ret[0], best
    
    def get_most_similar_node_to_from_list(self, node, lst):
        # print(f'got list {lst}\n-----------------------------------\n')
        # if not node in self.score_dic.keys():
        #     raise Exception("Node issue in NodeSimilarityOrganizer")

        #Reset index
        self.similar_index = 0  
        
        sub_set = []
        for l in lst:
            e = self.order_edge( (node, l) )
            active = self.score_dic[e[0]]
        
            sub_set.append( active[ e[1] ])
            
        if len(sub_set) == 0:
            return None, 0.0
        best = max(sub_set)
        ret = lst[ sub_set.index(best) ] #[key for key in active if active[key] == best]
        return ret, best
    
    
    #Used to walk down the order of node similarities.
    def get_next_most_similar_node_to(self, node):
        self.similar_index += 1 #Increase first.
        active = self.score_dic[node]
        sortd = sorted(active.values())
        best = sortd[self.similar_index]
        
        ret = [key for key in active if active[key] == best]
        return ret[0], best
    def get_top_ten_similar_nodes_to(self, node):
        if not node in self.score_dic.keys():
            raise Exception("Node issue in NodeSimilarityOrganizer")
    
        active = self.score_dic[node]
        sortd = sorted(active.values())
        sortd = sortd[:10]           
        
        ret = [key for key in active if active[key] in sortd]
        
        return ret
    
    
def calculate_homophily_ratio(data):
    src_labels = data.y[data.edge_index[0,:]]
    dst_labels = data.y[data.edge_index[1, :]]
    
    result = src_labels == dst_labels
    homo_count = sum(result)
    ratio = homo_count/ data.edge_index.shape[1]
    return ratio.item()

class EdgeScoreOrganizer(EdgeScoreOrganizer_orig):
    ###
    # Constructor.
    # @Param: initial_data- List of tuples containing and edge-score pair.
    def __init__(self, initial_data=None):
        super().__init__()
        self.total_score = 0.0
        self.counter = 0
        if initial_data != None:
            for edge, score in initial_data:
                self.add_score(edge, score)

            
    def size(self):
        return len(self.score_dic)
    ###
    # Combines the current object with another EdgeScoreOrganizer
    ###
    def merge(self, outsider):
         if outsider != None:   #This needs to be edge,score in outsider.
            for edge, score in outsider.score_dic.items():
                # score = outsider.get_score(edge)
                self.add_score(edge, score)
            
    ###
    # Returns True if an edge exists in the object.
    ###
    def edge_has_score(self, edge):
        e = self.order_edge(edge)
        if e in self.score_dic.keys():
            return True
        if (e[1], e[0]) in self.score_dic.keys():
            return True
        
        return False
    
    ### 
    # Returns the score for a given edge. 
    # There is no protection against an edge not existing.
    # Edge is a tuple
    ###
    def get_score(self, edge):
        raise Exception("get_score needs to be corrected.")            
        #Put edge in proper form
        # e = self.order_edge(edge)
        return self.score_dic[edge]
    
    ###
    # Adds an edge and a score to the dictionary. Edges may be reformated to prevent 
    # duplication.
    # @Param: edge-Tuple of nodes IDs
    # @Param: score- Curvature score as a float.
    ###
    def add_score(self, edge, score):
        # e = self.order_edge(edge)
        e = edge
        if not e[0] in self.score_dic:
            self.score_dic[e[0]] = {}
            
        self.score_dic[ e[0] ][e[1]] = score
        self.total_score += score
        self.counter += 1
    ###
    # Calculates the sum of the scores recorded.
    ###
    def calc_sum(self):
       sm = sum(self.score_dic.values())
       return sm
   
    ###
    # Returns the best score and the associated edge in the form (Edge, score)
    ###
    def get_best_score(self):
        best = max(self.score_dic.values())
        ret = [key for key in self.score_dic if self.score_dic[key] == best]
        return ( ret[0], self.score_dic[ret[0]] )
    