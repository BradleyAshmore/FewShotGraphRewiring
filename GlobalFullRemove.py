# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:45:30 2024

@author: ashmo
Global approach that evaluates all edges for removal.
"""
# Imports

import math
import argparse
import torch.nn.functional as F
import numpy as np
import time
import pickle
import json
import random
import torch
import sys
import os

from pathlib import Path
from FewShotModels import FNN #, NodeOnlyVariationalAutoEncoder, NodeAndStructureVariationalAutoEncoder
from FewShotUtilities import gcn_eval, load_cora, load_texas, load_citeseer, load_cornell, load_wisco, set_few_shot_labels, gcn_test, load_dataset_from_args
# from FewShotTrainingFunctions import train_autoencoder, train_edge_predictor, train_edge_predictor_for_homo

from ExperamentalScores import generate_similarity_score, train_mlp, SimilarityScoreCalculator
from ScoreModels import MLP
from GraphAnalysis import *

import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from torch_geometric.seed import seed_everything

def run_rewire_two(args, data, data_as_graph, curve_scores, node_organizer, one_hop_list, psuedo_labeler, thresh = 2, visualize = False, homo_stats=None, can_add=True, can_remove=True):
    #Divide edges based on curvature
    add_list = []
    remove_list = []
    edges_added = 0
    edges_removed = 0
    decoder = {'added':[], 'removed':[]}
    value_stats = []
    
    # #Find curve limit
    # for value in curve_scores.score_dic.values():
    #     if value > 1:
    #         value_stats.append(value)
    
    # value_stats = np.array(value_stats)
     
    # mv = value_stats.mean() #sum(value_stats) / len(value_stats)
    # std = value_stats.std()
    # print(f'Mean curvature is {mv} + {std}')
    print("Setting add and remove lists.")
    st = time.time() 
    for key, value in curve_scores.score_dic.items():
        print(f'k: {key}, v: {value}')
        if args.bottleneck_thresh > value:
        # if value < 100:
            add_list.append(key)
        if args.curve_thresh < value:
        # if  value > 500:
            # value_stats.append(value)
            remove_list.append(key)
            # pass
        #Iterate through curve_scores.
   
    # exit()
    ed = time.time()
    print(f'Time to assemble lists: {ed - st}')
   
    st = time.time()
    if args.thresh_type == 'median':
        mean_sim = node_organizer.get_median_score()
    elif args.thresh_type == 'mean':
        mean_sim = node_organizer.get_mean_score()
    
        
    stddev = node_organizer.get_stddev()
    ed = time.time()
    print(f'Time to to calc mean and std: {ed - st}')
   
    sim_thresh = mean_sim
    endpos = len(add_list) + len(remove_list)
    compressed_list = []    #Not used
   
    bn_breaks = 0
    ender = len(add_list)
    for ttt, edge in enumerate(add_list):
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
        
        should_break = False
        new_edge = False
        score_pair = []
        for candidate_edge in [(edge[0], e1), (edge[1], e2)]:

            sim_thresh = min(mean_sim * 4, 0.9)
            sim_measure = psuedo_labeler.calculate_score_for_single_edges(candidate_edge[0], candidate_edge[1], data)

            if bn_score < sim_measure:
                score_pair.append(True)
            else:
                score_pair.append(False)
            if sim_measure > sim_thresh:   #Curvature improvment. Edge is added.
            
                if data_as_graph.has_edge(candidate_edge[0], candidate_edge[1]) or data_as_graph.has_edge(candidate_edge[1], candidate_edge[0]):
                    # raise Exception("Double edge add....")
                    continue
                #Add edge to the graph. 
                new_edge = True
                edges_added += 1
                data_as_graph.add_edge(candidate_edge[0], candidate_edge[1])
                decoder['added'].append( (candidate_edge[0], candidate_edge[1]) )
                #Add edge to the curve score 
              
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
            if sim_measure < sim_thresh:
                if data_as_graph.degree(edge[0]) > 1 and data_as_graph.degree(edge[1]) > 1:
                    if data_as_graph.has_edge(edge[0], edge[1]):
                            
                            
                        data_as_graph.remove_edge(edge[0], edge[1])            
                        edges_removed += 1
                        decoder['removed'].append( (edge[0], edge[1]) )
                            # print('\t\tEdge removed!!')
                        #Remove Edge
                else:
                        pass
                        # print("CANNOT REMOVE ++++ DEGREES ARE == 1")
            else:
                pass    
   
        else:
                
            print(f'-----------Safety check on removal Num edges currently::: {curve_scores.size()} ')
        if visualize:
            nx.draw(data_as_graph, node_size=20)
            plt.show()

        
    return edges_added, edges_removed , decoder, bn_breaks
    

"""
Makes a filename for the results based on the arguments provided.
"""
def make_filename(args):
    folder = "./Results"
    if not os.path.exists(folder):
        os.mkdir(folder)

    folder += "/" + args.dataset
    if not os.path.exists(folder):
        os.mkdir(folder)

    folder += "/ShotSize_" + str(args.shot_size)
    if not os.path.exists(folder):
        os.mkdir(folder)

    if args.dataset.lower() in {'toniot'}:
        folder += "/Graph_Number_" + str(args.graph_num)
        if not os.path.exists(folder):
            os.mkdir(folder)
    
    folder += f"\\ResultsRecord_shot_size{args.shot_size}.csv"

    # args.dataset'-' + args.oversample_method + '-EdgesTo_' + args.add_edges_to +'-shot_size_' + str( args.shot_size) + '.csv'
    return folder


def set_plots_directory(args):
    folder = ".\\Plots-Global-deltaRepts"
    if not os.path.exists(folder):
        os.mkdir(folder)

    folder += "\\" + args.dataset
    if not os.path.exists(folder):
        os.mkdir(folder)

    folder += "\\ShotSize_" + str(args.shot_size)
    if not os.path.exists(folder):
        os.mkdir(folder)

    folder += "\\OversampleMultiplier_" + str(args.oversample_multiplier)
    if not os.path.exists(folder):
        os.mkdir(folder)

    folder += "\\"
    return folder

# Generate new nodes and associated labels.


# Check for autoencoder and edge predictor for the file name.


# Set args
#######################################################
# Parser prep
all_begin = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('-f')
parser.add_argument('--dataset', type=str, default="toniot")
parser.add_argument('--graph_num', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--bottleneck_thresh', type=float, default=1.0, help='The lower limit of accepted curvature scores. Anything below this value will be considered a bottleneck.')
parser.add_argument('--curve_thresh', type=float, default=100.0, help='The upper limit of accepted curvature scores. Anything above this value will be considered over-curved and will be considered for removal.')
parser.add_argument('--epoch_num', type=int, default=1000)
parser.add_argument('--edge_delta', type=float, default=.9)
# neighbor reconstruction loss weight
parser.add_argument('--oversample_method', default="ae_structure")
parser.add_argument('--oversample_multiplier', default=0)
parser.add_argument('--add_edges_to', default='sample')
parser.add_argument('--thresh_type', type=str, default="mean")
parser.add_argument('--repetitions', default=1)
parser.add_argument('--shot_size', default=7)
parser.add_argument('--threshold', default=6.0)
parser.add_argument('--verbose', default=False)
parser.add_argument('--return_result', type=bool, default=True)
parser.add_argument('--optimize', type=bool, default=True)

args = parser.parse_args()
args.device = device
seed_everything(42)
# Set args as appropriate datatypes
args.oversample_multiplier = int(args.oversample_multiplier)
args.repetitions = int(args.repetitions)
args.shot_size = int(args.shot_size)
args.threshold = float(args.threshold)
args.edge_delta = float(args.edge_delta)

def main_func(args):
    plot_folder = set_plots_directory(args)
    vanilla_results = []
    rewired_results = []
    rewired_precision = []
    rewired_recall = []
    
    add_list = []
    remove_list = []
    fn = make_filename(args)
    
    f = open(fn, 'a')
    f.write(f'\n\n\nThis ,is ,a ,new ,set ,of ,iterations,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,\nThresh method, {args.thresh_type}')
    f.close()
    
    if args.dataset == 'toniot':
        base_curve_score_file = f'./preprocessed_curve_scores/{args.dataset}_graph_{args.graph_num+1}_original_graph_curve_scores.json'
    else:
        base_curve_score_file = f'./preprocessed_curve_scores/{args.dataset}_original_graph_curve_scores.json'
    
    master_list = []
    master_devs = []
    
       
    f = open(fn, 'a')
    f.write(f'Edges Details, All or sampled:,{args.add_edges_to}, \n')
    f.write(f'Threshold: ,{args.threshold}, Oversample multiplier: , {args.oversample_multiplier}\n')
     # f.write(f'Mean:,{mean},StdDev:,{std},\n ')
    f.close()
    
    print("Performing classification on original graph.")
    seed_list = [random.randint(0, 9999999) for i in range(10)]
    data = load_dataset_from_args(args)
    
    
    mean, stddev, rep_results, precision_list, recall_list = gcn_eval(data, args=args, seed_list=seed_list, verbose=args.verbose, new_labels=False)
    vanilla_results.append( rep_results)
    print(f'Average F-1: {mean} +/- {stddev}') #', ,Precision:, {precision},Recall:,{recall}')
    og_score = mean
    
    
    print(vanilla_results)
    vanilla_precision = precision_list
    vanilla_recall = recall_list
    
    
    f = open(fn, 'a')
    f.write('\tClassification of original Graph: ,\n')
    # 
    van_master = []
    van_devs = []
    f.write("F-1 Score: ,")
    for i, row in enumerate(vanilla_results):
        for q in row:
            if type(q) == torch.Tensor: 
                f.write(str(q.item()) + " ,")
            else:
                f.write(str(q) + " ,")
        row = np.array(row)
        mean = row.mean()
        std = row.std()
        van_master.append(mean)
        van_devs.append(std)
        f.write(" ," + str(mean.item() *100) + ",+/-," + str(std.item() *100))  

    f.write("Precision: ,")
    temp = []
    # for i, row in enumerate(vanilla_precision):
    for q in vanilla_precision:
        if type(q) == torch.Tensor: 
            f.write(str(q.item()) + " ,")
        else:
            f.write(str(q) + " ,")
        temp.append(q)
    row = np.array(temp)
    mean = row.mean()
    std = row.std()
    van_master.append(mean)
    van_devs.append(std)
    f.write(" ," + str(mean.item() *100) + ",+/-," + str(std.item() *100)) 

    temp = []        
    f.write("Recall: ,")
    # for i, row in enumerate(vanilla_recall):
    for q in vanilla_recall:
        if type(q) == torch.Tensor: 
            f.write(str(q.item()) + " ,")
        else:
            f.write(str(q) + " ,")
        temp.append(q)
    row = np.array(temp)
    mean = row.mean()
    std = row.std()
    van_master.append(mean)
    van_devs.append(std)
    f.write(" ," + str(mean.item() *100) + ",+/-," + str(std.item() *100)) 
    
    f.write('  ,  \n')
    f.write('  ,  \nRewired ,\n')
    f.close()
    
    ae = None
    print("Classification on original graph done.")
     
    for rep in range(args.repetitions):
        result_list = []
        precision_list = []
        recall_list = []
        dev_list = []
        test_result_list = []
        master_list = []  
        master_devs = []
        #Reloading data
        data = load_dataset_from_args(args)
    
        print("Starting better version of curve calculation")
    
        if os.path.isfile(base_curve_score_file):
            curve_scores = EdgeScoreOrganizer()
            with open(base_curve_score_file) as jar:
                loaded_dic = json.load(jar)
                curve_scores.add_from_JSON_file(loaded_dic)
                # curve_scores.score_dic =
        else:
            print("No file found.")
            curve_scores = better_version_of_calculate_curve_scores(data)
            # print(curve_scores)
            print("Starting the pickle")
            as_json = json.dumps(curve_scores.prep_for_JSON())
            with open(base_curve_score_file, 'w') as jar:
                jar.write(as_json)
    
        # This is needed for sim scores.
        original_few_shot_mask = data.few_shot_mask.clone()
        print(f'Num edges before {data.edge_index[0].shape}')
    
    
        ##################### End of additions#################################
        print(f'\nRewiring Graph')
        graph_object = to_networkx(data, to_undirected=True)
        print('Setting adjacency.')
        data.edge_index = from_networkx(graph_object).edge_index
        psuedo_labeler = SimilarityScoreCalculator(args, data, data.few_shot_mask, epochs=20)
        
      
        print("Cycling through all nodes")
        start = time.time()
        all_curve_scores = curve_scores
        node_scores = NodeSimilarityOrganizer()
        node_scores = psuedo_labeler.calculate_score_for_all_pairs(data, node_scores)
        
        end = time.time()
        print(f'Elapsed: {end - start}')
      
    
        homo_measure = []
        starting_score = 10000.0
        delta_list = []
        curve_score_list = []
        one_hops = []
        complete_start = time.time()
        add_total = 0
        remove_total = 0
        total_edges = []
        edges_min = graph_object.size() * (1 - args.edge_delta)
        edges_max = graph_object.size() * (1 + args.edge_delta)
        strikes = 0
        can_add = True
        can_remove = True
        worst_scores = []
        graph_homo_pct_progress = []
        for epoch in range(1):
            # Take a homo percentage of the graph
            homo_pct = calculate_homophily_ratio(data)
            # graph_homo_pct_progress.append(homo_pct)
         ###########For visualization
            #Set colors
            # #Nodes
            # colorkey = ["red", "green", "blue", "black", "orange", 'purple', 'pink']
            # node_colors = ['gray' for i in range(data.x.shape[0])]
            # for i in data.few_shot_idx:
            #     node_colors[i] = colorkey[data.y[i]]
                
            # bottle_necks = []
            # curves = []
            # for key, value in all_curve_scores.score_dic.items():
            #     if value < 1.1:
            #         bottle_necks.append(key)
            #     else:
            #         curves.append(key)
                    
            start_time = time.time()
            # nx.draw(graph_object, node_size=20)
            # pos = nx.spring_layout(graph_object, seed=3113794652)  # positions for all nodes
            # pos = nx.spiral_layout(graph_object)
            
            
            # nx.draw_networkx_nodes(graph_object, pos, node_color=node_colors, node_size=20, alpha=.5)
           
            # nx.draw_networkx_edges(graph_object, pos, edgelist= bottle_necks, edge_color="tab:red")
            # nx.draw_networkx_edges(graph_object, pos, edgelist= curves, edge_color="tab:blue")
    
            # plt.show()
            
            
            # node_colors_truth = node_colors.copy()
            # for i, c in enumerate(data.y):
            #     node_colors_truth[i] = colorkey[c]
                
            # nx.draw_networkx_nodes(graph_object, pos, node_color=node_colors_truth, node_size=20, alpha=.5)
            # nx.draw_networkx_edges(graph_object, pos, edgelist= bottle_necks, edge_color="tab:red")
            # nx.draw_networkx_edges(graph_object, pos, edgelist= curves, edge_color="tab:blue")
    
            # plt.show()
       ################End visualization     
            added, removed, modified_edges, bns = run_rewire_two(args, data, graph_object, all_curve_scores, node_scores, one_hops, psuedo_labeler,
                                                        visualize=False, homo_stats=homo_measure, thresh=args.threshold, can_add=can_add, can_remove=can_remove)
            add_list.append(added)
            remove_list.append(removed)
            # nx.draw(graph_object, pos, node_color=node_colors, node_size=20)
            # plt.show()
            
            # nx.draw(graph_object,pos, node_color=node_colors_truth, node_size=20, alpha=.5)
            # plt.show()
            print(f'\nAdded {added} removed {removed}, Bottlenecks broken {bns}')
            # worst_scores.append(ws)
            added_stats = 0
            removed_stats = 0
            
            totr = len(modified_edges['removed'])
            tota = len(modified_edges['added'])
            tot = totr + tota
            
            for k, v in modified_edges.items():
                for edge in v:
                    src, dst = edge
                    if data.y[src] == data.y[dst]:
                        if k == 'added':
                            added_stats += 1 
                        else:
                            removed_stats += 1 
            tota = max(1, tota)    
            totr = max(1, totr)            
            print(f'Homo ratios Added {added_stats/tota:4f}, Removed: {removed_stats/totr:4f}\n')
            # add_total += added
            # add_list.append(add_total)
    
            remove_total += removed
            remove_list.append(remove_total)
            total_edges.append(graph_object.size())
          
            
            end_time = time.time()
            elapsed = end_time - start_time
    
            ending_score = all_curve_scores.calc_sum()
    
            score_change = ending_score - starting_score
            real_delta = starting_score - ending_score
            starting_score = ending_score
            delta_list.append(real_delta)
            # Only keep the adjacency information.
            data.edge_index = from_networkx(graph_object).edge_index
            homo_pct_after = calculate_homophily_ratio(data)
    
            print(f'\tEpoch: {epoch:4d} - edges = {data.edge_index.shape[1]} Edges Added: {added}, Edges removed: {removed} Time: {elapsed:4f}')
            print(f'\t\tHomophily outcome: Before: {homo_pct} After: {homo_pct_after}')
            curve_score_list.append(all_curve_scores.calc_sum())
    
    
            # Check for stop criteria
            edge_count = graph_object.size()
            if edge_count < edges_min:
                can_remove = False
            else:
                can_remove = True
            if edge_count > edges_max:
                can_add = False
            else:
                can_add = True
                # print("Stopping early due to # of edges exceding limit.")
                # break
        homo_pct_after = calculate_homophily_ratio(data)
    
        mean, stddev, rep_results, precision, recall = gcn_eval(data, args=args, seed_list=seed_list, verbose=args.verbose, new_labels=False)
        # gcn_result = gcn_eval(data, args=args, seed_list=seed_list, verbose=args.verbose, new_labels=True)
        rewired_results.append(rep_results)
        rewired_precision.append(precision)
        rewired_recall.append(recall)
        
        f = open(fn, 'a')
        f.write("F-1: ,")
        for q in rep_results:
            if type(q) == torch.Tensor:
                f.write(str(q.item()) + " ,")
            else:
                f.write(str(q) + ",")
                
            # row = np.array(row)
            # mean = row.mean()
            # std = row.std()
        master_list.append(mean)
        master_devs.append(std)
        f.write(" ," + str(mean.item()*100) + ",+/-," + str(std.item()*100))  
        added = add_list[i]
        removed = remove_list[i]
        f.write(', added ,' + str(added) + ',Removed ,' + str(removed) + ", HomoRatio Before," + str(homo_pct)+ ", Homo After, " + str(homo_pct_after) + ",  ,Homo ratio of Added," +str(added_stats/tota *100) + ", Homo percentage Removed: " + str(removed_stats/totr*100) +",\n")
        
        f.write('Precision: ,')
        temp = []
        for q in precision:
            if type(q) == torch.Tensor:
                f.write(str(q.item()) + " ,")
            else:
                f.write(str(q) + ",")
            temp.append(q)   
        temp = np.array(temp)
        mean = row.mean()
        std = row.std()
            # master_list.append(mean)
            # master_devs.append(std)
        f.write(" ," + str(mean.item()*100) + ",+/-," + str(std.item()*100))  
        added = add_list[i]
        removed = remove_list[i]
        f.write(",\n")
        
        f.write('Recall: ,')
        temp = []
        for q in recall:
            if type(q) == torch.Tensor:
                f.write(str(q.item()) + " ,")
            else:
                f.write(str(q) + ",")
            temp.append(q)   
        temp = np.array(temp)
        mean = row.mean()
        std = row.std()
            # master_list.append(mean)
            # master_devs.append(std)
        f.write(" ," + str(mean.item()*100) + ",+/-," + str(std.item()*100))  
        added = add_list[i]
        removed = remove_list[i]
        f.write('  ,  \n')
        
        f.close()
        
    # Record result for future processing.
        result_list.append(mean)
        dev_list.append(stddev)
        precision_list.append(precision)
        recall_list.append(recall)
    ##########
        if len(homo_measure) == 0:
            homo_measure = [0]
    print(f'\nPercent homo added {sum(homo_measure)/len(homo_measure)}')
    # print(f'\nHistory of removed edges {remove_list}')
    print(f'\n\nFinal results:\n\tOrigianal F-1: {og_score}')
    print("\n\n")
    print(result_list)

    print(f'Final Runtime: {time.time() - complete_start}\n\n')


    
    # f = open(fn, 'a')

   
    van_master = torch.tensor(van_master)
    van_devs = torch.tensor(van_devs)
    master_list = torch.tensor(master_list)
    master_devs = torch.tensor(master_devs)
    
    f = open(fn, 'a')
    f.write('\nIteration Results::::\n ,')
    f.write(f'Precision History: , {precision_list}\n')
    f.write(f'Recall History     , {recall_list}\n')
    f.write("\n Overall Averages:, ,\n")
    f.write("Vanilla F-1 Average:      ," + str(van_master.mean().item() *  100) + ", +/- ," + str(van_devs.mean().item()    * 100) + "\n")
    f.write("Rewire Grand F-1 Average: ," + str(master_list.mean().item() * 100) + ", +/- ," + str(master_devs.mean().item() * 100) + "\n")
     
    f.write('\n_____,_____,_____,_____,_____,_____,_____,\n\n')
    f.close()

    if args.return_result:
        return master_list.mean()
        
        
def blackbox(x, y):
    args.bottleneck_thresh = x
    args.curve_thresh = y
    args.return_result = True
    return main_func(args)
    # return -x ** 2/y+ 7*y + 6*x -4
if args.optimize:   #Overwrite arguments and iterate throught a set of values
    #This will iterate through the two threshold values.
    #X is bottleneck, Y is curve threshold
    for gn in {5}:
        for fff in { 5} :
            args.shot_size = fff
            pbounds = {'x': (-1, 10), 'y': (2, 300)}
        
            optimizer = BayesianOptimization(
                f=blackbox,
                pbounds=pbounds,
                random_state=1,
            )
            
            optimizer.maximize(
            init_points=4,
            n_iter=10,
            )
            f = open(f"./optimizationfile_graph_num{gn}_shot_size{fff}.txt", 'a')
            f.write(f'{args}\n\n')
            #print headers and iterations.
            f.write('{:11}\t,{:11}\t,{:11}\t, {:11}\n'.format('iteration',' target', ' bn', ' curve'))
            for c, it in enumerate(optimizer.res):
                f.write(f'{c:<10}\t, {it["target"]:<4.5f}\t, {it["params"]["x"]:4.5f}\t, {it["params"]["y"]:4.5f}\n' )
            f.write("\nBest combination:")
            f.write(f'-----\n{optimizer.max}\n----------\n\n\n')
            f.close()
else:
    main_func(args)

all_end = time.time()


print(f'Total runtime {all_end - all_begin}')