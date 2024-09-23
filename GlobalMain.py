# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 00:26:55 2024

@author: ashmo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:38:41 2024

@author: ashmo
This implements the global approach.
"""




# Imports

import math
import argparse
import torch.nn.functional as F
import numpy as np
import time
import pickle
import json

import torch
import sys
import os
import datetime as dt
from pathlib import Path
from FewShotModels import FNN, NodeOnlyVariationalAutoEncoder, NodeAndStructureVariationalAutoEncoder
from FewShotUtilities import gcn_eval, load_cora, load_texas, load_citeseer, load_cornell, load_wisco, set_few_shot_labels, gcn_test
from FewShotTrainingFunctions import train_autoencoder, train_edge_predictor, train_edge_predictor_for_homo

from experamentalscores import generate_similarity_score, train_mlp, SimilarityScoreCalculator

from ScoreModels import MLP

from GraphAnalysis import *
import matplotlib.pyplot as plt
sys.path.append("..")

"""
Makes a filename for the results based on the arguments provided.
"""


def make_filename(args):
    folder = ".\\Results-Sep21th"
    if not os.path.exists(folder):
        os.mkdir(folder)

    folder += "\\" + args.dataset
    if not os.path.exists(folder):
        os.mkdir(folder)

    folder += "\\ShotSize_" + str(args.shot_size)
    if not os.path.exists(folder):
        os.mkdir(folder)

    folder += "\\" + args.oversample_method
    if not os.path.exists(folder):
        os.mkdir(folder)

    folder += "\\ResultsRecord"
    time_string = str(dt.datetime.now())
    time_string = time_string[:time_string.index('.')]
    time_string = time_string.replace(':', '-')
    folder += "Thresh-" + str(args.thresh_type) + time_string + ".csv"
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


def oversample(args, data):
    if args.oversample_method == 'ae':
        # Load VAE
        ae_file, EP_name = get_pretrianed_model_names(args)
        ae = NodeOnlyVariationalAutoEncoder(
            data.x.shape[1], 64, 64, 14, None, device)
      #  save_file_name = ".\\ExperimentalVAE.mdl"
        ae.load_state_dict(torch.load(ae_file))

        oversampled, mu, sig = ae(
            data.x[data.few_shot_idx], data.edge_index, reparamiterize=True)
        oversampled = oversampled.ceil()
        oversampled_idx = data.few_shot_idx.clone()
    elif args.oversample_method == 'ae_structure':
        ae_file, EP_name = get_pretrianed_model_names(args)
        ae = NodeAndStructureVariationalAutoEncoder(
            data.x.shape[1], data.x.shape[0], 32, 2)
      #  save_file_name = ".\\ExperimentalVAE.mdl"
        ae.load_state_dict(torch.load(ae_file))

        oversampled, mu, sig = ae(data.x, data.edge_index, reparamiterize=True)
        oversampled = oversampled[data.few_shot_idx]
        # oversampled = oversampled.ceil()
        oversampled_idx = data.few_shot_idx.clone()
    elif args.oversample_method == 'oversample':
        oversampled = data.x[data.few_shot_idx].clone()
        oversampled_idx = data.few_shot_idx.clone()
    elif args.oversample_method == 'random':
        oversampled = torch.rand(data.x[data.few_shot_idx].shape)
        oversampled_idx = data.few_shot_idx.clone()
    elif args.oversample_method == 'zeros':
        oversampled = torch.zeros_like(data.x[data.few_shot_idx])
        oversampled_idx = data.few_shot_idx.clone()
    elif args.oversample_method == 'none':
        oversampled = torch.tensor([])
        oversampled_idx = torch.tensor([])
    else:
        raise Exception("Invalid Oversample method")

    return oversampled, oversampled_idx

# Check for autoencoder and edge predictor for the file name.


def get_pretrianed_model_names(args):
    if args.oversample_method == "ae_structure":
        ae_name = ".\\" + args.dataset.lower() + "-autoencoder_with_structure.mdl"
    else:
        ae_name = ".\\" + args.dataset.lower() + "-autoencoder.mdl"

    if args.homo_focus:
        edge_pred_name = ".\\" + args.dataset.lower() + "-homo-Magnitude-edgepredictor.mdl"
    else:
        edge_pred_name = ".\\" + args.dataset.lower() + "-edgepredictor.mdl"

    return ae_name, edge_pred_name


def load_dataset_from_args(args):
    if args.dataset.lower() == 'cora':
        # Make these 3 lines a function.
        data = load_cora(".\\data")
        data = data.to(device)
    elif args.dataset.lower() == 'texas':
        data = load_texas(".\\data")
        data = data.to(device)
        #Need to fix trainmasks.
        label_pool = torch.cat( (torch.where(data.train_mask)[0], torch.where(data.val_mask)[0]) )
        print(label_pool.shape)
        # Each iteration will have a different collection of labels.
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
    elif args.dataset.lower() == 'actor':
        data = load_actor(".\\data")
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
    # data.few_shot_mask = data.train_mask
    # data.few_shot_idx = torch.where(data.train_mask)[0]
    data.few_shot_idx, data.few_shot_mask = set_few_shot_labels(data, args, seed = 1234)
    print("Data loaded....")
    return data


"""
Update this....
@Params 
    oversampled - The newly generated nodes
    data - The data object that contains the original dataset
    edge_predictor - The model responsible for predicting edges
    oversampled_idx - The indexes of the nodes used for seed valuse. Also the
    indexes of nodes with labels.
    
"""
# Commenting out b/c I don't know that this is needed.


def oversampled_append(args, oversampled, data,  oversample_idx, threshold=.5):
    """
    This function evaluates nodes against the edge predictor.
    """
    def sub_function(x, y, j, new_edge_index, thresh):
        out = edge_predictor(x+y)
        probs = F.softmax(out)
        # print(probs)

        if probs[1] > thresh:
            # added += 1
            # new_edges.append([num_nodes + i,j])
            # Add to new list
            to_ret = torch.stack((torch.cat((new_edge_index[0], torch.tensor(num_nodes+i).unsqueeze(0))),
                                  torch.cat((new_edge_index[1], torch.tensor(j).unsqueeze(0)))))

            # And in reverse
            to_ret = torch.stack((torch.cat((to_ret[0], torch.tensor(j).unsqueeze(0))),
                                  torch.cat((to_ret[1], torch.tensor(num_nodes+i).unsqueeze(0)))))
            # cl = data.y[j].item()
            return to_ret, 1
        else:
            return new_edge_index, 0
    # Locate model files
    ae_file, EP_name = get_pretrianed_model_names(args)

    new_edge_index = torch.stack(
        (data.edge_index[0], data.edge_index[1]), dim=0)
    in_dim = data.x.shape[1]
    edge_predictor = FNN(in_dim, math.floor(in_dim / 6), 2, 1)
    # EP_name = ".\\BestAcc.mdl" #"HighFakeAccUse.mdl"
    num_nodes = data.x.shape[0]
    # ".\\EdgePredictor.mdl"
    edge_predictor.load_state_dict(torch.load(EP_name))

    # Compare against argument values.
    # Start with connection to labeled nodes\
    added = 0
    if args.pair_to_labeled:
        print("Adding edges to labeled nodes.")
        for i, x in enumerate(oversampled):
            # Connect to inspiration.
            new_edge_index = torch.stack((torch.cat((new_edge_index[0], torch.tensor(num_nodes+i).unsqueeze(0))),
                                          torch.cat((new_edge_index[1], torch.tensor(oversample_idx[i]).unsqueeze(0)))))

            # And in reverse
            new_edge_index = torch.stack((torch.cat((new_edge_index[0], torch.tensor(oversample_idx[i]).unsqueeze(0))),
                                          torch.cat((new_edge_index[1], torch.tensor(num_nodes+i).unsqueeze(0)))))

        added += len(oversampled)
        print("Oversampled nodes connected to labeled data.")

    #     #Option 1 - Brute force - Use data.x and check all nodes
    print("Adding edges.")
    for i, x in enumerate(oversampled):
        if args.add_edges_to == 'all':
            for j, y in enumerate(data.x):
                new_edge_index, adds = sub_function(
                    x, y, j, new_edge_index,  args.threshold)
                added += adds

        elif args.add_edges_to == 'sample':
            # Create a train/test split
            # Random indexes code
            # res = random.sample(range(0, data.x.shape[0]), 5)
            # res = torch.tensor(res)

            # Real edge code
            nv = oversample_idx[i]
            c = torch.where(data.edge_index[0] == nv)[0]
            candidates = data.edge_index[1][c]
            res = candidates
            for j in res:  # j is the node index
                # last = new_edge_index.shape[1]
                y = data.x[j]
                new_edge_index, adds = sub_function(
                    x, y, j, new_edge_index, args.threshold)
                added += adds
                if args.verbose:
                    print(f'Sampled Added edges: {added}')

            # Add new_edges to edge_index
        # Connect X to X.
    if args.pair_to_synthetic:
        for i, x in enumerate(oversampled):
            for j, y in enumerate(oversampled):  # j is the node index
                # j = num_nodes + j
                out = edge_predictor(torch.cat((x, y)))
                probs = F.softmax(out)
                # print(probs)
                # if probs.argmax().item() == 1:
                if probs[1] > threshold:
                    added += 1
                    # new_edges.append([num_nodes + i,j])
                    # Add to new list
                    new_edge_index = torch.stack((torch.cat((new_edge_index[0], torch.tensor(num_nodes+i).unsqueeze(0))),
                                                  torch.cat((new_edge_index[1], torch.tensor(num_nodes+j).unsqueeze(0)))))

                    # And in reverse
                    new_edge_index = torch.stack((torch.cat((new_edge_index[0], torch.tensor(num_nodes+j).unsqueeze(0))),
                                                  torch.cat((new_edge_index[1], torch.tensor(num_nodes+i).unsqueeze(0)))))
                    # cl = data.y[j].item()
                    # if cl in debug.keys():
                    #  debug[cl] += 1
                    # else:
                    #  debug[cl] = 1
            if args.verbose:
                print(f'Syn Added edges: {added}')

    # Add oversampled to X
    X_hat = torch.cat((data.x, oversampled), dim=0)

    # Labels for new X's
    if oversample_idx.shape[0] == 0:
        over_labels = torch.tensor([])
    else:
        over_labels = data.y[oversample_idx]

    new_labels = torch.cat((data.y, over_labels), dim=0)
    new_labels = new_labels.to(torch.long)
    new_train_mask = [True for x in range(len(oversampled))]
    new_train_mask = (
        torch.cat((data.few_shot_mask, torch.tensor(new_train_mask)), dim=0) == 1)
    new_test_mask = [False for x in range(len(oversampled))]
    new_test_mask = (
        torch.cat((data.test_mask, torch.tensor(new_test_mask)), dim=0) == 1)

    return X_hat, new_edge_index, over_labels, new_labels, new_train_mask, new_test_mask


# Set args
#######################################################
# Parser prep

# Options needed


# To Do
    #  XXX Return adds and subtracts from the rewiring epoch
    # Homophily study
    # Plot curvature
    #  XXX Add functionality for iterations.
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-f')
    parser.add_argument('--dataset', type=str, default="cornell")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--epoch_num', type=int, default=1000)
    parser.add_argument('--edge_delta', type=float, default=.9)
    parser.add_argument('--thresh_type', type=str, default='median')
    # neighbor reconstruction loss weight
    parser.add_argument('--oversample_method', default="ae_structure")
    parser.add_argument('--oversample_multiplier', default=0)
    parser.add_argument('--add_edges_to', default='sample')
    parser.add_argument('--pair_to_synthetic', default=True)
    parser.add_argument('--pair_to_labeled', default=True)
    parser.add_argument('--repetitions', default=3)
    parser.add_argument('--shot_size', default=1)
    parser.add_argument('--threshold', default=1.0)
    parser.add_argument('--verbose', default=False)
    parser.add_argument('--homo_focus', default=True)
    
    args = parser.parse_args()
    
    # Set args as appropriate datatypes
    args.oversample_multiplier = int(args.oversample_multiplier)
    args.pair_to_synthetic = args.pair_to_synthetic == "True"
    args.pair_to_labeled = args.pair_to_labeled == "True"
    args.repetitions = int(args.repetitions)
    args.shot_size = int(args.shot_size)
    args.threshold = float(args.threshold)
    args.edge_delta = float(args.edge_delta)
    args.thresh_type = args.thresh_type.lower()
    
    plot_folder = set_plots_directory(args)
    
    fn = make_filename(args)
    
    f = open(fn, 'a')
    f.write('\n\n\nThis ,is ,a ,new ,set ,of ,iterations,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,-,')
    f.close()
    
    base_curve_score_file = f'{args.dataset}_original_graph_curve_scores.json'
    master_list = []
    master_devs = []
    
       
    
    # print(f'Starting Repetition {rep + 1}\n----------\n')
    
    
     # Write header to disk.
     # fn = make_filename(args)
    
    f = open(fn, 'a')
    f.write(f'Edges Details, Labled Nodes, {args.pair_to_labeled}, All or sampled:,{args.add_edges_to}, Pair to syn:, {args.pair_to_synthetic}, \n')
    f.write(f'Threshold: ,{args.threshold}, Oversample multiplier: , {args.oversample_multiplier}\n')
     # f.write(f'Mean:,{mean},StdDev:,{std},\n ')
    f.close()
    
    print("Performing classification on original graph.")
    data = load_dataset_from_args(args)
    mean, stddev, ms = gcn_eval(data, args=args, verbose=args.verbose)
    og_score = mean
    # result = torch.tensor(0.0)
    print("Classification on original graph done.")
    # Write to file here....
    # test_result_list.append(result)
    f = open(fn, 'a')
    f.write('\tClassification of original Graph: ,')
    f.write(str(mean*100) + ",+/-," + str(stddev*100))
    f.write(',\n')
    f.close()
    ae = None
    
    f = open(fn, 'a')
    # f.write('\tClassification of oversampled Graph: ,')
    # f.write(str(mean) + ",+/-," + str(stddev))
    f.write(',\nRewired Accuracy ,')
    f.close()
     
    for rep in range(args.repetitions):
        result_list = []
        dev_list = []
        test_result_list = []
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
    
        # ############At this point the graph is oversampled.###################
        # spacers = torch.tensor( [False for i in range(data.few_shot_mask.shape[0] - original_few_shot_mask.shape[0])] )
        # original_few_shot_mask = torch.cat( (original_few_shot_mask, spacers) )
        # # full_mask = torch.tensor([True for x in range(data_mlp.x.shape[0])])
        # score_list = []
        # for i in range(10):
        #     #Train model to be able to generate psuedo labels.
        #     model = MLP(data.x.shape[1], data.y.max()+1)
    
        #     model, bs = train_mlp( args, model, data_mlp, full_mask, data)
        #     score_list.append(bs)
        #     # sim_scores = generate_similarity_score(data, data.few_shot_mask)
        #     # score_list.append(sim_scores)
    
        # ave1 = sum(score_list) / len(score_list)
    
        # score_list2 = []
        # for i in range(1):
        #     model = MLP(data.x.shape[1], data.y.max()+1)
    
        #     model, bs = train_mlp( args, model, data_mlp, full_mask, data)
        #     score_list2.append(bs)
    
        #     # sim_scores = generate_similarity_score(data, original_few_shot_mask)
        #     # score_list.append(sim_scores)
        # print(f'S1 {score_list}, S2 {score_list2}')
        # ave2 = sum(score_list2) / len(score_list2)
    
        # exit()
        # # print(f'\n\nOversampled nodes {ave1} vs. only original nodes {ave2} ')
        # #Calculate similarity score.
        # model = MLP(data.x.shape[1], data.y.max()+1)
    
        # model, bs = train_mlp( model, data, data.train_mask)
        # pred = model(data.x, sigmoid_output=True)
    #
    
        # Test autoencoder
        # #Generate new samples
        # encoder_samples = ae(data.x, None, reparamiterize=True)[0]
    
        # #Classify with trained model
        # preds = model(encoder_samples, sigmoid_output=True)
        # preds = preds.argmax(dim=1)
    
        # correct_predictions = preds == data.y
        # num_correct = sum(correct_predictions)
        # acc = num_correct / data.y.shape[0]
    
        # print(f'Autoencoder predictions num correct {num_correct}, acc: {acc}\n\n')
    
        # exit()
    
        ##################### End of additions#################################
        print(f'\nRewiring Graph')
        graph_object = to_networkx(data, to_undirected=True)
        print('Setting adjacency.')
        data.edge_index = from_networkx(graph_object).edge_index
        psuedo_labeler = SimilarityScoreCalculator(
            args, data, data.few_shot_mask, ae)
        
        # sim_scores = psuedo_labeler.set_scores_from_edges(data)
         
        all_curve_scores = curve_scores
        node_scores = NodeSimilarityOrganizer()
        #Removed for exhaustive setup.
        #
        # for k, v in sim_scores.items():
        #     if type(v) == torch.tensor:
        #         v = v.item()
        #     node_scores.add_score(k, v)
        num_nodes = data.x.shape[0]
        print("Cycling through all nodes")
                               ####Need to determine what is being calculated.
        calc_order = []
        start = time.time()
        for z in range(num_nodes):
            print(f'On Node {z}')
            start_row = time.time()  
            for a in range(num_nodes):
                # print(f'adding {z}, {a}')
                
                if z == a:
                    node_scores.add_score( (z, a), 0.0)
                    node_scores.add_score( (a, z), 0.0)
                elif z < a: 
                    break   #Break inner loop- values are already calculated.
                else:
                    sim_score = psuedo_labeler.calculate_score_for_single_edges(z, a, data)
                    node_scores.add_score((z,a), sim_score)
                    node_scores.add_score((a,z), sim_score)
                    
            end_row = time.time()
            calc_order.append( ( z, end_row - start_row) )
        end = time.time()
        print(f'Elapsed: {end - start}')
        
        val = node_scores.get_mean_score()
        meadian = node_scores.get_median_score()
        homo, hetero = node_scores.calculate_homo_hetero_similarity(data)
        print(f'Mean of scores {val}, median {meadian}')
        psuedo_labeler.print_match_stats(ratios=True)
        # exit()
        homo_measure = []
        starting_score = all_curve_scores.calc_sum()
        delta_list = []
        curve_score_list = []
        one_hops = []
        # for icount in range(data.x.shape[0]):
        #     one_hops.append( nx.ego_graph(graph_object, icount, radius=1, undirected=True) )
        complete_start = time.time()
        add_total = 0
        remove_total = 0
        add_list = []
        remove_list = []
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
         
            #Set colors
            #Nodes
            colorkey = ["red", "green", "blue", "black", "orange", 'purple', 'pink']
            node_colors = ['gray' for i in range(data.x.shape[0])]
            for i in data.few_shot_idx:
                if type(i) == torch.tensor:
                    i = i.item()
                node_colors[i] = colorkey[data.y[i]]
                
            bottle_necks = []
            curves = []
            for key, value in all_curve_scores.score_dic.items():
                if value < 1.1:
                    bottle_necks.append(key)
                else:
                    curves.append(key)
                    
            start_time = time.time()
            # nx.draw(graph_object, node_size=20)
            pos = nx.spring_layout(graph_object, seed=3113794652)  # positions for all nodes
            # pos = nx.spiral_layout(graph_object)
            
            
            # nx.draw_networkx_nodes(graph_object, pos, node_color=node_colors, node_size=20, alpha=.5)
           
            # nx.draw_networkx_edges(graph_object, pos, edgelist= bottle_necks, edge_color="tab:red")
            # nx.draw_networkx_edges(graph_object, pos, edgelist= curves, edge_color="tab:blue")
    
            # plt.show()
            
            
            node_colors_truth = node_colors.copy()
            for i, c in enumerate(data.y):
                node_colors_truth[i] = colorkey[c]
                
            # nx.draw_networkx_nodes(graph_object, pos, node_color=node_colors_truth, node_size=20, alpha=.5)
            # nx.draw_networkx_edges(graph_object, pos, edgelist= bottle_necks, edge_color="tab:red")
            # nx.draw_networkx_edges(graph_object, pos, edgelist= curves, edge_color="tab:blue")
    
            # plt.show()
            
            added, removed, modified_edges, bns = run_rewire(args, data, graph_object, all_curve_scores, node_scores, one_hops, psuedo_labeler,
                                                        visualize=False, homo_stats=homo_measure, thresh=args.threshold, can_add=can_add, can_remove=can_remove)
            
            # graph_analysis_func(data, graph_object, all_curve_scores, node_scores, one_hops, psuedo_labeler,
            #                                             visualize=False, homo_stats=homo_measure, thresh=args.threshold, can_add=can_add, can_remove=can_remove)
            
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
            add_total += added
            add_list.append(add_total)
    
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
    
        mean, stddev, ms = gcn_eval(data, args=args, verbose=args.verbose)
        f = open(fn, 'a')
        # f.write('')
        f.write(str(mean*100) + ",+/-," + str(stddev*100))  # f.write(str(result.item()) )
        f.write(', added ,' + str(added) + ',Removed ,' + str(removed) + ", HomoRatio Before," + str(homo_pct)+ ", Homo After, " + str(homo_pct_after) + ",  ,Homo ratio of Added," +str(added_stats/tota *100) + ", Homo percentage Removed: " + str(removed_stats/totr*100) +",\nRewired Accuracy,")
        
    
        f.close()
        # Record result for future processing.
        result_list.append(mean)
        dev_list.append(stddev)
    
        ##########
        if len(homo_measure) == 0:
            homo_measure = [0]
        print(f'\nPercent homo added {sum(homo_measure)/len(homo_measure)}')
        # print(f'\nHistory of removed edges {remove_list}')
        print(f'\n\nFinal results:\n\tOrigianal Acc: {og_score}')
        print("\n\n")
        print(result_list)
    
        print(f'Final Runtime: {time.time() - complete_start}\n\n')
    
    
        image_name_base = "Dataset-" + str(args.dataset) + "--Shot-" + str(
            args.shot_size) + "--Mult-" + str(args.oversample_multiplier) + "Rep-" + str(rep)
        # plt.plot(worst_scores)
        # plt.title("Worst scores by epoch.")
        # plt.show()
    
        plt.plot(result_list)
        plt.title("Model performance after oversample and rewire")
        plt.savefig(plot_folder + image_name_base + "-Acc.png")
        # plt.show()
        plt.clf()
    
        # plt.plot(test_result_list)
        # plt.title("Model performance after oversample and rewire on original graph")
        # plt.savefig(".\\Plots\\" + image_name_base +"-AccOrig.png")
        # # plt.show()
        # plt.clf()
    
        plt.plot(delta_list)
        plt.title("Curvature Deltas by Epoch")
        plt.savefig(plot_folder + image_name_base + "-Delta.png")
        # plt.show()
        plt.clf()
    
        plt.plot(curve_score_list)
        plt.title("Curvateur by Epoch")
        plt.savefig(plot_folder + image_name_base + "-Curvature.png")
        # plt.show()
        plt.clf()
    
        # print(f'Removal progress \n{remove_list}')
        plt.plot(add_list, label='Adds')
        plt.plot(remove_list, label='Removals')
        plt.plot(total_edges, label='Total Edges')
        plt.legend()
        plt.title("Additions and Removals by Epoch")
        plt.savefig(plot_folder + image_name_base + "-AdditionsAndRemovals.png")
        # plt.show()
        plt.clf()
    
        plt.plot(graph_homo_pct_progress, label='Homophily Ratio of Graph')
        # plt.plot(remove_list, label='Removals')
        # plt.plot(total_edges, label='Total Edges')
        plt.legend()
        plt.title("Homophily Ratio of Graph by Epoch")
        plt.savefig(plot_folder + image_name_base +
                    "-GraphHomophilyPercentage.png")
        # plt.show()
        plt.clf()
        # Plot modifications as a percentage of total edges.
        mod_percent = []
        for i in range(len(add_list)):
            mod_percent.append((add_list[i] + remove_list[i]) / total_edges[i])
    
        # print(f'Mod percent list\n\t{mod_percent}')
        plt.plot(mod_percent)
        plt.title("Edge Modifications As A Total Of All Edges.")
        plt.savefig(plot_folder + image_name_base + "-PercentageOfEdges.png")
        plt.clf()
    
        # Calculate homo-progression
        homo_progress = []
        sm = 0
        for i, x in enumerate(homo_measure):
            sm += x
            homo_progress.append(sm / (i+1))
    
        plt.plot(homo_progress)
        plt.title("Percentage of homoedges predicted as a function of predictions")
        plt.savefig(plot_folder + image_name_base + "-HomoProgress.png")
        # plt.show()
        plt.clf()
    
        master_list.append(result_list.copy())
        master_devs.append(dev_list.copy())
        del (data)
        del (graph_object)
    
    master_list = torch.tensor(master_list)
    master_devs = torch.tensor(master_devs)
    
    f = open(fn, 'a')
    f.write('\nIteration Results::::\n ,')
    for i in range(master_list.shape[0]):
        f.write(str(master_list[i].mean().item()) +
                ",+/-," + str(master_devs[i].mean().item()) + ",")
        
    f.write("\n OVerall Averate:, ,")
    f.write(str(master_list.mean().item()) + ", +/- ," + str(master_devs.mean().item()))
     
    f.write('\n_____,_____,_____,_____,_____,_____,_____,\n\n')
    f.close()
    
    f.close()
