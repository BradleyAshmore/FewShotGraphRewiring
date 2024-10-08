# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 14:34:30 2023

@author: ashmo
"""

import torch
from torch_geometric.data import Dataset, Data
from torch.nn.functional import normalize
import DataProcessor 
import csv
import random
import math
import os

def get_test_graph(data):
    return (data.x[data.test_mask], data.y[data.test_mask], data.test_edges)

def get_train_graph(data):
    return (data.x[data.train_mask], data.y[data.train_mask], data.train_edges)

class NetworkDataset(Dataset):
    def __init__(self, root="E:/DataSets", transform=None, pre_transform=None):
        super(NetworkDataset, self).__init__(root, transform, pre_transform)
        # self.file_loc = os.path.join(root, "toniot")
        
        # os.mkdirs(self.file_loc, exist_ok=True)
      #  self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        #Name of dataset
        #Open datasets into numpy arrays.
        # folderName = 'E:\\DataSets\\Processed_Network_dataset\\'
        fileNameBase = 'Network_dataset_'          #In the future this may iterate through files.
        fileSuffix = '.csv'
        toReturn = []
        raw_loc = os.path.join(self.root, "Processed_Network_dataset")
        for i in range(1, 10):
            toReturn.append(os.path.join(raw_loc, f'Network_dataset_{i}.csv'))
        # for i in range(23):
            # toReturn.append(folderName + fileNameBase + str(i + 1) + fileSuffix)
            
        # fileName = 'ExcerptDataset.csv'
        return toReturn
          # return ['some_file_1', 'some_file_2']
        # return 'E:\\DataSets\\Processed_Network_dataset\\ExcerptDataset.csv'

    @property
    def processed_file_names(self):
        return [os.path.join(self.root, 'ProcessedNetwork1.pt'),
                os.path.join(self.root, 'ProcessedNetwork2.pt'),
                os.path.join(self.root, 'ProcessedNetwork3.pt'),
                os.path.join(self.root, 'ProcessedNetwork4.pt'),
                os.path.join(self.root, 'ProcessedNetwork5.pt'),
                os.path.join(self.root, 'ProcessedNetwork6.pt'),
                os.path.join(self.root, 'ProcessedNetwork7.pt'),
                os.path.join(self.root, 'ProcessedNetwork8.pt'),
                os.path.join(self.root, 'ProcessedNetwork9.pt') ,
                os.path.join(self.root, 'ProcessedNetwork10.pt'),
                os.path.join(self.root, 'ProcessedNetwork11.pt'),
                os.path.join(self.root, 'ProcessedNetwork12.pt'),
                os.path.join(self.root, 'ProcessedNetwork13.pt'),
                os.path.join(self.root, 'ProcessedNetwork14.pt'),
                os.path.join(self.root, 'ProcessedNetwork15.pt'),
                os.path.join(self.root, 'ProcessedNetwork16.pt'),
                os.path.join(self.root, 'ProcessedNetwork17.pt'),
                os.path.join(self.root, 'ProcessedNetwork18.pt'),
                os.path.join(self.root, 'ProcessedNetwork19.pt'),
                os.path.join(self.root, 'ProcessedNetwork20.pt'),
                os.path.join(self.root, 'ProcessedNetwork21.pt'),
                os.path.join(self.root, 'ProcessedNetwork22.pt'),
                os.path.join(self.root, 'ProcessedNetwork23.pt')
]
#         return ['E:\\DataSets\\ProjectProcessed\\ProcessedNetwork1.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork2.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork3.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork4.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork5.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork6.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork7.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork8.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork9.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork10.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork11.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork12.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork13.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork14.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork15.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork16.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork17.pt',
#                 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork18.pt'
#                 # 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork19.pt',
#                 # 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork20.pt',
#                 # 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork21.pt',
#                 # 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork22.pt',
#                 # 'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork23.pt'
# ]


    def download(self):
        pass
        # Download to `self.raw_dir`.

    def process(self):
        print("Processing......")
        # Read data into huge `Data` list.
        node_features = []
        edge_features = []
        label = []

        #Create a dictionary to store node info
        graph = DataProcessor.DataProcessor()
        
        count = 0
        
        #Open file
        for openFile, writeFile in zip(self.raw_paths, self.processed_paths):
            with open(openFile) as x: #folderName + fileName, 'r') as x:
                reader_object = csv.reader(x, delimiter=",")
                ###NOTE: Columns are timestamp, src_ip, dst_ip, dst_port, protocol.
                ###I need an enum for these. Should be able to do if rom header info.
                header_info = next(reader_object)
                 
                #Read each row
                x = 1
                row = next(reader_object)
                while row: #Check if there is a row returned.
                    try:
                        graph.addNode(row)
                        count += 1
                        if count %1000 == 0:
                            print("On row ", str(count), " of file ", openFile)
                        #Get next row
                        # print("Reading row ", x)
                        # x += 1
                        row = next(reader_object)
                    except StopIteration:
                        #Remove variables
                        row = None
                
            graph.encodeGraph()             
            # graph.print()
            ajList = graph.makeAjacencyList()
    
            adj = []
            row = []
            col = []
            
            i = 0
            for destVector in ajList:
                for destination in destVector:
                    #cooOutput[0].append(i)
                    #adj.append(1)
                    row.append(i) #cooOutput[1].append(destination)
                    col.append(destination) # cooOutput[2].append(1)
                    
                    
                i += 1
          
            #Now have rows and columns of the matrix. Rows and columns become edge_index
            edge_index = torch.tensor( [row + col, col + row] , dtype=torch.long)
            
            
            #Add labels, add to data as Y
            size = graph.getNumNodes()
            labels = [0 for i in range(size)]
            nodeFeatures = [[] for i in range(size)]
            
            #Calculate degrees
            degrees = [0 for i in range(size)]
            
            for n in row + col:
             degrees[n] += 1
            
            #For every node in the row column look up IP address and get label and features
            #Need to use the encodings to ensure the order is preserved.  
            for encodedNode in range(size): #graph.decodeNodes(row):
                node = graph.decodeNode(encodedNode)
                hold = int(graph.getNodeLabel(node))
     
                labels[encodedNode] = hold
                nodeFeatures[encodedNode] = graph.getFeatures(node)
            
            #Create a tensor from the features and normalize
            featureTensor = torch.tensor(nodeFeatures)
            #normalizedFeatures = normalize(featureTensor, 1.0, dim=0)  #Note dim=0 is column wise normalization
        
            #Add node features, add to data as x
            data = Data(edge_index = torch.tensor(edge_index), x=featureTensor, y=torch.tensor(labels), degrees = torch.tensor(degrees))
            
            #Create training masks
            #Use labels as the number of nodes
            masterList = [i for i in range(len(labels))]
            trainMask = [False for i in range(len(labels))]
            testMask = [True for i in range(len(labels))]
            trainSplit = .8
            
            #Find bots
            bot_indexes = [i for i in range(len(labels)) if labels[i] == 1   ]
            numTrainBots = int(len(bot_indexes) * trainSplit)
            numTestBots = len(bot_indexes) - numTrainBots
            
            #Remover bots from masterList
            # for x in bot_indexes:
            #     masterList.remove(x)
                
            random.seed(12345)
            random.shuffle(masterList)
            # random.shuffle(bot_indexes)
            
            limit = math.floor(len(masterList) * .8)
            trainIdx = masterList[:limit]
            testIdx = masterList[limit:]
            
            for v in trainIdx:
                trainMask[v] = True
                testMask[v] = False
    
            #Test and train masks are set
            data.train_mask = torch.tensor(trainMask)    
            data.test_mask = torch.tensor(testMask)
            
            #Determine test, train, and removed edges
            test_edges = [[], []]
            train_edges = [[], []]
            conflict_edges = [[], []]
            
            for i in range(len(data.edge_index[0])):
                #Get src and destination
                src = data.edge_index[0][i]
                dst = data.edge_index[1][i]
                srcIsTest = src in testIdx
                dstIsTest = dst in trainIdx
                if srcIsTest and dstIsTest:
                    #Add to test edge
                    test_edges[0].append(src)
                    test_edges[1].append(dst)
                    
                elif not srcIsTest and not dstIsTest:
                    train_edges[0].append(src)
                    train_edges[1].append(dst)
                    
                else:
                    conflict_edges[0].append(src)
                    conflict_edges[1].append(dst)
        
            data.test_edges = torch.tensor(test_edges)
            data.train_edges = torch.tensor(train_edges)
            data.conflict_edges = torch.tensor(conflict_edges)
            
            torch.save(data, writeFile)
    
    def len(self):
        return len(self.processed_file_names)

    # def updateEdgeList(self, newList, graphIdx):
    #     self[graphIdx] = torch.tensor(newList)
        
    def get(self, idx):
        #osp.join(self.processed_dir), 
        return torch.load(self.processed_file_names[idx])        
    
    
class NetworkDatasetActiveOnly(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(NetworkDatasetActiveOnly, self).__init__(root, transform, pre_transform)
      #  self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        #Name of dataset
        #Open datasets into numpy arrays.
        folderName = 'E:\\DataSets\\Processed_Network_dataset\\'
        fileNameBase = 'Network_dataset_'          #In the future this may iterate through files.
        fileSuffix = '.csv'
        toReturn = []
        for i in range(8):
            toReturn.append(folderName + fileNameBase + str(i + 1) + fileSuffix)
            
        # fileName = 'ExcerptDataset.csv'
        return toReturn
        #  return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['E:\\DataSets\\ProjectProcessed\\ProcessedNetwork1.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork2.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork3.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork4.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork5.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork6.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork7.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork8.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork9.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork10.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork11.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork12.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork13.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork14.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork15.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork16.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork17.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork18.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork19.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork20.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork21.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork22.pt',
                'E:\\DataSets\\ProjectProcessed\\ProcessedNetwork23.pt'
]


    def download(self):
        pass
        # Download to `self.raw_dir`.

    def process(self):
        print("Processing......")
        # Read data into huge `Data` list.
        node_features = []
        edge_features = []
        label = []

        #Create a dictionary to store node info
        graph = DataProcessor.DataProcessor()
        
        #Open file
        for openFile, writeFile in zip(self.raw_paths, self.processed_paths):
            with open(openFile) as x: #folderName + fileName, 'r') as x:
                reader_object = csv.reader(x, delimiter=",")
                ###NOTE: Columns are timestamp, src_ip, dst_ip, dst_port, protocol.
                ###I need an enum for these. Should be able to do if rom header info.
                header_info = next(reader_object)
                 
                #Read each row
                x = 1
                row = next(reader_object)
                while row: #Check if there is a row returned.
                    try:
                        graph.addNode(row)
                        #Get next row
                        # print("Reading row ", x)
                        # x += 1
                        row = next(reader_object)
                    except StopIteration:
                        #Remove variables
                        row = None
            print("Removing silent nodes...")
            graph.removeSilentNodes()
            graph.encodeGraph() 
            print("Nodes remaining: " + str(graph.getNumNodes()))            
            # graph.print()
            ajList = graph.makeAjacencyList()
            total_degrees = []
            
            for edges in ajList:
                total_degrees.append(len(edges))
                
            #Calculate absolute degree here....
            adj = []
            row = []
            col = []
            
            i = 0
            for destVector in ajList:
                for destination in destVector:
                    #cooOutput[0].append(i)
                    #adj.append(1)
                    row.append(i) #cooOutput[1].append(destination)
                    col.append(destination) # cooOutput[2].append(1)
                    
                    
                i += 1
            
            #Now have rows and columns of the matrix. Rows and columns become edge_index
            edge_index = torch.tensor( [row, col] , dtype=torch.long)
            #Add labels, add to data as Y
            size = graph.getNumNodes()
            labels = [0 for i in range(size)]
            nodeFeatures = [[] for i in range(size)]
            
            #For every node in the row column look up IP address and get label and features
            #Need to use the encodings to ensure the order is preserved.  
            for encodedNode in range(size): #graph.decodeNodes(row):
                node = graph.decodeNode(encodedNode)
                hold = int(graph.getNodeLabel(node))
     
                labels[encodedNode] = hold
                nodeFeatures[encodedNode] = graph.getFeatures(node)
            
            #Create a tensor from the features and normalize
            featureTensor = torch.tensor(nodeFeatures)
            #normalizedFeatures = normalize(featureTensor, 1.0, dim=0)  #Note dim=0 is column wise normalization
        
            #Add node features, add to data as x
            data = Data(edge_index = torch.tensor(edge_index), x=featureTensor, y=torch.tensor(labels))
            
            #Create training masks
            #Use labels as the number of nodes
            masterList = [i for i in range(len(labels))]
            trainMask = [False for i in range(len(labels))]
            testMask = [True for i in range(len(labels))]
            
            random.seed(12345)
            random.shuffle(masterList)
            limit = math.floor(len(masterList) * .8)
            trainIdx = masterList[:limit]
            testIdx = masterList[limit:]
            
            for v in trainIdx:
                trainMask[v] = True
                testMask[v] = False
    
            data.train_mask = torch.tensor(trainMask)    
            data.test_mask = torch.tensor(testMask)
            data.node_degrees = torch.tensor(total_degrees)
            torch.save(data, writeFile)

    def len(self):
        return len(self.processed_file_names)

    # def updateEdgeList(self, newList, graphIdx):
    #     self[graphIdx] = torch.tensor(newList)
        
    def get(self, idx):
        #osp.join(self.processed_dir), 
        return torch.load(self.processed_file_names[idx])            