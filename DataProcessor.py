# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:21:42 2023

@author: ashmo
"""

from sklearn import preprocessing
import copy
import networkx as nx
import matplotlib as plt
import json
import os 
import math

###The DataProcessor class is used to convert a network capture into a graph.
###The DataProcessor object maintains a list of all nodes, links between nodes,
###the number of links between nodes, and the label of the node - benign or
###malicious.
class DataProcessor:
    #Constructor
    def __init__(self):
        self.graph_dictionary = {}
    #    self.transmitList = set()
        self.le = preprocessing.LabelEncoder()
       # self.encodings = None
        self.hasBeenEncoded = False
        self.isEncodingValid = False
        self.xx = 0
     
    #Checks for a node in the main dictionary.    
    def containsNode(self, nodeLabel):
        keys = self.graph_dictionary.keys()
        for k in keys:
            if k == nodeLabel:
                return True
        
        return False
    
    #Checks if a node is present in the master list. Node could be active or receive only.
   # def addNodeToMasterList(self, node):
    #    self.masterNodeList.add(node)
    
    def getAllNodes(self):
        return list(self.graph_dictionary.keys())
    
    #Returns all nodes in the graph
    ##NOTE: This is only src nodes currently, I don't like the name...
    def getSourceNodes(self):    
        return list(self.graph_dictionary.keys())
    
    def isDestinationOnly(self, label):
        hold = self.getDestOnlyNodes()
        #Is disjoint if it is a source node
        return (label in hold)
    
    #Return nodes that do not generate network traffic.
    # def getDestOnlyNodes(self):
    #     sourceNodes = set(self.graph_dictionary.keys())
    #     return self.masterNodeList.difference(sourceNodes)
        
    def getAllNodeData(self, label):
        #Probably needs a safety check.
        return self.graph_dictionary[label]
    
    #Returns the label of the node, 1 for malicous or 0 for benign
    def getNodeLabel(self, node):
        #Probably needs a safety check.

        return self.graph_dictionary[node][1]
    
    def getFeatures(self, node):
        return self.graph_dictionary[node][2]
      
    def isNodeABot(self, node):
        return (self.getNodeLabel(node) == 1)
    
    #Returns a dictionary of connected nodes and the number of connnections
    def getLinksAndOccurances(self, label):
        #Probably needs a safety check.
        return self.graph_dictionary[label][0]
    
    #Returns only the nodes that connect to the passed label.
    def getLinks(self, label):
        #Probably needs a safety check.
        # if self.isDestinationOnly(label):
        #     return []
        
        return list(self.getLinksAndOccurances(label).keys())
    
    #Returns a boolean is a link exists from the fromNode to the toNeighbor node
    def connectionExists(self, fromNode, toNeighbor):
        if self.containsNode(fromNode):
            #Get list of connectoins
            connections = self.getLinks(fromNode)   #Returns a dict_key object
            for c in connections:
                if c == toNeighbor:
                    return True

        #Failure case for all the if-statements                
        return False
        
    #Updates the internal list of nodes to add a new link 
    def addLinkToNode(self, node, destNode, classLabel, features):
        hold = self.getAllNodeData(node)
        dic = hold[0]
        dic[destNode] = 1
        feats = []
        inDegree = hold[3]
        outDegree = hold[4]
        for i in range(len(features)):
            feats.append(hold[2][i] + features[i])
        
        
        #Tuple, recreate
        hold = (dic, classLabel, feats, inDegree, outDegree)
        self.graph_dictionary[node] = hold
        
    #Updates the internal list of nodes to add an additional connection to an
    #existing link.        
    def updateExistingLink(self, fromNode, toNode, classLabel, features = None):        
        hold = self.getAllNodeData(fromNode)
        dic = hold[0]
        dic[toNode] += 1
        inDegree = hold[3]
        outDegree = hold[4]
        #This is a tuple and therefore must be recreated. Probably inefficent.
        
        # if(hold[1] == 1):
        #     print("FOUND EXISTING BOT")
            
        # if(classLabel == 1):
        #     print("Making a bot")
            
        labelUsed = max(hold[1], classLabel)
        if features is None:
            hold = (hold[0], labelUsed, hold[2], inDegree, outDegree)
        else:
            temp = []
            for i in range(len(features)):
                temp.append(hold[2][i] + features[i])
            
            feats = tuple(temp)
            hold = (hold[0], labelUsed, feats, inDegree, outDegree)
        #    print("Waiting")
        #Update    
        self.graph_dictionary[fromNode] = hold
        
     
        
    #Add accepts a list of data from the network capture. 
    #@param data - is a specialize list.
    def addNode(self, data):
        protoDecoder = { "tcp" : 0,
                     "udp" : 1,
                     "icmp" : 2
            }
        
        #Safety check. Some collections end abruptly. 
        if data[7 ] == data[8 ] and data[8] == data[10]:
            return 0
        #Process
        #1. Pull out IP addresses, check dictionary calculate stats
        #Correction - Build a graph from the src and destination
        srcIP = data[1].strip() #Label, stays string
        destIP = data[3].strip()    #Labels, stays string
        destPort = data[4].strip()  #Label, stays string
        proto = str(data[5]).strip().lower() #Used to organize the feature vector, needs encoded
        service = data[6].strip()   #Needs encoded
        durration = data[7].strip()
        # print(durration, " is type ", type(durration), " line Num ", self.xx)
        # self.xx += 1
        # if durration == '':
        #     durration = 0.0
        # else:
        durration = float(durration)
            
        srcBytes = float(data[8])
        destBytes = float(data[9])
        missedBytes = float(data[11])
        srcPackets = float(data[12])
        srcIPBytes = float(data[13])
        destPackets = float(data[14])
        destIPBytes = float(data[15])
        label = int(data[43])    #Determines if bot or not.

        #Add nodes to appropriate lists        
        # self.addNodeToMasterList(srcIP)
        # self.addNodeToMasterList(destIP)

        # features = [durration, srcBytes, dstBytes, float(data[11]), float(data[12]), 
        #             float(data[13]), float(data[14]), float(data[15])]
        #Features has 3 sub vectors. Each subvector has the form
        #<durration, srcBytes, destBytes, missedBytes, srcPackets, srcIPBytes, destPackets, destIPBytes>
        #The vectors are organized in the order
        #<TCP, UDP, ICMP>
        
        #Step 1 is create a source subvector.
        srcVector = [durration, srcBytes, srcPackets, srcIPBytes ]
        dstVector = [destBytes, destPackets, destIPBytes, missedBytes]
        
        #Step 2 finds the location of the subvector.
        positionOffset = protoDecoder[proto]
        
        #Step 3 create an empty feature vector
        srcVecLen = len(srcVector)
        dstVecLen = len(dstVector)
        srcFeatures = [0.0 for i in range(srcVecLen * len(protoDecoder.keys()))]
        dstFeatures = [0.0 for i in range(dstVecLen * len(protoDecoder.keys()))]
        
        #features = [0.0 for i in range(completeVectorLen * len(protoDecoder.keys()))]
        #features = srcFeatures + dstFeatures #This is a list of zeros.
        #completeVectorLen = len(srcVector) + len(dstVector)
        
        
        #Step 4 place subvectors in correct location
        for i in range(srcVecLen):
            # off = (subVecLen * positionOffset) + i
            # l = len(features)
            # print('I is {0},  ofset is {1}, feature Length is {2}'.format(i, off,  l) )
            srcFeatures[(srcVecLen * positionOffset) + i] = srcVector[i]
        
        for i in range(dstVecLen):
            dstFeatures[(dstVecLen * positionOffset) + i] = dstVector[i]

        #Concatenate lists with <src features, 0's> or <0's , dest features>
        srcFeaturesFull = srcFeatures +  [0.0 for i in range(dstVecLen * len(protoDecoder.keys()))]
        dstFeaturesFull = [0.0 for i in range(srcVecLen * len(protoDecoder.keys()))] + dstFeatures
        if not self.containsNode(srcIP): 
            #Create a new node 
            # num of occurances, Total src packets (12), average sorce packets
            #Dictionary holds touples of another dictionary and a label ==> { {dic}, label}
                                                 #Adjacency, label, node features, in-degree, out-degree
             self.graph_dictionary[srcIP] = ( { destIP : 1 }, label, srcFeaturesFull, 0, 0)
             self.isEncodingValid = False    
             #{destIP : 1, "label" : label}   #Dictionary of destinations and # of connections 
            
        else:   #The node has already been found, update stats
          #Check for existing connection  
          if self.connectionExists(srcIP, destIP): 
                
                self.updateExistingLink(srcIP, destIP, label, features = srcFeaturesFull)

          else:
                self.addLinkToNode(srcIP, destIP, label, srcFeaturesFull)
                self.isEncodingValid = False #This may not always be the case.

        #Repeat for destination IP
        if not self.containsNode(destIP):
            
            self.graph_dictionary[destIP] = ( { srcIP : 0 }, 0, dstFeaturesFull, 0, 0)
            self.isEncodingValid = False
            
        else:
            if self.connectionExists(destIP, srcIP):
                self.updateExistingLink(destIP, srcIP, 0, dstFeaturesFull)
                
            else:
                self.addLinkToNode(destIP, srcIP, 0, dstFeaturesFull)
                self.isEncodingValid = False
        
        #Update in and out degrees
        self.increaseOutDegree(srcIP)
        self.increaseInDegree(destIP)
        
    #A print method for debuggin purposes.
    def print(self, botsOnly = False):
        #Get a list of nodes
        nodes = self.getSourceNodes()
        
        #For each node get the connectections and print
        for node in nodes:
            if botsOnly:
                if self.isNodeABot(node):
                    transformed = self.le.transform([node])
                    print('Source: ', node, " Encoding - ", str(transformed), "Class Label - ", self.getNodeLabel(node))
                    
            else:
                print('Source ', node)
                destinations = self.getLinksAndOccurances(node) 
                for dest, count in destinations.items():
                    print("\t", dest, ":", count)
                
                #Now that connections are done print the class label
                print("Class Label - ", self.getNodeLabel(node))
                print("")
    
    def printBots(self):
        self.print(botsOnly = True)
        
        
    #Convert nodes into a numeric label
    #This converts node labels, string of IP addresses, to integer labels
    def encodeGraph(self):
        #Now uses all nodes
        self.le.fit( list(self.graph_dictionary.keys()) )
        self.hasBeenEncoded = True
        # print("Nodes", src_nodes)
        # print("Done", nums)l
    
    #Returns an IP address from a node encoding    
    def decodeNode(self, nodeEncoding):
        if self.hasBeenEncoded:
            return  self.le.inverse_transform([nodeEncoding])[0]
            #return #self.le.inverse_transform(t)
        return None
    
    def decodeNodes(self, nodeEncondings):
        if self.hasBeenEncoded:
            return self.le.inverse_transform(nodeEncondings)
        return None
    
    #Returns the number of source nodes
    def getNumNodes(self):
  #      return len(self.masterNodeList)
      return len(self.graph_dictionary.keys())
    
    def increaseInDegree(self, node):
        temp = self.graph_dictionary[node]
        self.graph_dictionary[node] = (temp[0], temp[1], temp[2], temp[3] + 1, temp[4])
        
    def increaseOutDegree(self, node):
        temp = self.graph_dictionary[node]
        self.graph_dictionary[node] = (temp[0], temp[1], temp[2], temp[3], temp[4] + 1)
    
    #Returns an encoded adjacency list. Leaving this as a numpy array
    #Every row equates to a node. Each entry/column in a row is a connecting node.
    ##Note I may be better off encoding everything....
    def makeAjacencyList(self):
        if not self.hasBeenEncoded: #Encode first
            self.encodeGraph()
        
        #Make list here.
        #Encodings are integers, use those. THe index is the implicit encoded label.
        sz = self.getNumNodes()
        out = []
        for i in range(sz):
            #Get original label. Must be a list.
            #Returns a Numpy array. Take the first element.
            lbl = self.le.inverse_transform([i])[0]
            
            #Get all connections to the node and make a deep copy
            temp = copy.deepcopy( self.getLinks( lbl ) )
       
            # print(temp)
            # print("---\n")
            #Temp contains the actual labels. Need to encode.
            #Append to output
            out.append(self.le.transform(temp))
       
        return out
    
    def convertToCOOFormat(self):
        adjMatrix = self.makeAjacencyList()
        #COO is a list of 3 lists. Row, columns, data or source, dest, data
        cooOutput = [ [], [], []]
        
        i = 0
        for destVector in adjMatrix:
            for destination in destVector:
                cooOutput[0].append(i)
                cooOutput[1].append(destination)
                cooOutput[2].append(1)
                
            i += 1
            
        return cooOutput
    
   
    def getGraphDiameter(self):
        
        ajList = self.makeAjacencyList()
        diam = 0
        
        #This builds a list of distances from a node to all others
        for ii in range(len(ajList)):    
        
          # create a queue for doing BFS
          q = [] #deque()
          #ii = 0
          # mark the source vertex as discovered
          discovered = [False for i in range(len(ajList))]
          discovered[ii] = True
       
          # enqueue source vertex
          q.append(ii)
          
          #This is the cost from the current node to the indexed node.  
          diameterCount = [0 for i in range(len(ajList))]
          # loop till queue is empty
          while len(q) > 0:
              
              # dequeue front node and print it
              v = q.pop() #q.popleft()
             #  print(v, end=' ')
              lowestCost = diameterCount[v]
           #   print("NOde ", v)
              # do for every edge (v, u)
              for u in ajList[v]:
      
                  if not discovered[u]:
                      # mark it as discovered and enqueue it
                      discovered[u] = True
                      diameterCount[u] = diameterCount[v] + 1 
                      q.append(u)
                  else:
                      #Check for lower costs
                      diameterCount[u] = min(diameterCount[u], diameterCount[v] + 1)
     
         #Need a final pass to make sure no nodes were left out...
#          print("DIAM COUNT BEFORE")
#          print(diameterCount)
          for i in range(len(ajList)):
                 #Look at every node again.
                 #DiamList is the best known score from starting node to node i.
                 
                 #For every destinatin
                 for dest in ajList[i]:
                     #Best current score is known best score
                     diameterCount[i] = min(diameterCount[i], diameterCount[dest] + 1)
                 pass
              
            #Save the biggest of the current round or the biggests of the previous rounds
          diam = max(max(diameterCount), diam)
         
 #       print("\n\nDIAM COUNT after")
  #      print(diameterCount)
   #     print("DIAMETER: ", diam) 
   
        return diam
    
    def getNumLinks(self):
        ajList = self.makeAjacencyList()
        count = 0
        for row in ajList:
            count += len(row)
            
        count = int(count/2)
        return count

    def getNumBots(self):
        botCount = 0
        for node in self.graph_dictionary.keys():
            label = self.graph_dictionary[node][1]
         #   print(type(label))
            if label == 1:
                botCount += 1
        
        return botCount
    
    def visualize(self, c = None):
        g = nx.Graph()
        ajList = self.makeAjacencyList()
        cmap = c
        nSize = [10 for i in range(len(ajList))]
        labels = {}
        for i in range(len(ajList)):
            if self.isNodeABot(self.decodeNode(i)):
                labels[i] = i
                nSize[i] = 50
            
            for j in range(len(ajList[i])):
                g.add_edge(i, ajList[i][j])
           
        pos = nx.spring_layout(g)
        nx.draw_spring(g, node_size=10, node_color=cmap, with_labels=False)
       
       # nx.draw_networkx_labels(g, pos, labels, font_size = 16)
        
    def visualizeWithColor(self):
        colorList = ["blue" for i in range(self.getNumNodes())]
        for node in range(len(self.graph_dictionary.keys())):
            color = "blue"
            if self.isNodeABot(self.decodeNode(node)):
                print("FOUND A BOT!!!!!!!!!!! ",node )
                colorList[node] = "red"
                               
           
            
        self.visualize(c = colorList)
    
    def getBots(self):    
        listOfBots = []
        for node in self.getAllNodes():
            if self.isNodeABot(node):
                listOfBots.append(node)
    
        return listOfBots
    
    def getBotAdjacencyList(self):
        listOfBots = self.getBots()
        toReturn = []
        #Now has a list of bots. Convert to an adjacency list
        for node in listOfBots:
            nodeCon = self.le.transform([node])[0]
            hold = self.getLinks(node)
            #Convert from a string IP address to an int
            for i in range(len(hold)):
                hold[i] = self.le.transform([hold[i]])[0]
            toReturn.append([nodeCon, hold])
            
            
        return toReturn
        
    def writeGraphToFile(self, graphIndex):
        pathToResults = os.getcwd() + "\\SavedGraphs"
        
        if not os.path.exists(pathToResults):
            print("Making Directory")
            os.mkdir(pathToResults)
            
        pathToResults = pathToResults + "\\dataset_" + str(graphIndex) + ".json"
       
        jsonObject = json.dumps(self.graph_dictionary, indent=4)
        fileObj = open(pathToResults, 'w')
        fileObj.write(jsonObject)
        fileObj.close()
        
    def readGraphFromFile(self, path = "", fileNum = 1):
        fileObj = None
        
        if path == "":
           pathToResults = os.getcwd() + "\\SavedGraphs\\Dataset_" + str(fileNum) + ".json"
        else:
           pathToResults = path
           
        with open(pathToResults, 'r') as fileObj:
           jsonObject = json.load(fileObj)
        
      #  print(jsonObject)
        self.graph_dictionary = jsonObject
        self.le = preprocessing.LabelEncoder()
       # self.encodings = None
        self.hasBeenEncoded = False
        self.isEncodingValid = False
        
    def getInDegree(self, node):
        return self.graph_dictionary[node][3]
    
    def getOutDegree(self, node):
        return self.graph_dictionary[node][4]
    
    def removeSilentNodes(self):
        
        silentList = []
        for node in self.graph_dictionary.keys():
             #Find all silent nodes
             if self.getOutDegree(node) == 0:
                 silentList.append(node)
        
        #Remove silent nodes
        dicCopy = self.graph_dictionary.copy()
        for node in silentList:
            del(dicCopy[node])  #Remove entries for silent nodes
       
        #Remove connections to silent nodes that have been removed. 
        #Get list of remaining nodes
        newDic = {}
        lastingNodes = list(dicCopy.keys())
        for node in lastingNodes:  
            targetTuple = dicCopy[node]
            targetConnections = targetTuple[0].copy()
            for val in dicCopy[node][0].keys(): #All destingation Nodes
                if val in silentList:
                   del(targetConnections[val])
                   
            newDic[node] = (targetConnections, targetTuple[1], targetTuple[2], targetTuple[3], targetTuple[4])
                       
        self.graph_dictionary = newDic
     
    #Calculates the L2 norm between the feature vectors of two nodes    
    def calcNodeDistance(self, n1, n2):
        feat1 = self.getFeatures(n1)
        feat2 = self.getFeatures(n2)
        
        sm = 0
        for i in range(len(feat1)):
            temp = feat2[i] - feat1[i]
            temp = temp * temp
            sm += temp

        return math.sqrt(sm) 
      

def genericJSONWritter(fileName, jText):
    #Create a new file
    fileObj = open(fileName, 'w')
    jsonObject = json.dumps(jText, indent=4)

    fileObj.write(jsonObject)
    fileObj.close()
    