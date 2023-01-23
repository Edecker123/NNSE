import torch #first we import the library 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor,Lambda #first w
import matplotlib.pyplot as plt 
import numpy as np 
import os 
from torch import nn, softmax 
import torchvision as models
import time
import torch.fx
from torch.fx.node import Node
from typing import Dict
import tabulate
import torch
import torch.fx
from torch.fx.node import Node
import torchvision.models.resnet as res
from NNparse import targetLook, Node,loadArgs, loadKwargs, parseNet, getOps, oParseMods

class jacNode():
    def __init__(self,inp, func):
        self.inp=inp
        self.func=func
        self.child=[]
    
    def addChild(self, child):
        self.child.append(child)
        

def genAdjList(NN): #assume that NN is the hashmap gained from parseNet assume it is reversed
    adjL={}
    for i in NN: 
        if i in adjL:
            continue
        else: 
            adjL[i]=[]
            for j in NN[i].argnodes:
                try:
                    if j.name in NN:
                        adjL[i].append(j)
                except:
                    continue
                
    return adjL

def adjToNodes(Adj):
    for i in Adj: 
        for j in range(0, len(Adj[i])):
            value=Adj[i][j]
            node=jacNode(value, i)
            Adj[i][j]=node
            
                
def dagConnect(Adj): 
    adjToNodes(Adj)
    element=None
    for i in Adj: 
        for j in range(0, len(Adj[i])):
            for k in Adj[Adj[i][j].inp.name]: 
                Adj[i][j].addChild(k)
        element=i
    
    return element

def traverse(node,Adj):
    print(node)
    if len(node.child)==0 :
        return None
    else: 
        return traverse(node, Adj)
    
    

def pathFinder(Adj,k):
    dagConnect(Adj)
    #step 1 is grab root 
    traverse(k)
    
    