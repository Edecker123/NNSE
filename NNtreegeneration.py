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
from Fcount import GEMMflops
class jacNode():
    def __init__(self,inp, func):
        self.inp=inp
        self.func=func
        self.name=[] #max length of 2
        self.child=[]
        self.operation=None
        self.visited=False
    def addChild(self, child):
        self.child.append(child)
        

def sizeofJac(NN, nodename): 
    #size of jac is raw outputs by raw inputs 
    output=NN[nodename[1].name].result
    size=1
    for i in output.shape: 
        size=size*i

    input=NN[nodename[0]].result
    isize=1
    for i in input.shape:
        isize=isize*i
    
    return [size,isize]
    

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

def adjToNodes(Adj,NN):
    for i in Adj: 
        for j in range(0, len(Adj[i])):
            value=Adj[i][j]
            node=jacNode(value, i)
            node.name=[i,value]
            # node.operation=NN[value.name].operation #this is the child operations
            Adj[i][j]=node

            
                
def dagConnect(Adj,NN): 
    adjToNodes(Adj,NN)
    element=None
    for i in Adj: 
        for j in range(0, len(Adj[i])):
            for k in Adj[Adj[i][j].inp.name]: 
                Adj[i][j].addChild(k)
        element=i
    
    return element


def Traverse(node,Adj,path,paths,count,NN,flops):
    dimensions=sizeofJac(NN,node.name)
    if len(path)>=1:
        print(dimensions,path[-1])
        flops[0]+= GEMMflops(path[-1],dimensions)
        path.append([dimensions[0],path[-1][1]])
    else: 
        path.append(dimensions)

    if len(node.name)==2: 
        for i in Adj[node.name[1].name]:
            if i.visited==False:
                i.visited==True
                Traverse(i,Adj,path,paths,count,NN,flops)
        path.pop()
    if len(Adj[node.name[1].name])==0:
        paths[count[0]]=[]
        for i in path:
            paths[count[0]].append(i)
        paths[count[0]].append(node.name)
        count[0]+=1

def pathFinder(Adj, k,NN): 
    path=[]
    paths={}
    count=[0]
    flops=[0]
    for i in Adj[k]:
        Traverse(i, Adj,path,paths,count,NN,flops)
    return flops[0]/1000000000000

