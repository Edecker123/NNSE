import torch #first we import the library 
import tabulate
import torch
import torch.fx
from torch.fx.node import Node
import collections
import functorch
from functorch import jacrev,vjp,jvp,vmap
from typing import Dict
import PyTorch_CIFAR10.cifar10_models.mlp as mlp
import PyTorch_CIFAR10.cifar10_models.vgg as vgg
import PyTorch_CIFAR10.cifar10_models.googlenet as goog
# import PyTorch_CIFAR10.cifar10_models.resnet as res
import PyTorch_CIFAR10.cifar10_models.resnet_orig as mog

from NNparse import targetLook, Node,loadArgs, loadKwargs, parseNet, getOps, oParseMods
from torch import autograd
import os
from Fcount import NNSE, GEMMflops
import time
from NNtreegeneration import genAdjList, pathFinder,dagConnect
import torchvision.models.resnet as res
import torchvision.models.googlenet as goog
import torchvision.models.alexnet as alex
import torchvision.models.vgg as vgg


def convse(convs):
    shapearr=[]
    lenc=len(convs)-1
    flop=0
    for j in range(0, len(convs)):
        lens=lenc-j
        result1=convs[lens].result
        result2=convs[lens-1].result

        size1=1
        for i in result1.shape: 
            size1=size1*i

        size2=1
        for i in result2.shape:
            size2=size2*i
        
        shape=[size2,size1]
        
        if len(shapearr)>0: 
            flop+=GEMMflops(shapearr[-1],shape)
            shapearr.append([shape[0],shapearr[-1][1]])
        else: 
            shapearr.append(shape)
    return flop


'''
Will get the graph object (defined in torch.fx library) of a neural network 
'''
def getGraphModule(net): 
    trace=torch.fx.symbolic_trace(net)
    Ngraph=trace.graph
    nodes=Ngraph.nodes


    for i in nodes: 
       pass 
    return nodes

