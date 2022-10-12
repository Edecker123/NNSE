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

#parses the forward pass into a neural network, see parseNetworkB for the parsing of operations in the backpass 
def parseNetworkIn(net):
    opHash={}
    opTypes=[]
    graphedModel = torch.fx.symbolic_trace(net)
    modules=graphedModel._modules
    for module in modules:
        inner_mod=modules[module]._modules
        for i in inner_mod:
            operation=inner_mod[i]
            otype=type(operation)
            if otype in opHash:
                opHash[otype].append(operation)
            else:
                opHash[otype]=[]
                opTypes.append(otype)
                opHash[otype].append(operation)
    return opHash,opTypes

def operationTraverse(pred):
    operation_stack=[]
    operation_stack.append(pred.grad_fn)
    while(len(operation_stack)!=0):
        operation=operation_stack.pop()
        if type(operation) is tuple:
            operation=operation[0]
        if operation==None:
            continue
        for i in operation.next_functions:
            operation_stack.append(i)
            print(i)


