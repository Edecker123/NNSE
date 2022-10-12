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
import collections
from typing import Dict
import PyTorch_CIFAR10.cifar10_models.vgg as vgg
from NNSENNparse import  parseNetworkIn
f=[]
mod=[]
m=vgg.vgg11_bn()
g=torch.fx.symbolic_trace(m)
for i in g.graph.nodes:
    if i.op=='call_module':
        mod.append(i)
    elif i.op=='call_function':
        f.append(i)

opHash={}
opTypes=[]
#grab modules
graphedModel = torch.fx.symbolic_trace(m)
modules=graphedModel._modules

opStack=[]
opStack.append(modules)


while len(opStack)!=0:
    mods=opStack.pop()
    if type(mods)==torch.nn.modules.module.Module:
        mods=mods._modules
    for i in mods:
        otype=type(mods[i])
        #check if it is a one layer module
        if otype==torch.nn.modules.module.Module:
            opStack.append(mods[i])
        else:
            if otype in opHash:
                opHash[otype].append(mods[i])
            else:
                opHash[otype]=[]
                opTypes.append(otype)
                opHash[otype].append(mods[i])

op=opHash[torch.nn.modules.pooling.MaxPool2d][0]

for i in opTypes:
    print(i)