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

from typing import Dict
import PyTorch_CIFAR10.cifar10_models.googlenet as vgg
from NNSENNparse import  parseNetworkIn
f=[]
mod=[]
m=vgg.GoogLeNet()
g=torch.fx.symbolic_trace(m)
for i in g.graph.nodes:
    if i.op=='call_module':
        mod.append(i)
    elif i.op=='call_function':
        f.append(i)

print(type(g._modules))
for i in g._modules:
    x=g._modules[i]._modules
    # print(type(g._modules[i]))
    for l in x:
        k=x[l]._modules
        if type(x[l])!=torch.nn.modules.module.Module:
            print(type(x[l]))
        for p in k:
            z=k[p]._modules
            # if type(k[p])!='<class torch.nn.modules.module.Module>':
            #     print(type(k[p]))