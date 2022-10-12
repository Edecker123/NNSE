import torch #first we import the library 
import tabulate
import torch
import torch.fx
from torch.fx.node import Node
import collections
from typing import Dict
import PyTorch_CIFAR10.cifar10_models.vgg as vgg
from NNparse import  parseNet


m=vgg.vgg11_bn()
g=torch.fx.symbolic_trace(m)


for i in parseNet(m):
    print(parseNet(m)[i])