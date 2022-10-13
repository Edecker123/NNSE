import torch #first we import the library 
import tabulate
import torch
import torch.fx
from torch.fx.node import Node
import collections
from typing import Dict
import PyTorch_CIFAR10.cifar10_models.vgg as vgg
from NNparse import  parseNet
from NNparse import oParse

m=vgg.vgg11_bn()
g=torch.fx.symbolic_trace(m)



o=oParse(m)

for i in o:
    print(i,o[i], '\n')


