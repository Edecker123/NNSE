
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
import PyTorch_CIFAR10.cifar10_models.resnet as res
import PyTorch_CIFAR10.cifar10_models.resnet_orig as mog

from NNparse import targetLook, Node,loadArgs, loadKwargs, parseNet, getOps, oParseMods
from torch import autograd
import os
from Fcount import NNSE
import time
from NNtreegeneration import genAdjList, pathFinder,dagConnect

import torchvision.models.googlenet as goog
import torchvision.models.alexnet as alex
m=mlp.MLP()
v=vgg.vgg19_bn(False)
g=goog()
a=alex()
device=torch.device('mps')
r=res.resnet34()
import torchvision.models.resnet as res
x=torch.zeros(1,3,224,224)


d=parseNet(g,x)

print(g(x))
p=genAdjList(d)
k=dagConnect(p,d)
# for i in p:
#     for j in p[i]:
#         print(j.child)
paths=pathFinder(p,k,d,True)
print(paths , "TFLOPs")

