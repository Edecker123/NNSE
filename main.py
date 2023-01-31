
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
m=mlp.MLP()
v=vgg.vgg19()
g=goog()
a=alex()
device=torch.device('mps')
r=res.resnet18()
import torchvision.models.resnet as res
x=torch.zeros(1,3,128,128)


d=parseNet(r,x)

print(r(x).shape)
p=genAdjList(d)
k=dagConnect(p,d)
# for i in p:
#     for j in p[i]:
#         print(j.child)
paths=pathFinder(p,k,d,True)
print(paths , "GFLOPs")

# print(NNSE(r,x)/1000000000, "GFLOPs")
convs=[]
for i in d: 
    if type(d[i].operation) == torch.nn.modules.conv.Conv2d:
        convs.append(d[i])

    if d[i].nodeop=="output":
        convs.append(d[i])
    if d[i].nodeop=="placeholder":
        convs.append(d[i])

lens=len(convs)
shapearr=[]
flop=0
for j in range(1, len(convs)):
    lens=lens-1
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

print((1000)*flop/1000000000000 ,"GFLOPs for only convs assuming triangular matrix")