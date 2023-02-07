
import torch #first we import the library 
import torch
import torch.fx
import PyTorch_CIFAR10.cifar10_models.mlp as mlp
import PyTorch_CIFAR10.cifar10_models.vgg as vgg
import PyTorch_CIFAR10.cifar10_models.googlenet as goog
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
import torchvision.models.resnet as res
from testsuite import convse,getGraphModule
mlpNetwork=mlp.MLP()
vgg11=vgg.vgg11()
googleNet=goog()
alexNet=alex()
resNet18=res.resnet18()
#instantiation for code testing data
device=torch.device('mps')
x=torch.zeros(1,3,224,224)
d=parseNet(vgg11,x)

graoh=getGraphModule(googleNet)


p=genAdjList(d)
k=dagConnect(p,d)
# for i in p:
#     for j in p[i]:
#         print(j.child)
paths=pathFinder(p,k,d,True)
print(paths , "GFLOPs")

# # print(NNSE(r,x)/1000000000, "GFLOPs")
convs=[]
for i in d: 
    if type(d[i].operation) == torch.nn.modules.conv.Conv2d:
        convs.append(d[i])

    if d[i].nodeop=="output":
        convs.append(d[i])
    if d[i].nodeop=="placeholder":
        convs.append(d[i])



flop=convse(convs)

# print(flop/1000000000 ,"GFLOPs for only convs assuming triangular matrix")
# print(NNSE(resNet18,x)/1000000000 , "GFLOPs")