from turtle import Shape
import torch #first we import the library 
import tabulate
import torch
import torch.fx
from torch.fx.node import Node
import collections
from typing import Dict
import PyTorch_CIFAR10.cifar10_models.mlp as mlp
import PyTorch_CIFAR10.cifar10_models.vgg as vgg
import PyTorch_CIFAR10.cifar10_models.googlenet as goog
import PyTorch_CIFAR10.cifar10_models.resnet as res
import PyTorch_CIFAR10.cifar10_models.inception as inc
from NNparse import targetLook, Node,loadArgs, loadKwargs, parseNet

m=mlp.MLP()
v=vgg.vgg11_bn(False)
g=goog.GoogLeNet()
r=res.resnet18()
i=inc.inception_v3()


x=torch.ones(3,3,224,224)
x2=torch.ones(1,784)
i(x)

m_d=parseNet(m, x2)
v_d=parseNet(v, x)
g_d=parseNet(g,x)
r_d=parseNet(r,x)
i_d=parseNet(i,x)

def getOps(netGraph):
    op={}
    for i in netGraph:
        if type(netGraph[i].operation) in op:
            op[type(netGraph[i].operation)]+=1
        else:
            op[type(netGraph[i].operation)]=1
    return op




