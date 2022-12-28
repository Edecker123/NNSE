from turtle import Shape
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
import PyTorch_CIFAR10.cifar10_models.inception as inc
from NNparse import targetLook, Node,loadArgs, loadKwargs, parseNet, getOps
from torch import autograd

m=mlp.MLP()
v=vgg.vgg11_bn(False)
g=goog.GoogLeNet()
r=res.resnet18()
i=inc.inception_v3()

x=torch.zeros(1,3,128,128)


# m_d=parseNet(m, x2)
# v_d=parseNet(v, x)
# g_d=parseNet(g,x)
r_d=parseNet(r,x)
# i_d=parseNet(i,x)

print(r_d)

    





