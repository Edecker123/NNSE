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
m=torch.load('c')

# m=torch.nn.BatchNorm2d(3)
# m=torch.nn.Sequential(torch.nn.Conv2d(3,4, (2,2)), torch.nn.BatchNorm2d(4))
torch.save(m,'c')
x=torch.tensor([[[[1.0,3.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]],[[2.0,2.0,2.0],[2.0,2.0,2.0],[2.0,2.0,2.0]],[[3.0,3.0,3.0],[3.0,3.0,3.0],[3.0,3.0,3.0]]]])
# x=torch.ones(1,3,12,12)

# print(m(x))
x.requires_grad=True
j=torch.autograd.functional.jacobian(m, x)
print(j.shape)
print(j[0][2][0][0][0][0]) #it seems that each channel has a dependancy on every input channel
                           #

x2=torch.ones(1,1,784)


# m_d=parseNet(m, x2)
# v_d=parseNet(v, x)
# g_d=parseNet(g,x)
r_d=parseNet(r,x)
# i_d=parseNet(i,x)

for i in r_d:
    if type(r_d[i].operation)==torch.nn.modules.batchnorm.BatchNorm2d:
        print(r_d[i].prev)
    





