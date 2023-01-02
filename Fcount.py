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
from NNparse import targetLook, Node,loadArgs, loadKwargs, parseNet, getOps
import PyTorch_CIFAR10.cifar10_models.resnet as res
import PyTorch_CIFAR10.cifar10_models.inception as inc


x=torch.zeros(1,3,128,128)
i=inc.inception_v3()
res=res.resnet18()
i_d=parseNet(i,x)
k_d=parseNet(res,x)


r=torch.nn.functional.relu
cat=torch.cat
avg=torch.nn.functional.adaptive_avg_pool2d
avg2=torch._C._nn.avg_pool2d
maxpool=torch.nn.functional.max_pool2d
add=k_d['add_1'].operation
avg3=i_d['avg_pool2d'].operation


#op is of type node
def Icount(op):
    if type(op.operation) == torch.nn.modules.linear.Linear:
        return linearF(op)
    if type(op.operation)== torch.nn.modules.activation.ReLU or op.operation==r:
        return ReluF(op)
    if type(op.operation) == torch.nn.modules.activation.Sigmoid:
        return SigmoidF(op)
    if type(op.operation) == torch.nn.modules.conv.Conv2d:
        return conv2DF(op)
    if type(op.operation) == torch.nn.modules.batchnorm.BatchNorm2d:
        return(batchnorm2DF(op))
    if type(op.operation) == torch.nn.modules.pooling.MaxPool2d or op.operation==maxpool:
        return(maxpool2DF)
    if type(op.operation) == torch.nn.modules.pooling.AdaptiveAvgPool2d or op.operation==avg or op.operation==avg2 or op.operation==avg3:
        return(avgpool2DF)
    if op.operation == k_d['add_1'].operation:
        return(addF(op))
    print(op.operation)

def NNSE(nn, x):
    netHash=parseNet(nn,x)

    for op in netHash:
        flops=Icount(netHash[op])


def linearF(op):
    pass

def ReluF(op):
    pass

def SigmoidF(op):
    pass

def conv2DF(op):
    pass

def batchnorm2DF(op):
    pass

def maxpool2DF(op):
    pass

def avgpool2DF(op):
    pass

def addF(op):
    pass