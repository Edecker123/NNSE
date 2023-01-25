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

count=0
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
        return maxpool2DF(op)
    if type(op.operation) == torch.nn.modules.pooling.AdaptiveAvgPool2d or op.operation==avg or op.operation==avg2 or op.operation==avg3:
        return avgpool2DF(op)
    if op.operation == k_d['add_1'].operation:
        return(addF(op))

    return  0

def NNSE(nn, x):
    netHash=parseNet(nn,x)
    inferenceflops=0
    for op in netHash:
            
        flops=Icount(netHash[op])
        inferenceflops+=flops
    return inferenceflops
    
def linearF(op):
    size=op.inputs[0].shape
    outputsize=op.result.shape
    if type(outputsize)==int:
        flops=size[0]*outputsize
        return flops 
    else:
        flops=1
        for i in outputsize:
            flops=flops*i
        return flops*size[0]

def ReluF(op):
    inputsize=op.inputs[0].shape
    flops=1
    for i in inputsize:
        flops=flops*i
    return flops

def SigmoidF(op):
    size=op.inputs[0].shape
    outputsize=op.result.shape
    flops=1
    for i in size:
        flops=flops*i
    return flops*3

def conv2DF(op):
    size=op.inputs[0].shape
    kernalsize=op.operation.kernel_size
    outputsize=op.result.shape
 
    H=(size[2]-kernalsize[0]) / op.operation.stride[0]
    W=(size[3]-kernalsize[1]) / op.operation.stride[1]#assumes no padding #assumes no dialation modification
    flops=((kernalsize[0]*kernalsize[1])*size[1] + 1)*outputsize[1]*outputsize[2]*outputsize[3]#add channel dimension back
    return flops

def batchnorm2DF(op):
    size=op.inputs[0].shape
    flops=1
    for i in size:
        flops=flops*i
    return flops*9
    

def maxpool2DF(op):
    kernelsize=0
    if len(op.inputs)>1:
        kernelsize=op.inputs[1]
        if type(kernelsize) == tuple:
                kernelsize=kernelsize[0]*kernelsize[1]
        if type(kernelsize)==int:
            kernelsize=kernelsize*kernelsize
        else:
            print(3)
    else:
        try:
            kernelsize=op.operation.kernel_size
            if type(kernelsize) == tuple:
                kernelsize=kernelsize[0]*kernelsize[1]
            if type(kernelsize)==int:
                kernelsize=kernelsize*kernelsize
        except:
            print("could not get kernel size")

    size=op.result[0].shape
    flops=1
    for i in size:
        flops=flops*i
    
    return flops*kernelsize

#assumes no padding and size of 1
def avgpool2DF(op):
    size=op.inputs[0].shape
    size0=op.result.shape
    flops=1
    for i in size0:
        flops=flops*i
    
    flops=flops*(size[2]-size0[2]+1)*(size[3]-size0[3]+1)
    return (flops) #add channel dimension back

def addF(op):
    shape1=op.inputs[0].shape
     #shapes are the same
    flops=1
    for i in shape1: 
        flops=i*flops
    
    return 2*flops 

def GEMMflops(matdem1, matdem2): 
    #check they can  multiply: 
    if matdem1[0]!=matdem2[1]:
        return None
    
    else:
        return 2*matdem1[0]*matdem1[1]*matdem2[1]
