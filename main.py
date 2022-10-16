from turtle import Shape
import torch #first we import the library 
import tabulate
import torch
import torch.fx
from torch.fx.node import Node
import collections
from typing import Dict
import PyTorch_CIFAR10.cifar10_models.mobilenetv2 as vgg
from NNparse import targetLook, Node,loadArgs, loadKwargs, parseNet

m=vgg.MobileNetV2
x=torch.ones(3,3,224,224)
m(x)


d=parseNet(m,x)

for i in d:
    print(d[i].operation)






