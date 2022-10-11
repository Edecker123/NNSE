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

from typing import Dict

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten=nn.Flatten() #this will take the image and flatten it into a vector of pixels 
        self.linear_relu_stack=nn.Sequential(
            nn.Linear(28*28, 64), #this applies a linear transformation on the vectors with the indices as weights and biasses
            nn.ReLU(), #this is our activation
            nn.Linear(64,10),
            nn.ReLU()
        )
    
    def forward(self, x):
        x=self.flatten(x)
        logits=self.linear_relu_stack(x)

        return logits
# model=torch.load('model.pth')
model=NeuralNetwork()

X=torch.ones(1,784)
def operationTraverse(pred):
    operation_stack=[]
    operation_stack.append(pred.grad_fn)
    while(len(operation_stack)!=0):
        operation=operation_stack.pop()
        if type(operation) is tuple:
            operation=operation[0]
        if operation==None:
            continue
        for i in operation.next_functions:
            operation_stack.append(i)
            print(i)

a=torch.tensor(2)
b=torch.tensor(3)

def m(a,b):
    return a*b

gm = torch.fx.symbolic_trace(model)
modules=gm._modules
for module in modules:
    inner_mod=modules[module]._modules
    for i in inner_mod:
        operation=inner_mod[i]
        print(operation)




