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

#parses the forward pass into a neural network, see parseNetworkB for the parsing of operations in the backpass
def parseNetMods(m):
    opHash={}
    opTypes=[]
    #grab modules
    graphedModel = torch.fx.symbolic_trace(m)
    modules=graphedModel._modules

    opStack=[]
    opStack.append(modules)


    while len(opStack)!=0:
        mods=opStack.pop()
        if type(mods)==torch.nn.modules.module.Module:
            mods=mods._modules
        for i in mods:
            otype=type(mods[i])
            #check if it is a one layer module
            if otype==torch.nn.modules.module.Module:
                opStack.append(mods[i])
            else:
                if otype in opHash:
                    opHash[otype].append(mods[i])
                else:
                    opHash[otype]=[]
                    opTypes.append(otype)
                    opHash[otype].append(mods[i])
    return opHash

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


def oParseMods(net):
    m=torch.fx.symbolic_trace(net)
    #first we grab a container of the modules (these are in order )
    #check if first element can be broken down, if not append then pop, else front append
    modules=[]
    oParse={}
    count=0
    for i in m._modules:
        modules.append(m._modules[i])
        print(i)
    #start with a stack of modules in order 
    while len(modules)!=0:
        #first we pop off the front element 
        mod=modules.pop(0)
        print(mod)
        # print(type(mod))
        #if its breakable then the type will tell 
        #if breakable append and loop 
        #else keep popped 
        if len(mod._modules)>0:
            inter=mod._modules
            catlist=[]
            #now we make a list to append to 
            for i in inter:
                catlist.append(mod._modules[i])
            #push the broken modules back to the front 
            modules=catlist+modules
        else:
            #this means it is a actual module and we append it to our output list
            oParse[count]=mod
            count+=1
    return oParse

def targetLook(target, mods):
    atoms=target.split('.')
    op=None
    while len(atoms)!=0:
        key=atoms.pop(0)
        inter=mods[key]
        if len(atoms)!=0:
            mods=inter._modules
        else:
            op=inter
            return op

def loadArgs(node,graph):
    #first we grab the args
    args=node.args
    argsR=[]
    #next we look them up 
    for i in args:
        if type(i)==torch.fx.node.Node:
            N=graph[i.name]
            argsR.append(N.result)
        else:
            argsR.append(i)
    return tuple(argsR)
class Node():
    def __init__(self, inputs, name, operation,result,nodeop):
        self.operation=operation
        self.inputs=inputs
        self.result=result
        self.name=name
        self.nodeop=nodeop

