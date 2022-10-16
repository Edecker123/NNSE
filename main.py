from turtle import Shape
import torch #first we import the library 
import tabulate
import torch
import torch.fx
from torch.fx.node import Node
import collections
from typing import Dict
import PyTorch_CIFAR10.cifar10_models.resnet as vgg
from NNparse import targetLook, Node,loadArgs

m=vgg.resnet50()
x=torch.ones(3,3,224,224)
m(x)
g=torch.fx.symbolic_trace(m)
mods=g._modules
g=g.graph

# class ShapeProp:
#     """
#     Shape propagation. This class takes a `GraphModule`.
#     Then, its `propagate` method executes the `GraphModule`
#     node-by-node with the given arguments. As each operation
#     executes, the ShapeProp class stores away the shape and
#     element type for the output values of each operation on
#     the `shape` and `dtype` attributes of the operation's
#     `Node`.
#     """
#     def __init__(self, mod):
#         self.mod = mod
#         self.graph = mod.graph
#         self.modules = dict(self.mod.named_modules())

#     def propagate(self, *args):
#         args_iter = iter(args)
#         env : Dict[str, Node] = {}

#         def load_arg(a):
#             return torch.fx.graph.map_arg(a, lambda n: env[n.name])

#         def fetch_attr(target : str):
#             target_atoms = target.split('.')
#             attr_itr = self.mod
#             for i, atom in enumerate(target_atoms):
#                 if not hasattr(attr_itr, atom):
#                     raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
#                 attr_itr = getattr(attr_itr, atom)
#             return attr_itr

#         for node in self.graph.nodes:
#             if node.op == 'placeholder':
#                 result = next(args_iter)
#             elif node.op == 'get_attr':
#                 result = fetch_attr(node.target)
#             elif node.op == 'call_function':
#                 result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
#             elif node.op == 'call_method':
#                 self_obj, *args = load_arg(node.args)
#                 kwargs = load_arg(node.kwargs)
#                 result = getattr(self_obj, node.target)(*args, **kwargs)
#             elif node.op == 'call_module':
#                 result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

#             # This is the only code specific to shape propagation.
#             # you can delete this `if` branch and this becomes
#             # a generic GraphModule interpreter.
#             if isinstance(result, torch.Tensor):
#                 node.shape = result.shape
#                 node.dtype = result.dtype

#             env[node.name] = result
#         return load_arg(self.graph.result)

for node in g.nodes:
    print(node.op,type(node.target), node.target)
def parseNet(net,inp):
    graph={}
    trace=torch.fx.symbolic_trace(net)
    mods=trace._modules
    Ngraph=trace.graph
    nodes=Ngraph.nodes
    for node in nodes:
        if node.op=="placeholder":
            N=Node(None, node.name, None,inp,node.op)
            graph[node.name]=N
        elif node.op=="call_module":
            operation=targetLook(node.target,mods)
            args=loadArgs(node,graph)
            result=operation(*args)
            N=Node(args, node.name,operation, result, node.op)
            graph[node.name]=N
        elif node.op=="call_function":
            operation=node.target
            args=loadArgs(node,graph)
            result=operation(*args)
            N=Node(args, node.name,operation, result, node.op)
            graph[node.name]=N
        elif node.op=="call_method":
            
        elif node.op=="get_attr":
            print(node)
    print(graph)

parseNet(m,x)





