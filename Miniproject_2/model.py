# Miniproject 2

import torch
from collections import OrderedDict
from typing import Dict, Optional
from torch import empty , cat , arange
from torch.nn.functional import fold, unfold

torch.set_grad_enabled(False)

class Module(object):
    _parameters=[]
    _dl_dw=[]
    def __init__(self):
        pass
    def forward (self,*input) :
        raise NotImplementedError
    def backward (self, *cose):
        raise NotImplementedError
    def param (self) :
        return []

class Relu(Module):
    slope=1.0

    def __init__(self,*input):
        super(Module).__init__()
        return 

    def forward(self,input):
        output= self.slope *0.5 * (input + input.abs())
        return output
    
    __call__ = forward

class Sigmoid(Module):
    slope=1.0

    def __init__(self,*input):
        super(Module).__init__()
        return 

    # def __call__(self, *input):
        # return self.forward(*input)

    def forward(self,input):
        output = 1 / (1 + (-input).exp())
        return output
    
    __call__ = forward


class Sequential(Module):
    def __init__(self, **kwargs):
        super(Module, self).__init__()
        self._modules: Dict[str, Optional['Module']] = OrderedDict()

        for key, val in kwargs.items():
            self.add_module(key, val)

    def __str__(self):
        to_print = '\n'
        for key,val in self._modules.items():
            to_print += key + '\n'
        return to_print

    def add_module(self, name, module):
        self._modules[name] = module

    def forward(self, input):
        print('input: ', input)
        for name , module in self._modules.items():
            input = module(input)
            print(name, input)
        return input

    __call__ = forward


class Model():
    
    def __init__(self):
        # do some stuff 
        
        pass

    

    
