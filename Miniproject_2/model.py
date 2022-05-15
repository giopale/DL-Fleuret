# Miniproject 2

import torch
import math
from torch import empty , cat , arange
from torch.nn.functional import fold, unfold

torch.set_grad_enabled(False)

class Module(object):
    parameters=[]
    dl_dw=[]
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

        def __init__(self):
            super(Module).__init__()
            return 

        def forward(self,input):
            output= self.slope *0.5 * (input + input.abs())
            return output

class Sigmoid(Module):
        slope=1.0

        def __init__(self, *input):
            super(Module).__init__()
            return 

        # def __call__(self, *input):
            # return self.forward(*input)

        def forward(self,input):
            output = 1 / (1 + (-input).exp())
            return output


class Model():
    
    def __init__(self):
        # do some stuff 

        pass

    
