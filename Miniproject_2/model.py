# Miniproject 2

import torch
from collections import OrderedDict
from typing import Dict, Optional
from torch import empty , cat , arange
from torch.nn.functional import fold, unfold

torch.set_grad_enabled(False)


def conv2d(input, weight, stride=1, padding=0, dilation=1):
    N, _, h_in, w_in = input.shape
    out_channels, in_channels, kernel_size = weight.shape[:-1]
    
    assert input.shape[1] == in_channels
    
    h_out = int((h_in + 2*padding - dilation*(kernel_size-1)-1)/stride+1)
    w_out = int((w_in + 2*padding - dilation*(kernel_size-1)-1)/stride+1)

    x = unfold(input, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride)
    cΠks, L = x.shape[1], x.shape[2]
    
    x = torch.transpose(x, 1, 2).reshape(-1, cΠks)
    weight_flat = weight.reshape(out_channels, cΠks)
    
    x = x @ weight_flat.t()
    x = x.reshape(N, L, out_channels).transpose_(1, 2)
    x = fold(x, output_size=[h_out, w_out], kernel_size=1, padding=0, dilation=dilation, stride=1)
    return x


def conv_transpose2d(input, weight, stride=1, padding=0, dilation=1):
    N, _, h_in, w_in = input.shape
    in_channels, out_channels, kernel_size = weight.shape[:-1]
    
    eff_input = augment(input, nzeros=stride-1, padding=kernel_size-1-padding)
    return  conv2d(eff_input, weight.flip(2,3).transpose(0,1), stride=1, padding=0, dilation=1) 


def augment(input, nzeros, padding=0):
    shape = input.shape
    nold  = shape[-1]
    nnew  = nold + (nold-1)*nzeros
    
    new = torch.zeros(*shape[:2], nnew, nnew)
    new[:,:,::(nzeros+1),::(nzeros+1)] = input
                
    if padding: new = unfold(new,1, padding=padding).reshape(*new.shape[:2],*[new.shape[-1]+2*padding]*2)
    return new


def conv_backward(input, dL_dy, weight, stride=1, padding=0, dilation=1):
    out_channels, in_channels, kernel_size = weight.shape[:-1]
    dL_dx = conv_transpose2d(dL_dy, weight, stride=stride, padding=padding, dilation=dilation)

    ignored = int(input.shape[-1]-dL_dx.shape[-1])
    if ignored:
        dL_dx = unfold(dL_dx, 1, padding=ignored).reshape(*dL_dx.shape[:2],*[dL_dx.shape[-1]+2*ignored]*2)
        dL_dx = dL_dx[:,:,ignored:, ignored:]


    dL_df = torch.zeros_like(weight.transpose(0,1))
    dL_dy_aug = augment(dL_dy, nzeros=stride-1, padding=0)

    x = input if not ignored else input[:,:,:-ignored, :-ignored]
    for mu in range(x.shape[0]):
        for alpha in range(in_channels):
            dLdy = dL_dy_aug[mu].view(1, out_channels,*dL_dy_aug.shape[2:]).transpose(0,1)
            xx   = x[mu,alpha].view(1,1,*x.shape[2:])
            dL_df[alpha] += conv2d(xx, dLdy)[0]

    dL_df.transpose_(0,1)
    return dL_dx, dL_df



class Module(object):
    def __init__(self):
        self.parameters = []
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
        super(Relu,self).__init__()
        return 

    def forward(self,input):
        output= self.slope *0.5 * (input + input.abs())
        return output
    
    __call__ = forward

class Sigmoid(Module):
    def __init__(self,*input):
        super(Sigmoid,self).__init__()
        return 

    def forward(self,input):
        output = 1 / (1 + (-input).exp())
        return output
    
    __call__ = forward

class Sequential(Module):
    def __init__(self, *args):
        super(Sequential,self).__init__()
        self._modules: Dict[str, Optional['Module']] = OrderedDict()

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def __str__(self):
        to_print = '\n'
        for key,val in self._modules.items():
            to_print += key + '\n'
        return to_print

    def add_module(self, name, module):
        self._modules[name] = module

    def forward(self, input):
        for _ , module in self._modules.items():
            input = module(input)
        return input

    __call__ = forward

class MSELoss(Module):
    def __init__(self, *input):
        super(MSELoss,self).__init__()
    
    def forward(self,input,reference):
        n=input.size().numel()
        output = ((input-reference)**2).sum()/n
        return output

    __call__ = forward

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv2d, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride   = stride
        self.padding  = padding
        self.dilation = dilation

        self.weight   = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        self.weight.normal_()
        
    def forward(self, input):
        return conv2d(input, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation)
    
    __call__ = forward
    
    def backward(self, input, dL_dy):
        dL_dx, dL_df = conv_backward(input, dL_dy, self.weight, stride=self.stride,\
                                     padding=self.padding, dilation=self.dilation)
        return dL_dx, dL_df
    
    
class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvTranspose2d,self).__init__
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride   = stride
        self.padding  = padding
        self.dilation = dilation

        self.weight   = torch.Tensor(in_channels, out_channels, kernel_size, kernel_size)
        self.weight.normal_()
        
    def forward(self, input):
        return conv_transpose2d(input, self.weight, stride=self.stride,\
                                padding=self.padding, dilation=self.dilation)
    
    __call__ = forward
    
    def backward(self, input, dL_dy):
        p = self.kernel_size-1-self.padding
        z = self.stride-1
        
        eff_input  = augment(input, nzeros=z, padding=p)
        eff_weight = self.weight.flip(2,3).transpose(0,1)
        dL_dx, dL_df = conv_backward(eff_input, dL_dy, eff_weight, stride=1, padding=0, dilation=1)
        
        dL_df = dL_df.flip(2,3).transpose(0,1)
        return dL_dx[:,:,p:-p:z+1, p:-p:z+1], dL_df


































# ---------------------------------------------------------------------------------------------------
#                                           NETWORK
# ---------------------------------------------------------------------------------------------------

class Model():
    
    def __init__(self):  
        relu1 = Relu(name='name')
        sig1 = Sigmoid(name='name')  
        pass

    
    

    
