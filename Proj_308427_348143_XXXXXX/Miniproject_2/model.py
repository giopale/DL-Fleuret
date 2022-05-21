# Miniproject 2

from email import generator
import torch, math
from collections import OrderedDict
from typing import Dict, Optional
from torch.nn.functional import fold, unfold
import copy

torch.set_grad_enabled(False)

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    assert dimensions >= 2

    num_input_fmaps  = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    
    if tensor.dim() > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    
    fan_in  = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_normal_(tensor, gain=1.4142135623730951):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    tensor.normal_(0, std)
    return 

def constant_(tensor, val:float):
    tensor.fill_(val)
    return 

def _init_weights(model):
    if isinstance(model, Conv2d) or isinstance(model, TransposeConv2d):
        xavier_normal_(model.weight) # model.weight.normal_(0,0.5,generator=torch.manual_seed(0)) #
        constant_(model.bias, 0.)


def conv2d(X, K, stride=1, padding=0, dilation=1, bias=torch.Tensor([])):
    N, _, H, W = X.shape
    out_channels, in_channels, h, w = K.shape
    assert X.shape[1] == in_channels
    assert w == h
    
    h_out = int((H + 2*padding - dilation*(h-1)-1)/stride+1)
    w_out = int((W + 2*padding - dilation*(w-1)-1)/stride+1)

    Xprime  = unfold(X, kernel_size=h, padding=padding, dilation=dilation, stride=stride)
    cΠks, L = Xprime.shape[1], Xprime.shape[2]
    Xprime  = torch.transpose(Xprime, 1, 2).reshape(-1, cΠks)

    Kprime  = K.reshape(out_channels, cΠks)
    
    Yprime = Xprime @ Kprime.t()    
    Y = Yprime.reshape(N, L, out_channels).transpose_(1, 2)
    Y = fold(Y, output_size=[h_out, w_out], kernel_size=1, padding=0, dilation=1, stride=1)

    if len(bias):
        bias = bias.expand(N,h_out,w_out,out_channels).permute(0,3,1,2)
        Y += bias
    return Y


def conv_transpose2d(Y, K, stride=1, padding=0, dilation=1, bias=torch.Tensor([])):
    N, _, H, W = Y.shape
    in_channels, out_channels, h, w = K.shape
    assert Y.shape[1] == in_channels
    assert w == h
    
    h_out = (H-1)*stride - 2*padding + dilation*(h-1) + 1
    w_out = (W-1)*stride - 2*padding + dilation*(w-1) + 1

    Yprime = Y.flatten(-2,-1)
    Yprime = Yprime.transpose_(1, 2).flatten(0,1)
    
    KT = K.flatten(1,-1)
    
    Xprime = Yprime @ KT
    Xprime = Xprime.reshape(N, -1, Xprime.shape[-1]).transpose(1,2)
    X = fold(Xprime, output_size=[h_out,w_out], kernel_size=h, padding=padding, dilation=dilation, stride=stride)
    
    if len(bias):
        bias = bias.expand(N,h_out,w_out,out_channels).permute(0,3,1,2)
        X += bias
    return X




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
    return (dL_dx, dL_df)




class Module(object):
    def __init__(self):
        self.weight   = torch.Tensor([])
        self.bias     = torch.Tensor([])
        self.d_weight = torch.Tensor([])
        self.d_bias   = torch.Tensor([])
        pass
    def forward (self) :
        raise NotImplementedError
    def backward (self, input):
        raise NotImplementedError
    def param (self) :
        return self.parameters


class ReLU(Module):
    def __init__(self):
        super().__init__()
        return 

    def forward(self, input):
        return torch.relu(input)
    __call__ = forward

    def forward_and_vjp(self, input):
        def _vjp(dL_dy):
            return (dL_dy*(input > 0), torch.Tensor([]), torch.Tensor([]))
        return self.forward(input), _vjp


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        return 

    def forward(self,input):
        return torch.sigmoid(input)
    __call__ = forward

    def forward_and_vjp(self, input):
        def _vjp(dL_dy):
            dsigma_dx = torch.sigmoid(input)*(1.-torch.sigmoid(input))
            return (dL_dy*dsigma_dx , torch.Tensor([]), torch.Tensor([]))
        return self.forward(input), _vjp


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, initialize=False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride   = stride
        self.padding  = padding
        self.dilation = dilation

        self.weight   = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        self.bias     = torch.Tensor(out_channels)
        if initialize: _init_weights(self)
        
    def forward(self, input):
        return conv2d(input, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=self.bias)
    __call__ = forward

    def forward_and_vjp(self, input):
        def _vjp(dL_dy): 
            dL_dx, dL_df = conv_backward(input, dL_dy, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation)
            return dL_dx, dL_df, dL_dy.sum((0,2,3))
        return self.forward(input), _vjp
            
    
class TransposeConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, initialize=False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride   = stride
        self.padding  = padding
        self.dilation = dilation

        self.weight   = torch.Tensor(in_channels, out_channels, kernel_size, kernel_size)
        self.bias     = torch.Tensor(out_channels)
        if initialize: _init_weights(self)

    def forward(self, input):
        return conv_transpose2d(input, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=self.bias)
    __call__ = forward
    

    def forward_and_vjp(self, input):
        p = self.kernel_size-1-self.padding
        z = self.stride-1
        
        eff_input  = augment(input, nzeros=z, padding=p)
        eff_weight = self.weight.flip(2,3).transpose(0,1)
        
        def _vjp(dL_dy):
            dL_dx, dL_df = conv_backward(eff_input, dL_dy, eff_weight, stride=1, padding=0, dilation=1)
            dL_df = dL_df.flip(2,3).transpose(0,1)
            dL_dx = dL_dx[:,:,p:-p:z+1, p:-p:z+1]
            return dL_dx, dL_df, dL_dy.sum((0,2,3))
        return self.forward(input), _vjp


class MSE(Module):
    def __init__(self):
        super().__init__()
        self.input : torch.Tensor
        self.reference : torch.Tensor
    
    def forward(self, input, reference):
        return (((input-reference)**2).sum()/input.size().numel()).item()
    __call__ = forward

    def forward_and_vjp(self, reference):
        dL_dy = lambda input : 2*(input - reference)/input.size().numel()
        return dL_dy


class Sequential():
    def __init__(self, *args, initialize=True):
        self._modules: Dict[str, Optional['Module']] = OrderedDict()
        self.nb_modules = len(args)

        for idx, module in enumerate(args):
            self._add_module(str(idx), module)

        if initialize: self._initialize()


    def _add_module(self, name, module):
        self._modules[name] = module

    def _initialize(self):
        for module in self._modules.values():
            _init_weights(module)
        return 

    def parameters(self):
        for module in self._modules.values():
            if len(module.weight):
                yield [module.weight, module.bias, module.d_weight, module.d_bias]

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input
    __call__ = forward

    def forward_and_vjp(self, input, vjp_loss):
        VJP = [None]*self.nb_modules
        for i,module in enumerate(self._modules.values()):
            input, VJP[i] = module.forward_and_vjp(input)

        dL_dy = vjp_loss(input)
        for i, (module, vjp) in enumerate( zip( reversed(self._modules.values()), reversed(VJP) )  ):
            dL_dy, module.d_weight, module.d_bias = vjp(dL_dy)







#===========================================================
#                          MODEL                                            
#===========================================================  


class Model():

    def __init__(self) -> None:
        self.stride      = 2
        self.kernel_size = 2
        self.features    = 32

        self.loss = MSE()

        self.eta         = 0.75
        self.gamma       = 0.5
        self.params_old  = None
        self.batch_size  = 16
    
        conv1   = Conv2d(in_channels=3, out_channels=self.features, stride=self.stride,  kernel_size=self.kernel_size)
        conv2   = Conv2d(in_channels=self.features, out_channels=self.features, stride=self.stride,  kernel_size=self.kernel_size)
        tconv1  = TransposeConv2d(in_channels=self.features, out_channels=self.features,  stride=self.stride,  kernel_size=self.kernel_size, padding=0, dilation=1)
        tconv2  = TransposeConv2d(in_channels=self.features, out_channels=3,  stride=self.stride,  kernel_size=self.kernel_size, padding=0, dilation=1)
        relu    = ReLU()
        sigmoid = Sigmoid()

        self.net = Sequential(conv1, relu, conv2, relu, tconv1, relu, tconv2, sigmoid, initialize=True)

        
    def predict(self,x) -> torch.Tensor:
        return self.net.forward(x)


    def train(self, train_input, train_target, nb_epochs=5):
        for e in range(nb_epochs):
            for inputs, targets in zip(train_input.split(self.batch_size), train_target.split(self.batch_size)):

                dL_dy = self.loss.forward_and_vjp(targets)
                self.net.forward_and_vjp(inputs, dL_dy)
                self.SGD()
            print("\rCompleted: %d/%d"%(e+1,nb_epochs), end=' ')
        return 


    def SGD(self):
        for p in self.net.parameters():
                        
            #if self.params_old:
            grad_w = self.eta * p[2]
            grad_b = self.eta * p[3]
            #else:
            #    grad = self.gamma * self.params_old[1] + self.eta * p[1] 

            p[0] -= grad_w
            p[1] -= grad_b

        #self.params_old = copy.deepcopy(list(self.net.parameters()))

    def save(self, filename) -> None:
        torch.save(self.state_dict(), filename)

    def load_pretrained_model(self, filename='Proj_308427_348143_XXXXXX/Miniproject_2/bestmodel.pth') -> None:
        new_model = torch.load(filename)
        self=new_model

                

