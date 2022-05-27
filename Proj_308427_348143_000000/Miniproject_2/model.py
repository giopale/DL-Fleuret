# Miniproject 2
import torch, math
from collections import OrderedDict
from typing import Dict, Optional
from torch.nn.functional import fold, unfold, pad
import pickle, copy



#==================================================================================================================#
#==================================================================================================================#
#                                               INITIALIZERS
#==================================================================================================================#
#==================================================================================================================#


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




#==================================================================================================================#
#==================================================================================================================#
#                                                  FUNCTIONALS
#==================================================================================================================#
#==================================================================================================================#


def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()



def process_dataset(data, normalize=False):
    if data.dtype==torch.uint8: data = data.float()
    if data.max()>1 and normalize: data = data/255.
    return data



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
                
    if padding: 
        new = pad(new, (padding, padding, padding, padding))
        #new = unfold(new,1, padding=padding).reshape(*new.shape[:2],*[new.shape[-1]+2*padding]*2)
    return new


def conv_backward(input, dL_dy, weight, stride=1, padding=0, dilation=1):
    dL_dx = conv_transpose2d(dL_dy, weight, stride=stride, padding=padding, dilation=dilation)
    
    ignored = int(input.shape[-1]-dL_dx.shape[-1])
    if ignored:
        dL_dx = pad(dL_dx, (0,ignored,0,ignored))
        #dL_dx = unfold(dL_dx, 1, padding=ignored).reshape(*dL_dx.shape[:2],*[dL_dx.shape[-1]+2*ignored]*2)
        #dL_dx = dL_dx[:,:,ignored:, ignored:]

    xt = input[:,:,:-ignored, :-ignored].transpose(0,1) if ignored else input.transpose(0,1)
    dL_dy_aug = augment(dL_dy, nzeros=stride-1, padding=0).transpose(0,1)
    dL_df = conv2d(xt, dL_dy_aug).transpose(0,1)
    return dL_dx, dL_df


def tconv_backward(input, dL_dy, weight, stride=1, padding=0, dilation=1):
    dL_dx = conv2d(dL_dy, weight, stride=stride, padding=padding, dilation=dilation)
    
    ignored = int(input.shape[-1]-dL_dx.shape[-1])
    if ignored:
        dL_dx = pad(dL_dx, (0,ignored,0,ignored))
        #dL_dx = unfold(dL_dx, 1, padding=ignored).reshape(*dL_dx.shape[:2],*[dL_dx.shape[-1]+2*ignored]*2)
        #dL_dx = dL_dx[:,:,ignored:, ignored:]

    xt = input[:,:,:-ignored, :-ignored].transpose(0,1) if ignored else input.transpose(0,1)
    xt = augment(xt, nzeros=stride-1, padding=0)
    dL_df = conv2d(dL_dy.transpose(0,1), xt).transpose(0,1)
    return dL_dx, dL_df






#==================================================================================================================#
#==================================================================================================================#
#                                                   MODULES
#==================================================================================================================#
#==================================================================================================================#

class Module(object):
    def __init__(self):
        self.weight   = torch.Tensor([])
        self.bias     = torch.Tensor([])
        self.d_weight = torch.Tensor([])
        self.d_bias   = torch.Tensor([])
        pass
    def forward (self,input) :
        raise NotImplementedError
    def forward_and_vjp (self, input):
        raise NotImplementedError
    def param (self) :
        return self.parameters


class ReLU(Module):
    def __init__(self):
        super().__init__()
        return 

    def forward(self, input):
        return torch.relu(input)#torch.threshold(input,0,0)
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
        def _vjp(dL_dy):
            dL_dx, dL_df = tconv_backward(input, dL_dy, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation)
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



#==================================================================================================================#
#==================================================================================================================#
#                                               SEQUENTIAL CONTAINER
#==================================================================================================================#
#==================================================================================================================#


class Sequential():
    def __init__(self, *args, initialize=True):
        self._modules: Dict[str, Optional['Module']] = OrderedDict()
        self.nb_modules = len(args)

        for idx, module in enumerate(args):
            self._add_module(str(idx), module)

        if initialize: self._initialize()
        self.set_state_dict()

    def _add_module(self, name, module):
        self._modules[name] = module

    def _initialize(self):
        for module in self._modules.values():
            _init_weights(module)
        return 

    def parameters(self):
        for i, module in enumerate(self._modules.values()):
            if len(module.weight):
                yield i, [module.weight, module.bias, module.d_weight, module.d_bias]

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


    def set_state_dict(self):
        self.state_dict : Dict[str, torch.Tensor] = OrderedDict()
        for i, p in self.parameters():
            self.state_dict[str(i)+'.weights'] = p[0]
            self.state_dict[str(i)+'.bias'] = p[1]
        return 

    def load_state_dict(self, state_dict):
        for i, p in self.parameters():
            p[0].data = state_dict[str(i)+'.weights']
            p[1].data = state_dict[str(i)+'.bias']
        return 






#==================================================================================================================#
#==================================================================================================================#
#                                                   MODEL
#==================================================================================================================#
#==================================================================================================================#


class Model():

    def __init__(self) -> None:
        #============================
        #       MODEL DEFS
        #============================
        self.stride      = 2
        self.kernel_size = 2
        self.features    = 64
        self.nb_epochs   = 10

        self.loss = MSE()
    
        conv1   = Conv2d(in_channels=3, out_channels=self.features, stride=self.stride,  kernel_size=self.kernel_size)
        conv2   = Conv2d(in_channels=self.features, out_channels=self.features, stride=self.stride,  kernel_size=self.kernel_size)
        tconv1  = TransposeConv2d(in_channels=self.features, out_channels=self.features,  stride=self.stride,  kernel_size=self.kernel_size, padding=0, dilation=1)
        tconv2  = TransposeConv2d(in_channels=self.features, out_channels=3,  stride=self.stride,  kernel_size=self.kernel_size, padding=0, dilation=1)
        relu    = ReLU()
        sigmoid = Sigmoid()

        self.net = Sequential(conv1, relu, conv2, relu, tconv1, relu, tconv2, sigmoid, initialize=True)

        #============================
        #       TRAINING DEFS
        #============================
        self.eta         = 10.
        self.gamma       = 0.
        self.params_old  = None
        self.batch_size  = 32
        self.num_epochs  = 5


    #============================
    #          FORWARD
    #============================
    def predict(self,x) -> torch.Tensor:
        #PREPROCESS
        mult = x.max()>1
        if x.dtype==torch.uint8: x = x.float()
        if mult: x = x/255.

        #SEQUENTIAL NN
        x = self.net.forward(x)
        
        #POSTPROCESS
        if mult: x = x*255.
        return x

    #============================
    #            TRAIN
    #============================
    def train(self, train_input, train_target, num_epochs=None) -> None:
        if num_epochs is not None: self.nb_epochs = num_epochs

        # pre-process
        train_target = process_dataset(train_target, normalize=True)

        #train
        for e in range(self.nb_epochs):
            for inputs, targets in zip(train_input.split(self.batch_size), train_target.split(self.batch_size)):

                dL_dy = self.loss.forward_and_vjp(targets)
                self.net.forward_and_vjp(inputs/255., dL_dy)
                self.SGD()
        return

    #============================
    #          VALIDATE
    #============================
    def validate(self, val_input, val_target):         
        denoised = self.predict(val_input)/255.
        psnr = compute_psnr(denoised, val_target)
        return psnr


    def train_and_validate(self, train_input, train_target, val_input, val_target, num_epochs=None, filename=None) -> None:
        if num_epochs is not None: self.nb_epochs = num_epochs

        # pre-process
        train_target = process_dataset(train_target, normalize=True)
        val_input = process_dataset(val_input, normalize=False)

        #train
        i=0
        for _ in range(self.nb_epochs):
            for inputs, targets in zip( train_input.split(self.batch_size), train_target.split(self.batch_size) ):

                dL_dy = self.loss.forward_and_vjp(targets)
                self.net.forward_and_vjp(inputs/255., dL_dy)
                self.SGD()

                if i%125==0:
                    psnr = self.validate(val_input, val_target)

                    if filename:
                        with open(filename, 'a') as file:
                            file.write("%d\t %.10f\n"%((i*self.batch_size), psnr))
                    else:
                        print("%d\t %.3f"%((i*self.batch_size), psnr))
                i+=1
        return


    #============================
    #             SGD
    #============================
    def SGD(self):
        for _, p in self.net.parameters():           
            #if self.params_old:
            grad_w = self.eta * p[2]
            grad_b = self.eta * p[3]
            #else:
            #    grad = self.gamma * self.params_old[1] + self.eta * p[1] 

            p[0] -= grad_w
            p[1] -= grad_b

        #self.params_old = copy.deepcopy(list(self.net.parameters()))


    #============================
    #           LOADERS
    #============================
    def save(self, filename='bestmodel.pth') -> None:
        with open(filename, "wb") as fp:
            pickle.dump(self.net.state_dict, fp)

    def load_pretrained_model(self, filename='Proj_308427_348143_000000/Miniproject_2/bestmodel.pth') -> None:
        with open(filename, "rb") as fp:
            state_dict = pickle.load(fp)
        self.net.load_state_dict(state_dict)

                

