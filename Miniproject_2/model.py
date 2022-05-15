# Miniproject 2

from torch import empty , cat , arange
from torch.nn.functional import fold, unfold

torch.set_grad_enabled(False)


class Module (object) :
    def forward (self, *input) :
        raise NotImplementedError
    def backward (self, *gradwrtoutput):
        raise NotImplementedError
    def param (self) :
        return []