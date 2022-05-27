import torch
import numpy as np
import argparse

from torch import nn
from torch.nn import functional as F

from model import *


eta_list = np.array([1e-3, 1e-2, 1e-1, 1e-0, 1e1, 1e2, 1e3, 1e4])

parser = argparse.ArgumentParser()
parser.add_argument('--Njob', dest='Njob', type=float, help='Add Njob')
args = parser.parse_args()
Njob = int(args.Njob)


valid_in, valid_tg = torch.load('val_data.pkl') #validation set (noise-clean)
train_in, train_tg = torch.load('train_data.pkl')


cut = 50000
valid_tg = valid_tg.float()/ 255.
train_in = train_in[:cut]
train_tg = train_tg[:cut]


mod = Model()

mod.num_epochs = 3
mod.batch_size = 16
mod.eta        = eta_list[Njob]
mod.momentum   = 0.9
mod.weight_decay = 0.

filename = "psnr_lr=%.6f_cut=%d_nbepochs=%d_bs=16_.txt"%(mod.eta, cut, mod.num_epochs)
mod.train(train_in, train_tg, val_input=valid_in, val_target=valid_tg, filename=filename)