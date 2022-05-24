import torch
import numpy as np
import argparse

from torch import nn
from torch.nn import functional as F

from model import *


cut_list = np.array([4e3, 8e3, 10e3, 16e3, 20e3, 30e3, 50e3])
epoch_list = np.array(150000//cut_list, dtype=int)


parser = argparse.ArgumentParser()
parser.add_argument('--Njob', dest='Njob', type=float, help='Add Njob')
args = parser.parse_args()
Njob = int(args.Njob)


valid_in, valid_tg = torch.load('val_data.pkl') #validation set (noise-clean)
train_in, train_tg = torch.load('train_data.pkl')


cut = int(cut_list[Njob])
valid_in = valid_in.float()/ 255.
valid_tg = valid_tg.float()/ 255.
train_in = train_in[:cut].float()/ 255.
train_tg = train_tg[:cut].float()/ 255.


mod = Model()

mod.num_epochs = epoch_list[Njob]
mod.batch_size = 16
mod.eta        = 0.1
mod.momentum   = 0.9
mod.weight_decay = 0.0005

filename = "psnr_cut=%d_nbepochs=%d_bs=16_.txt"%(cut, mod.num_epochs)
mod.train(train_in, train_tg, val_input=valid_in, val_target=valid_tg, filename=filename)