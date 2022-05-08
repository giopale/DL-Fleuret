from re import L
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import time
import sys
import argparse


from Miniproject_1.model import Model

# Print to file
timestr = time.strftime("%Y%m%d-%H%M%S")
filename=f'NetTest-{timestr}.txt'
def myprint(string):
    with open(filename, 'a') as file:
        file.write('\n')
        file.write(string)

    # sys.stdout = f # Change the standard output to the file we created.

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--train-samples', dest='train_samples', type=int, help='Number of train samples to use')
parser.add_argument('--batch-size', dest='batch_size', type=int, help='Batch size to use')
parser.add_argument('--epochs', dest='epochs', type=int, help='Number of epochs')
parser.add_argument('--param-file', dest='param_file', type=str, help='Name of the file where parameters are saved')
args = parser.parse_args()




# Print message
with open(filename, 'a') as file:
    file.write('########################################################'+'\n')
    file.write('####################### Network test ###################'+'\n')
    file.write('########################################################'+'\n')
    file.write(''+'\n')
    file.write('date & time: {0}'.format(time.strftime("%Y/%m/%d-%H:%M:%S"))+'\n')



# Istantiate network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Model().to(device)


#Configuration
# Parse command arguments

config_net={
    # Net params
    'oute' : 64,                                # nb of channels in encoding layers
    'outd' : 2*64,                              # nb ofchannels in middle decoding layers
    'ChIm' : 3,                                 # input's nb of channels
    'kers' : 3,                                 # fixed kernel size for all convolutional layers
    'nb_elayers' : 4,                           # number of encoding layers 
}
for key, val in config_net.items(): setattr(net,key,val)

config_train={
    # Training 
    'n_train_samples': args.train_samples ,
    'batch_size': 32 if args.batch_size is None else args.batch_size,
    'criterion' : nn.MSELoss(),
    'n_epochs'  : 5 if args.epochs is None else args.epochs,
    # optimizer -> set from model.py
    # scheduler -> set form model.py
}
net.criterion  = config_train['criterion']
net.batch_size = config_train['batch_size']

with open(filename, 'a') as file:
    file.write(''+'\n')
    file.write('####################### Configuration ##################'+'\n')
    file.write(''+'\n')
    file.write('Net parameters:'+'\n')
    for key,val in config_net.items():
        file.write('-> '+str(key)+' : '+str(val)+'\n')
    file.write('')
    file.write('Training parameters:'+'\n')
    for key,val in config_train.items():
        file.write('-> '+str(key)+' : '+str(val)+'\n')
    file.write('')


# Load dataset or portion of it
valid_input, valid_target = torch.load('val_data.pkl',map_location=device)#validation set (noise-clean)
train_input, train_target = torch.load('train_data.pkl',map_location=device) #test set (noise-noise)

valid_input  = valid_input.float() / 255.
valid_target = valid_target.float()/ 255.
train_input  = train_input.float() / 255.
train_target = train_target.float()/ 255.

num_samples = config_train['n_train_samples']
if num_samples is not None:
    train_input  = train_input[:num_samples]
    train_target = train_target[:num_samples]

# Training
with open(filename, 'a') as file:
    file.write(''+'\n')
    file.write('####################### Training #######################'+'\n')
    file.write(''+'\n')

t_train=time.time()
net.train(train_input, train_target, config_train['n_epochs'], valid_input, valid_target, filename)
t_train=time.time()-t_train

file_params=args.param_file
if file_params is not None:
    net.save(file_params)


with open(filename, 'a') as file:
    file.write('\n')
    file.write('Time:'+'\n')
    file.write('-> elapsed:\t{0:.1f} s'.format(t_train)+'\n')
    file.write('-> per epoch:\t{0:.1f} s'.format(t_train/config_train['n_epochs'])+'\n')



# Print source code
with open(filename, 'a') as file:
    file.write('\n' * 20)
    file.write('####################### Source code ####################')
    file.write('')
    source = open('Miniproject_1/model.py', 'r')
    content = source.read()
    file.write(content)
    source.close()


# f.close()
