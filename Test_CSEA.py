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

def normalize_dataset(dataset):
    for d in dataset:
        mean = d.mean([-1,-2])
        std  = d.std([-1,-2])
        norm = torchvision.transforms.Normalize(mean, std, inplace=True)
        norm(d)
    return dataset

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--train-samples', dest='train_samples',\
                         type=int, help='Number of train samples to use')
parser.add_argument('--batch-size', dest='batch_size',
                         type=int, help='Batch size to use')
parser.add_argument('--epochs', dest='epochs',
                         type=int, help='Number of epochs')
parser.add_argument('--param-file', dest='param_file', type=str, \
                help='Name of the file where parameters are saved')
args = parser.parse_args()




# Print to file
timestr = time.strftime("%Y%m%d-%H%M%S")
filename=f'NetTest-{timestr}.txt'
f=open(filename, 'w')
sys.stdout = f # Change the standard output to the file we created.

# Print message
print('########################################################')
print('####################### Network test ###################')
print('########################################################')
print('')
print('date & time: {0}'.format(time.strftime("%Y/%m/%d-%H:%M:%S")))

# Istantiate network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Model().to(device)

#Configuration

# Parse command arguments

config_net={
    # Net params
    'oute' : 64,                                # nb of channels in encoding layers
    'outd' : 2*64,                            # nb ofchannels in middle decoding layers
    'ChIm' : 3,                                 # input's nb of channels
    'kers' : 3,                                 # fixed kernel size for all convolutional layers
    'nb_elayers' : 3,                           # number of encoding layers 
    'stride_maxpool': None,                        # stride maxpooling layer

}
for key, val in config_net.items():
    setattr(net,key,val)

config_train={
    # Training 
    'n_train_samples' : None if args.train_samples is None else args.train_samples ,
    'batch_size': 32 if args.batch_size is None else args.batch_size,
    'criterion': nn.MSELoss(),
    'n_epochs' : 5 if args.epochs is None else args.epochs,
    # optimizer -> set from model.py
    # scheduler -> set form model.py

}
net.criterion=config_train['criterion']
net.batch_size=config_train['batch_size']


print('')
print('####################### Configuration ##################')
print('')
print('Net parameters:')
for key,val in config_net.items():
    print('-> ',key,' : ', val)
print('')
print('Training parameters:')
for key,val in config_train.items():
    print('-> ',key,' : ', val)
print('')


# Load dataset or portion of it
valid_input, valid_target = torch.load('val_data.pkl',map_location=device)#validation set (noise-clean)
train_input, train_target = torch.load('train_data.pkl',map_location=device) #test set (noise-noise)

num_samples = config_train['n_train_samples']
if num_samples is not None:
    valid_input=torch.narrow(valid_input,0,0,num_samples)
    valid_target=torch.narrow(valid_target,0,0,num_samples)
    train_input=torch.narrow(train_input,0,0,num_samples)
    train_target=torch.narrow(train_target,0,0,num_samples)

train_in = normalize_dataset(train_input.float())
train_tg = normalize_dataset(train_target.float())
valid_in = normalize_dataset(valid_input.float())
valid_tg = normalize_dataset(valid_target.float())

# Training

print('')
print('####################### Training #######################')
print('')
train_start=time.time()

# ########### The real training happens here ###############

net.train_and_validate(train_in, train_tg, \
    config_train['n_epochs'], valid_in, valid_tg)

filename=args.param_file
if filename is not None:
    net.save(filename)



train_end=time.time()
print('')
print('Time:')
print('-> elapsed:\t{0:.1f} s'.format( train_end-train_start))
print('-> per epoch:\t{0:.1f} s'.format( \
    (train_end-train_start)/config_train['n_epochs']))








# Print source code
print(print('\n' * 20))
print('####################### Source code ####################')
print('')
source = open('Miniproject_1/model.py', 'r')
content = source.read()
print(content)
source.close()


f.close()
