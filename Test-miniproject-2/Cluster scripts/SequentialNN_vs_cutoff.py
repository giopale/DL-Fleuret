from numpy import iinfo
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import argparse

    
def standardize_dataset(dataset, method='per_image'):
    if dataset.dtype!=torch.float: dataset=dataset.float()
    if method=='per_image':
        mu  = dataset.mean((-1,-2)).view([*dataset.shape[:2],1,1])
        std = dataset.std((-1, -2)).view([*dataset.shape[:2],1,1])
        dataset.data.sub_(mu).div_(std)
    else:
        mu  = dataset.mean(0)
        std = dataset.std(0)
        dataset.data.sub_(mu).div_(std)
    return 

        
        
def validate(model, val_input, val_target):
        with torch.no_grad():          
            denoised = model(val_input)
            mse = F.mse_loss(denoised, val_target)
            psnr = (-10 * torch.log10(mse + 10**-8)).item()
        return psnr


def train_nn(model, criterion, train_input, train_target, mini_batch_size, nb_epochs, eta=0.75, val_input=None, val_target=None, filename=None):
    optimizer = torch.optim.SGD(model.parameters(), lr=eta, )
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    i = 0
    for e in range(nb_epochs):
        for inputs, targets in zip(train_input.split(mini_batch_size), train_target.split(mini_batch_size)):
            output = model(inputs)
            loss = criterion(output, targets)

            model.zero_grad()
            loss.backward()

            optimizer.step()
        
            if val_input is not None and val_target is not None and i%125==0:
                mse, psnr = validate(val_input, val_target)
                scheduler.step(mse)

                if filename:
                    with open(filename, 'a') as file:
                        file.write("%d\t %.10f\t %.10f\n"%((i*mini_batch_size), mse, psnr))

        i+=1
        #print("\rCompleted: %d/%d"%(e+1,nb_epochs), end=' ')
    return 



parser = argparse.ArgumentParser()
parser.add_argument('--Njob', dest='Njob', type=float, help='Add Njob')
args = parser.parse_args()
Njob = int(args.Njob)

eta_list = np.array([1e-3, 1e-2, 1e-1, 5e-1, 0.75, 1, 1.5])


stride = ks = 2
outch  = 64

conv1  = nn.Conv2d(in_channels=3, out_channels=outch,  kernel_size=ks, stride=stride, bias=True)
conv2  = nn.Conv2d(in_channels=outch, out_channels=outch,  kernel_size=ks, stride=stride, bias=True)

tconv1 = nn.ConvTranspose2d(in_channels=outch, out_channels=outch,  kernel_size=ks,\
                            stride=stride, padding=0, dilation=1, bias=True)
tconv2 = nn.ConvTranspose2d(in_channels=outch, out_channels=3,  kernel_size=ks,\
                            stride=stride, padding=0, dilation=1, bias=True)

relu      = nn.ReLU()
sigmoid   = nn.Sigmoid()
criterion = nn.MSELoss()


Net = nn.Sequential(conv1, relu, conv2, relu, tconv1, relu, tconv2, sigmoid)


cut=50000
train_in, train_tg = torch.load('../train_data.pkl')
train_in = train_in[:cut].float()/255.
train_tg = train_tg[:cut].float()/255.

val_in, val_tg = torch.load('../val_data.pkl')
val_in = val_in.float()/255.
val_tg = val_tg.float()/255.

bs  = 16
ne  = 6
eta = eta_list[Njob]

filename = "sequential_cut=%d_nbepochs=%d_eta=%.6f_.txt"%(cut,ne,eta)
train_nn(Net, criterion, train_in, train_tg, bs, ne, eta=eta, val_input=val_in, val_target=val_tg, filename=filename)