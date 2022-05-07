#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.nn import functional as F


# In[2]:


def normalize_dataset(dataset):
    for d in dataset:
        mean = d.mean([-1,-2])
        std  = d.std([-1,-2])
        norm = torchvision.transforms.Normalize(mean, std, inplace=True)
        norm(d)
    return dataset


# In[3]:


class Encoder_Block(nn.Module):
    def __init__(self, in_channles, out_channels, conv_ksize, maxp_ksize):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channles, out_channels=out_channels,                               kernel_size=conv_ksize, padding = 'same')
        
        self.maxp = nn.MaxPool2d(kernel_size=maxp_ksize)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv(x)) #convolution
        x = self.maxp(x) #pooling
        return x
    
    
class Decoder_Block(nn.Module):
    def __init__(self, in0, in1, out1, conv_ksize):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=in0, out_channels=in1 , kernel_size=conv_ksize, padding='same')
        self.conv1 = nn.Conv2d(in_channels=in1, out_channels=out1, kernel_size=conv_ksize, padding='same')

    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=2, mode='nearest') #upsample
        x = torch.cat((x,y),dim=1) #concatenate
        x = F.leaky_relu(self.conv0(x)) #first convolution 
        x = F.leaky_relu(self.conv1(x)) #second convlution
        return x
    
    
class autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        oute = 64       # nb of channels in encoding layers
        outd = 2*oute   # nb ofchannels in middle decoding layers
        ChIm = 3        # input's nb of channels
        kers = 3        # fixed kernel size for all convolutional layers
        nb_elayers = 3  # number of encoding layers 
            
        #ENCODER
        self.conv0 = nn.Conv2d(in_channels=ChIm, out_channels=oute, kernel_size=kers, padding='same')
        self.conv1 = nn.Conv2d(in_channels=oute, out_channels=oute, kernel_size=kers, padding='same')
        eblock = Encoder_Block(in_channles=oute, out_channels=oute, conv_ksize=kers, maxp_ksize=2)
        self.eblocks = nn.ModuleList([eblock]*nb_elayers)
        
        #DECODER
        dblock0 = Decoder_Block(in0=2*oute, in1=outd, out1=outd, conv_ksize=kers)
        dblock1 = Decoder_Block(in0=outd+oute, in1=outd, out1=outd, conv_ksize=kers)
        dblock2 = Decoder_Block(in0=outd+ChIm, in1=outd//2, out1=outd//3, conv_ksize=kers)
        self.dblocks = nn.ModuleList([dblock0] + [dblock1]*(nb_elayers-2) + [dblock2])
        
        self.conv2 = nn.Conv2d(in_channels=outd//3, out_channels=ChIm, kernel_size=kers, padding='same')
        
    def forward(self, x):
        #ENCODER
        pout = [x]
        y = self.conv0(x)
        for l in (self.eblocks[:-1]):
            y = l(y)
            pout.append(y)
        y = self.eblocks[-1](y)
        y = self.conv1(y)
        
        #DECODER
        for i,l in enumerate(self.dblocks):
            y = l(y, pout[-(i+1)])
        y = self.conv2(y)
        
        return y#y3
    
    
#y  = self.conv0(x)
#y1 = self.eblocks[0](y)
#y2 = self.eblocks[1](y1)
#y3 = self.eblocks[2](y2)
#self.conv1(y3)
#
#y3 = self.dblocks[0](y3, y2)
#y3 = self.dblocks[1](y3, y1)
#y3 = self.dblocks[2](y3, x)
#y3 = self.conv2(y3)


# In[5]:


def traininig_step(model, criterion, optimizer, train_input, train_target, batch_size):
    model.train()
    for inputs, targets in zip(train_input.split(batch_size), train_target.split(batch_size)):
        output = model(inputs)
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


def validate(model, criterion, val_input, val_target, batch_size):
    model.eval()
    with torch.no_grad():          
        denoised = model(val_input)
        denoised = denoised/denoised.max()

        ground_truth = val_target
        ground_truth = ground_truth/ground_truth.max()

        mse = criterion(denoised, ground_truth).item()
        psnr = -10 * np.log10(mse + 10**-8)
    return mse, psnr


def training_protocol(nb_epochs, model, criterion, train_input, train_target, val_input, val_target, batch_size):
    #optimizer  = torch.optim.Adam(model.parameters(), lr=5e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    print("Epoch:\t Tr_Err:\t  PSNR[dB]:")
    for epoch in range(nb_epochs):
        loss = traininig_step(model, criterion, optimizer, train_input, train_target, batch_size)
        mse, psnr = validate(model, criterion, val_input, val_target, batch_size) 
        scheduler.step(mse)
        print("%d\t %.3f\t  %.3f"%(epoch, loss, psnr))
            

            


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


valid_input, valid_target = torch.load('val_data.pkl') #validation set (noise-clean)
train_input, train_target = torch.load('train_data.pkl') #test set (noise-noise)

train_in = normalize_dataset(train_input.float())
train_tg = normalize_dataset(train_target.float())

print("Vector shape: ",train_input.shape)


# In[7]:


#fig,ax = plt.subplots(2,2, figsize=(8,8))
#select = 666
#
#ax[0,0].imshow(train_input[select].permute(1,2,0), origin='upper')
#ax[0,1].imshow(train_target[select].permute(1,2,0), origin='upper')
#ax[0,0].set_title("Training input (noisy)")
#ax[0,1].set_title("Training target (noisy)")
#
#
#ax[1,0].imshow(valid_input[select].permute(1,2,0), origin='upper')
#ax[1,1].imshow(valid_target[select].permute(1,2,0), origin='upper')
#ax[1,0].set_title("Validation input (noisy)")
#ax[1,1].set_title("Validation target (clean)");


# In[ ]:





# In[8]:


model, criterion = autoencoder(), nn.MSELoss()

batch_size = 500
nb_epochs  = 10


# In[9]:


training_protocol(nb_epochs, model, criterion, train_in, train_tg,                   valid_input.float(), valid_target.float(), batch_size)


# In[ ]:





# In[10]:


denoised = model(valid_input.float()).detach()
denoised = denoised/denoised.max()
ground_truth = valid_target.float()
ground_truth = ground_truth/ground_truth.max()
noisy = valid_input.float()
noisy = noisy/noisy.max()

mse = criterion(denoised, ground_truth).item()
-10 * np.log10(mse + 10**-8)


# In[11]:


#fig,ax = plt.subplots(1,3, figsize=(15,12))
#select = 164

#ax[0].imshow(noisy[select].permute(1,2,0), origin='upper')
#ax[1].imshow(denoised[select].permute(1,2,0), origin='upper')
#ax[2].imshow(ground_truth[select].permute(1,2,0), origin='upper')

#ax[0].set_title("Validation input (noisy)")
#ax[1].set_title("Denoised input)")
#ax[2].set_title("Validation target (clean)");


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




