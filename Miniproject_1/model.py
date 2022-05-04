import torch
import torchvision
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    
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
        eblock = self._EncoderBlock(in_channles=oute, out_channels=oute, conv_ksize=kers, maxp_ksize=2)
        self.eblocks = nn.ModuleList([eblock]*nb_elayers)
        
        #DECODER
        dblock0 = self._DecoderBlock(in0=2*oute, in1=outd, out1=outd, conv_ksize=kers)
        dblock1 = self._DecoderBlock(in0=outd+oute, in1=outd, out1=outd, conv_ksize=kers)
        dblock2 = self._DecoderBlock(in0=outd+ChIm, in1=outd//2, out1=outd//3, conv_ksize=kers)
        self.dblocks = nn.ModuleList([dblock0] + [dblock1]*(nb_elayers-2) + [dblock2])
        
        self.conv2 = nn.Conv2d(in_channels=outd//3, out_channels=ChIm, kernel_size=kers, padding='same')

    class _EncoderBlock(nn.Module):
        
        def __init__(self, in_channles, out_channels, conv_ksize, maxp_ksize):
            super().__init__()
            self.conv = nn.Conv2d(in_channels=in_channles, out_channels=out_channels,\
                                   kernel_size=conv_ksize, padding = 'same')

            self.maxp = nn.MaxPool2d(kernel_size=maxp_ksize)

        def forward(self, x):
            x = F.leaky_relu(self.conv(x)) #convolution
            x = self.maxp(x) #pooling
            return x
    
    
    class _DecoderBlock(nn.Module):
        
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
    
        
    def predict(self, x):
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

    # TODO: implement this structure
    # def train(self, train ̇input, train ̇target, num ̇epochs) -> None:
        # :train ̇input: tensor of size (N, C, H, W) containing a noisy version of the images
        
        #:train ̇target: tensor of size (N, C, H, W) containing another noisy version of the
        # same images, which only differs from the input by their noise.


    def train(self, criterion, optimizer, train_input, train_target, batch_size):

        self.train()
        for inputs, targets in zip(train_input.split(batch_size), train_target.split(batch_size)):
            output = self(inputs)
            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss

    def validate(self, criterion, val_input, val_target, batch_size):

        self.eval()
        with torch.no_grad():          
            denoised = self(val_input)
            denoised = denoised/denoised.max()

            ground_truth = val_target
            ground_truth = ground_truth/ground_truth.max()

            mse = criterion(denoised, ground_truth).item()
            psnr = -10 * torch.log10(mse + 10**-8)
        return mse, psnr

    def load_pretrained_model(self) -> None:
        ## This loads the parameters saved in bestmodel.pth into the model
        # TODO: implement
        pass

    def predict(self, test_input) -> torch.Tensor:
        #:test ̇input: tensor of size (N1, C, H, W) that has to be denoised by the trained or the loaded network.
        #: returns a tensor of the size (N1, C, H, W)
        pass
    
    
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