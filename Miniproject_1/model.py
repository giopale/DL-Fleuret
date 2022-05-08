import torch
import torchvision
from torch import nn
from torch.nn import functional as F
    
    
def standardize_dataset(dataset, method='per_image'):
    if dataset.dtype!=torch.float: dataset=dataset.float()
    if method=='per_image':
        mu  = dataset.mean((-1,-2)).view([*dataset.shape[:2],1,1])
        std = dataset.std((-1, -2)).view([*dataset.shape[:2],1,1])
        dataset.sub_(mu).div_(std)
    else:
        mu  = dataset.mean(0)
        std = dataset.std(0)
        dataset.sub_(mu).div_(std)
    return 


class Model(nn.Module):

#==================================================================================================================#
#==================================================================================================================#
#                                               NETWORK
#==================================================================================================================#
#==================================================================================================================#

    class _Encoder_Block(nn.Module):
        def __init__(self, in_channles, out_channels, conv_ksize, maxp_ksize):
            super().__init__()
            self.conv = nn.Conv2d(in_channels=in_channles, out_channels=out_channels,\
                                kernel_size=conv_ksize, padding='same')
            
            self.maxp = nn.MaxPool2d(kernel_size=maxp_ksize)
            self.relu = nn.PReLU(out_channels) #nn.LeakyReLU(inplace=True)
            
        def forward(self, x):
            x = self.relu(self.conv(x)) #convolution
            x = self.maxp(x)            #pooling 
            return x
    
    
    class _Decoder_Block(nn.Module):
        def __init__(self, in0, in1, out1, conv_ksize):
            super().__init__()
            self.conv0 = nn.Conv2d(in_channels=in0, out_channels=in1 , kernel_size=conv_ksize, padding='same')
            self.conv1 = nn.Conv2d(in_channels=in1, out_channels=out1, kernel_size=conv_ksize, padding='same')
            self.relu0 = nn.PReLU(in1) #nn.LeakyReLU(inplace=True)
            self.relu1 = nn.PReLU(out1)
            
        def forward(self, x, y):
            x = F.interpolate(x, scale_factor=2, mode='nearest') #upsample
            x = torch.cat((x,y),dim=1)    #concatenate
            x = self.relu0(self.conv0(x)) #first convolution 
            x = self.relu1(self.conv1(x)) #second convlution
            return x
            
            
    def __init__(self):
        super().__init__()

        #============================
        #       MODEL DEFS
        #============================

        oute = 64       # nb of channels in encoding layers
        outd = 2*oute   # nb ofchannels in middle decoding layers
        ChIm = 3        # input's nb of channels
        kers = 3        # fixed kernel size for all convolutional layers
        nb_elayers = 4  # number of encoding layers 
            
        #ENCODER
        self.conv0 = nn.Conv2d(in_channels=ChIm, out_channels=oute, kernel_size=kers, padding='same')
        self.conv1 = nn.Conv2d(in_channels=oute, out_channels=oute, kernel_size=kers, padding='same')
        eblock = self._Encoder_Block(in_channles=oute, out_channels=oute, conv_ksize=kers, maxp_ksize=2)
        self.eblocks = nn.ModuleList([eblock]*nb_elayers)
        
        
        #DECODER
        dblock0 = self._Decoder_Block(in0=2*oute, in1=outd, out1=outd, conv_ksize=kers)
        dblock1 = self._Decoder_Block(in0=outd+oute, in1=outd, out1=outd, conv_ksize=kers)
        dblock2 = self._Decoder_Block(in0=outd+ChIm, in1=outd//2, out1=outd//3, conv_ksize=kers)
        self.dblocks = nn.ModuleList([dblock0] + [dblock1]*(nb_elayers-2) + [dblock2])
        
        self.conv2 = nn.Conv2d(in_channels=outd//3, out_channels=ChIm, kernel_size=kers, padding='same')
        self.relu  = nn.PReLU() #nn.LeakyReLU(inplace=True)
        

        #============================
        #       TRAINING DEFS
        #============================

        self.criterion  = nn.MSELoss()
        self.batch_size = 32
        self.optimizer  = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
        self.scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.do_print   = True


    def predict(self, x):
        #ENCODER
        pout = [x]
        y = self.relu(self.conv0(x))

        for l in self.eblocks[:-1]:
            y = l(y)
            pout.append(y)
        y = self.eblocks[-1](y)
        y = self.relu(self.conv1(y))
        
        #DECODER
        for i,l in enumerate(self.dblocks):
            y = l(y, pout[-(i+1)])
        y = torch.sigmoid(self.conv2(y))
        return y



#==================================================================================================================#
#==================================================================================================================#
#                                               TRAINING
#==================================================================================================================#
#==================================================================================================================#


    #============================
    #           TRAIN
    #============================

    def train(self, train_input, train_target, num_epochs, val_input, val_target, filename=None) -> None:
        if self.do_print: 
            if filename is not None:
                with open(filename, 'a') as file:
                    file.write('Training on {0} epochs:'.format(num_epochs)+'\n')
                    file.write("Epoch:\t Tr_Err:\t  PSNR[dB]:"+'\n\n')

        for epoch in range(num_epochs):
            for inputs, targets in zip(train_input.split(self.batch_size),\
                                            train_target.split(self.batch_size)):
                output = self.predict(inputs)
                loss   = self.criterion(output, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            mse, psnr = self.validate(val_input, val_target)
            self.scheduler.step(mse)

            if filename is not None:
                with open(filename, 'a') as file:
                    file.write("%d\t %.3f\t  %.3f"%(epoch, loss, psnr)+'\n')

    #============================
    #           VALIDATE                                            
    #============================        

    def validate(self, val_input, val_target):
        with torch.no_grad():          
            denoised = self.predict(val_input)
            mse = F.mse_loss(denoised, val_target)
            psnr = (-10 * torch.log10(mse + 10**-8)).item()
        return mse, psnr


#==================================================================================================================#
#==================================================================================================================#
#                                               LOADERS
#==================================================================================================================#
#==================================================================================================================#


    def load_pretrained_model(self) -> None:
        ## This loads the parameters saved in bestmodel.pth into the model
        # TODO: implement
        pass

    def save(self, filename) -> None:
        torch.save(self.state_dict(), filename)

    def load(self, filename) -> None:
        new_model = self.torch.load(filename)
        self=new_model
