from numpy import iinfo
import torch
from torch import nn
from torch.nn import functional as F

    
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

def process_dataset(data, normalize=False):
    if data.dtype==torch.uint8: data = data.float()
    if data.max()>1 and normalize: data = data/255.
    return data


def compute_psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x-y) ** 2).mean((1,2,3))).mean()


def _init_weights(model):
    if isinstance(model,nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_in', nonlinearity='leaky_relu')


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




#==================================================================================================================#
#==================================================================================================================#
#                                               NETWORK
#==================================================================================================================#
#==================================================================================================================#

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        #============================
        #       MODEL DEFS
        #============================
        oute       = 32
        outd       = 2*oute   # nb ofchannels in middle decoding layers
        ChIm       = 3        # input's nb of channels
        kers       = 3        # fixed kernel size for all convolutional layers
        nb_elayers = 4        # number of encoding layers 

        self.num_epochs = 1
            
        #ENCODER
        self.conv0 = nn.Conv2d(in_channels=ChIm, out_channels=oute, kernel_size=kers, padding='same')
        self.conv1 = nn.Conv2d(in_channels=oute, out_channels=oute, kernel_size=kers, padding='same')
        eblock = _Encoder_Block(in_channles=oute, out_channels=oute, conv_ksize=kers, maxp_ksize=2)
        self.eblocks = nn.ModuleList([eblock]*nb_elayers)
        
        
        #DECODER
        dblock0 = _Decoder_Block(in0=2*oute, in1=outd, out1=outd, conv_ksize=kers)
        dblock1 = _Decoder_Block(in0=outd+oute, in1=outd, out1=outd, conv_ksize=kers)
        dblock2 = _Decoder_Block(in0=outd+ChIm, in1=outd//2, out1=outd//3, conv_ksize=kers)
        self.dblocks = nn.ModuleList([dblock0] + [dblock1]*(nb_elayers-2) + [dblock2])
        
        self.conv2 = nn.Conv2d(in_channels=outd//3, out_channels=ChIm, kernel_size=kers, padding='same')
        self.relu  = nn.ReLU() #nn.LeakyReLU(inplace=True)

        # WEIGHTS INIT
        # self.apply(_init_weights)
        

        #============================
        #       TRAINING DEFS
        #============================

        self.criterion  = nn.MSELoss()
        self.batch_size = 16

        self.eta        = 0.1
        self.momentum   = 0.9
        self.weight_decay = 0.
        self.optimizer  = torch.optim.SGD(self.parameters(), lr=self.eta, momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    def predict(self, x):
        #PROCESSING
        mult = x.max()>1
        if x.dtype==torch.uint8: x = x.float()
        if mult: x = x/255.

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
        
        #POSTPROCESSING
        if mult: y = y*255.
        return y


#==================================================================================================================#
#==================================================================================================================#
#                                               TRAINING
#==================================================================================================================#
#==================================================================================================================#


    #============================
    #           TRAIN
    #============================

    def train(self, train_input, train_target, num_epochs=None) -> None:
        if num_epochs is not None: self.num_epochs = num_epochs

        # pre-process
        standardize_dataset(train_input , method='per_image')
        standardize_dataset(train_target, method='per_image')
        train_target = process_dataset(train_target, normalize=False)

        #train
        for epoch in range(self.num_epochs):
            for inputs, targets in zip(train_input.split(self.batch_size), train_target.split(self.batch_size)):
                output = self.predict(inputs)
                loss   = self.criterion(output/255., targets/255.).requires_grad_()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    #============================
    #           VALIDATE                                            
    #============================        

    def validate(self, val_input, val_target):
        with torch.no_grad():          
            denoised = self.predict(val_input)/255.
            mse = F.mse_loss(denoised, val_target)
            psnr = compute_psnr(denoised, val_target)
        return mse, psnr


    def train_and_validate(self, train_input, train_target, val_input, val_target, num_epochs=None, filename=None) -> None:
        if num_epochs is not None: self.num_epochs = num_epochs

        # pre-process
        standardize_dataset(train_input , method='per_image')
        standardize_dataset(train_target, method='per_image')
        train_target = process_dataset(train_target, normalize=False)
        val_target   = process_dataset(val_target, normalize=True)

        #train
        i=0
        for _ in range(self.num_epochs):
            for inputs, targets in zip(train_input.split(self.batch_size), train_target.split(self.batch_size)):
                output = self.predict(inputs)
                loss   = self.criterion(output/255., targets/255.).requires_grad_()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i%125==0:
                    mse, psnr = self.validate(val_input, val_target)
                    self.scheduler.step(mse)

                    if filename:
                        with open(filename, 'a') as file:
                            file.write("%d\t %.10f\t %.10f\n"%((i*self.batch_size), mse, psnr))
                    else:
                        print("%d\t %.3f\t %.3f"%((i*self.batch_size), mse, psnr))
                i+=1



#==================================================================================================================#
#==================================================================================================================#
#                                               LOADERS
#==================================================================================================================#
#==================================================================================================================#


    def save(self, filename) -> None:
        torch.save(self.state_dict(), filename)

    def load_pretrained_model(self, filename='Proj_308427_348143_XXXXXX/Miniproject_1/bestmodel.pth') -> None:
        self.load_state_dict(torch.load(filename))

