{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2cd21fc0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torchvision\n",
    "\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from torch import nn\n",
    "from torch.nn import functional as F"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "2fda357d",
   "metadata": {},
   "outputs": [],
   "source": [
    "def normalize_dataset(dataset):\n",
    "    for d in dataset:\n",
    "        mean = d.mean([-1,-2])\n",
    "        std  = d.std([-1,-2])\n",
    "        norm = torchvision.transforms.Normalize(mean, std, inplace=True)\n",
    "        norm(d)\n",
    "    return dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0cfafba6",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Encoder_Block(nn.Module):\n",
    "    def __init__(self, in_channles, out_channels, conv_ksize, maxp_ksize):\n",
    "        super().__init__()\n",
    "        self.conv = nn.Conv2d(in_channels=in_channles, out_channels=out_channels,\\\n",
    "                               kernel_size=conv_ksize, padding = 'same')\n",
    "        \n",
    "        self.maxp = nn.MaxPool2d(kernel_size=maxp_ksize)\n",
    "        \n",
    "    def forward(self, x):\n",
    "        x = F.leaky_relu(self.conv(x)) #convolution\n",
    "        x = self.maxp(x) #pooling\n",
    "        return x\n",
    "    \n",
    "    \n",
    "class Decoder_Block(nn.Module):\n",
    "    def __init__(self, in0, in1, out1, conv_ksize):\n",
    "        super().__init__()\n",
    "        self.conv0 = nn.Conv2d(in_channels=in0, out_channels=in1 , kernel_size=conv_ksize, padding='same')\n",
    "        self.conv1 = nn.Conv2d(in_channels=in1, out_channels=out1, kernel_size=conv_ksize, padding='same')\n",
    "\n",
    "    def forward(self, x, y):\n",
    "        x = F.interpolate(x, scale_factor=2, mode='nearest') #upsample\n",
    "        x = torch.cat((x,y),dim=1) #concatenate\n",
    "        x = F.leaky_relu(self.conv0(x)) #first convolution \n",
    "        x = F.leaky_relu(self.conv1(x)) #second convlution\n",
    "        return x\n",
    "    \n",
    "    \n",
    "class autoencoder(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        oute = 64       # nb of channels in encoding layers\n",
    "        outd = 2*oute   # nb ofchannels in middle decoding layers\n",
    "        ChIm = 3        # input's nb of channels\n",
    "        kers = 3        # fixed kernel size for all convolutional layers\n",
    "        nb_elayers = 3  # number of encoding layers \n",
    "            \n",
    "        #ENCODER\n",
    "        self.conv0 = nn.Conv2d(in_channels=ChIm, out_channels=oute, kernel_size=kers, padding='same')\n",
    "        self.conv1 = nn.Conv2d(in_channels=oute, out_channels=oute, kernel_size=kers, padding='same')\n",
    "        eblock = Encoder_Block(in_channles=oute, out_channels=oute, conv_ksize=kers, maxp_ksize=2)\n",
    "        self.eblocks = nn.ModuleList([eblock]*nb_elayers)\n",
    "        \n",
    "        #DECODER\n",
    "        dblock0 = Decoder_Block(in0=2*oute, in1=outd, out1=outd, conv_ksize=kers)\n",
    "        dblock1 = Decoder_Block(in0=outd+oute, in1=outd, out1=outd, conv_ksize=kers)\n",
    "        dblock2 = Decoder_Block(in0=outd+ChIm, in1=outd//2, out1=outd//3, conv_ksize=kers)\n",
    "        self.dblocks = nn.ModuleList([dblock0] + [dblock1]*(nb_elayers-2) + [dblock2])\n",
    "        \n",
    "        self.conv2 = nn.Conv2d(in_channels=outd//3, out_channels=ChIm, kernel_size=kers, padding='same')\n",
    "        \n",
    "    def forward(self, x):\n",
    "        #ENCODER\n",
    "        pout = [x]\n",
    "        y = self.conv0(x)\n",
    "        for l in (self.eblocks[:-1]):\n",
    "            y = l(y)\n",
    "            pout.append(y)\n",
    "        y = self.eblocks[-1](y)\n",
    "        y = self.conv1(y)\n",
    "        \n",
    "        #DECODER\n",
    "        for i,l in enumerate(self.dblocks):\n",
    "            y = l(y, pout[-(i+1)])\n",
    "        y = self.conv2(y)\n",
    "        \n",
    "        return y#y3\n",
    "    \n",
    "    \n",
    "#y  = self.conv0(x)\n",
    "#y1 = self.eblocks[0](y)\n",
    "#y2 = self.eblocks[1](y1)\n",
    "#y3 = self.eblocks[2](y2)\n",
    "#self.conv1(y3)\n",
    "#\n",
    "#y3 = self.dblocks[0](y3, y2)\n",
    "#y3 = self.dblocks[1](y3, y1)\n",
    "#y3 = self.dblocks[2](y3, x)\n",
    "#y3 = self.conv2(y3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f8332d5b",
   "metadata": {},
   "outputs": [],
   "source": [
    "def traininig_step(model, criterion, optimizer, train_input, train_target, batch_size):\n",
    "    model.train()\n",
    "    for inputs, targets in zip(train_input.split(batch_size), train_target.split(batch_size)):\n",
    "        output = model(inputs)\n",
    "        loss = criterion(output, targets)\n",
    "\n",
    "        optimizer.zero_grad()\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "    return loss\n",
    "\n",
    "\n",
    "def validate(model, criterion, val_input, val_target, batch_size):\n",
    "    model.eval()\n",
    "    with torch.no_grad():          \n",
    "        denoised = model(val_input)\n",
    "        denoised = denoised/denoised.max()\n",
    "\n",
    "        ground_truth = val_target\n",
    "        ground_truth = ground_truth/ground_truth.max()\n",
    "\n",
    "        mse = criterion(denoised, ground_truth).item()\n",
    "        psnr = -10 * np.log10(mse + 10**-8)\n",
    "    return mse, psnr\n",
    "\n",
    "\n",
    "def training_protocol(nb_epochs, model, criterion, train_input, train_target, val_input, val_target, batch_size):\n",
    "    #optimizer  = torch.optim.Adam(model.parameters(), lr=5e-4)\n",
    "    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)\n",
    "    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')\n",
    "    \n",
    "    print(\"Epoch:\\t Tr_Err:\\t  PSNR[dB]:\")\n",
    "    for epoch in range(nb_epochs):\n",
    "        loss = traininig_step(model, criterion, optimizer, train_input, train_target, batch_size)\n",
    "        mse, psnr = validate(model, criterion, val_input, val_target, batch_size) \n",
    "        scheduler.step(mse)\n",
    "        print(\"%d\\t %.3f\\t  %.3f\"%(epoch, loss, psnr))\n",
    "            \n",
    "\n",
    "            "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a9e7748e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "235613e9",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "33c03022",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "bd08dda7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Vector shape:  torch.Size([50000, 3, 32, 32])\n"
     ]
    }
   ],
   "source": [
    "valid_input, valid_target = torch.load('val_data.pkl') #validation set (noise-clean)\n",
    "train_input, train_target = torch.load('train_data.pkl') #test set (noise-noise)\n",
    "\n",
    "train_in = normalize_dataset(train_input.float())\n",
    "train_tg = normalize_dataset(train_target.float())\n",
    "\n",
    "print(\"Vector shape: \",train_input.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "f28dc304",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x576 with 4 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig,ax = plt.subplots(2,2, figsize=(8,8))\n",
    "select = 666\n",
    "\n",
    "ax[0,0].imshow(train_input[select].permute(1,2,0), origin='upper')\n",
    "ax[0,1].imshow(train_target[select].permute(1,2,0), origin='upper')\n",
    "ax[0,0].set_title(\"Training input (noisy)\")\n",
    "ax[0,1].set_title(\"Training target (noisy)\")\n",
    "\n",
    "\n",
    "ax[1,0].imshow(valid_input[select].permute(1,2,0), origin='upper')\n",
    "ax[1,1].imshow(valid_target[select].permute(1,2,0), origin='upper')\n",
    "ax[1,0].set_title(\"Validation input (noisy)\")\n",
    "ax[1,1].set_title(\"Validation target (clean)\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "004543d9",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "cc8613c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "model, criterion = autoencoder(), nn.MSELoss()\n",
    "\n",
    "batch_size = 500\n",
    "nb_epochs  = 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "d58d4284",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch:\t Tr_Err:\t  PSNR[dB]:\n",
      "0\t 0.313\t  17.234\n",
      "1\t 0.307\t  20.102\n",
      "2\t 0.305\t  21.132\n",
      "3\t 0.303\t  21.891\n",
      "4\t 0.306\t  22.116\n",
      "5\t 0.301\t  22.554\n",
      "6\t 0.301\t  22.984\n",
      "7\t 0.300\t  23.066\n",
      "8\t 0.300\t  22.991\n",
      "9\t 0.299\t  22.845\n"
     ]
    }
   ],
   "source": [
    "training_protocol(nb_epochs, model, criterion, train_in, train_tg, \\\n",
    "                  valid_input.float(), valid_target.float(), batch_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "94101941",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "fb35efd5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "22.844746778212574"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "denoised = model(valid_input.float()).detach()\n",
    "denoised = denoised/denoised.max()\n",
    "ground_truth = valid_target.float()\n",
    "ground_truth = ground_truth/ground_truth.max()\n",
    "noisy = valid_input.float()\n",
    "noisy = noisy/noisy.max()\n",
    "\n",
    "mse = criterion(denoised, ground_truth).item()\n",
    "-10 * np.log10(mse + 10**-8)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e743a977",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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NzG8i27bqo7QOoueO5zwbNtQ5bc92Lue1teuXO3Wzjrdvx4tlb7h1vSXMxERERERkQzQ5ExERERERCQDNDkTERERERHJAE3OREREREREMkCTMxERERERkQzQ5ExERERERCQDNDkTERERERHJgHnNOStXSjj/8ee4dTM/t6LYzbNMxg7xLKF+HieGIqm3eJwGQonvPDU/6yOJ7LsAnsljuTKt50luTu+An+MBAG947S/Q+lmXnOHWump83n/V3Ztp/f9c9WFa/5t3/INbW7G6QrfdvpNnryQFvn0h5weRJDmepze8j+f1vfttNbfWiozr1965ndYv+fIf0vr//cB73dpt2/a6tYl6k+73ZGA5Q7FEHjvzn+MhkpuVkuwpAMgl/LnC8sYskkGUi77/5t8vS3lfi+WcxRpnIFk/m++9l257+8Yf0PrE+GG3lotkS/YP8pyzXU97Gq1venCrWxvex/N0YvltrTRyzMlj1mjy87Ra5a+h5Zy/fbXJe8Dmzd20fsEBni35+IsvcWvjKc+OOxWkrQYmDvs9OKxa4W/b5K93IeXnRYhkJbKtY1liCPy22d8PYhGPFrl2QiRTMEeycT/1L++n2y5bfRqtHz6ww63dd9cddNtFS/3HGgDWrDub1nc+4Get1ko8xyyXi+Q0khwzAMjnybVTwrcttHiPyZH5Q08Pv1/llPe+Nefy67b+fj9HrXuJn4GWI8dDfzkTERERERHJAE3OREREREREMkCTMxERERERkQzQ5ExERERERCQDNDkTERERERHJAE3OREREREREMmBel9IPaUBtxF8qfOTAAbc2fng33XdXZJrZTPlysrWav0xngS2xDaC/zJfpnKyOujW21D0AtAJfXrQ2OUHrlYq/hPEbfvMVdNuuIl/eeNN37nJruSJ/QMrdXbR+8PAwrVer/jK5Wx8aodvmKvzxqpIlvgEgT5429dI43XbxivX8tif989QCX0r29D6+XPWWcR678Bt/+k63ds4FT3Br9/7we3S/JwMDX2K9Njzs1loN3luQ8sctRJYQDqn/XMpHlrnORZa7Z8tosyX829vypfLTJt9+ouYvG3/bd2+i227dtonfNn2u8HHXybgAYMs2fxlsALjvPn/Z5dER/7UAAELiL60MAGnk9SIhS10nkXPFEh7VMUGWPY8kiOBQi//CjzbtpPWRlv9aNGl83KeE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\n",
      "text/plain": [
       "<Figure size 1080x864 with 3 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig,ax = plt.subplots(1,3, figsize=(15,12))\n",
    "select = 164\n",
    "\n",
    "ax[0].imshow(noisy[select].permute(1,2,0), origin='upper')\n",
    "ax[1].imshow(denoised[select].permute(1,2,0), origin='upper')\n",
    "ax[2].imshow(ground_truth[select].permute(1,2,0), origin='upper')\n",
    "\n",
    "ax[0].set_title(\"Validation input (noisy)\")\n",
    "ax[1].set_title(\"Denoised input)\")\n",
    "ax[2].set_title(\"Validation target (clean)\");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0276663e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b699e27c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6e804a42",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b1911e90",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
