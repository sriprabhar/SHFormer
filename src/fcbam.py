import torch
import numpy as np
from torch import nn
import cmath
import math

class DCT_2D(nn.Module):
    '''
        Computes 2D-DCT of the input
    '''
    def __init__(self) -> None:
        super().__init__()


    def torch_dct(self,ip):
        
        length = ip.shape[-1] 
        
        arr = torch.cat((ip,torch.flip(ip,[-1])),dim=-1)

        fft_out = torch.fft.fft(arr,norm='forward')
        #print(fft_out.shape, torch.min(torch.abs(fft_out)), torch.max(torch.abs(fft_out)))
        req_fft = fft_out[:,:,:,:length]
        req_sign = torch.sign(req_fft.real)

        torch_out = req_fft.abs() * req_sign

        return torch_out

    def forward(self,ip):
        row = self.torch_dct(ip)
        rowt = row.transpose(-1,-2)
        outt = self.torch_dct(rowt)
        return outt.transpose(-1,-2)

class DCT_Channel(nn.Module):
    '''
        Computes 1D-DCT on the channel dimension of the input (2D-DCT)
    '''
    def __init__(self) -> None:
        super().__init__()


    def torch_dct(self,ip):
        
        length = ip.shape[1] # C channel in B, C, H, W 
        
        arr = torch.cat((ip,torch.flip(ip,[1])),dim=1)

        fft_out = torch.fft.fft(arr,norm='forward',dim=1)
        #print(fft_out.shape, torch.min(torch.abs(fft_out)), torch.max(torch.abs(fft_out)))
        req_fft = fft_out[:,:length,:,:]
        req_sign = torch.sign(req_fft.real)

        torch_out = req_fft.abs() * req_sign

        return torch_out

    def forward(self,ip):
        out = self.torch_dct(ip)
        return out
    
class iDCT_Channel(nn.Module):
    '''
        Computes 1D-iDCT on the channel dimension of the input (2D-iDCT)
    '''
    def __init__(self) -> None:
        super().__init__()
        self.dct_chan = DCT_Channel()
        
#     def torch_idct(self,ip):
    def forward(self,ip):
#         print(ip.shape)
        b,numElements,H,W = ip.shape # C channel in B, C, H, W 
        vGrid = np.array([range(2*numElements)]) 
        vShiftGrid = np.array([cmath.exp(-1j * 2 * math.pi * (numElements - 0.5) * v / (2 * numElements)) for v in vGrid[0]])
        vShiftGrid = torch.from_numpy(vShiftGrid).to(ip.device)
        vShiftGrid = vShiftGrid.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
#         viDct = self.dct_chan(ip)
        viDct = ip
        viDct    = viDct * (2 * numElements)
        viDct_flip = torch.flip(viDct[:,1:,:,:],[1]) ##  viDct_flip = torch.flip(viDct[1:],[-1])
        viDct_flip_negated = torch.mul(viDct_flip, -1)
        vXDft_reverse = torch.cat([viDct, torch.zeros(b,1,H,W,dtype=ip.dtype, device=ip.device),viDct_flip_negated],dim=1)
        #print(vXDft_reverse.device,vShiftGrid.device)
#         print("vXDft_reverse: ",vXDft_reverse.shape,"vShiftGrid: ",vShiftGrid.shape)
        vXDft_rev = torch.mul(vXDft_reverse,vShiftGrid)
        vxx = torch.fft.ifft(vXDft_rev,dim=1)
        vxx_flip = torch.flip(vxx[:,:numElements,:,:],dims=[1])
        vxx_flip_real = vxx_flip.real
        return vxx_flip_real

#class FCA(nn.Module):
#    '''
#        Computes Attention based on the frequency spectrum
#    '''
#    #def __init__(self,channels,topk=1,reduction=16) -> None:
#    def __init__(self,channels,topk=4,reduction=8,topk_chans=4) -> None: #best setting with 32 top k dct coeffs
#    #def __init__(self,channels,topk=1,reduction=4) -> None:
#        super().__init__()
#        self.channels = channels
#        self.dct_2d   = DCT_2D()
#        self.topk=topk
#        self.fc = nn.Sequential(
#            nn.Linear(channels * topk, channels // reduction, bias=False),
#            nn.ReLU(inplace=True),
#            nn.Linear(channels // reduction, channels, bias=False),
#            nn.Sigmoid()
#        )
#        # print(self.fc)
#        self.dct_chan = DCT_Channel()
#        self.topk_chans = topk_chans
#        self.conv1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3,padding=1),nn.BatchNorm2d(channels),nn.ReLU())
#        self.conv2 = nn.Conv2d(topk_chans,1,kernel_size=1,padding=0)
#        self.sigmoid = nn.Sigmoid()    
#        self.idct_chan = iDCT_Channel()
#
#    def forward(self,ip):
#        batches,channels,_,_ = ip.shape
#        dct_coeffs = self.dct_2d(ip)
#        dct_coeffs_re = dct_coeffs.reshape((batches,channels,-1))
#        top_4_dct_coeffs,_ = torch.topk(dct_coeffs_re,self.topk,dim=-1)
#        top_4_dct_coeffs = top_4_dct_coeffs.reshape((batches,-1))
#
#        weights = self.fc(top_4_dct_coeffs).view(batches,channels,1,1)
#        channel_attn_out = ip * weights.expand_as(ip)
#        channel_attn_out = self.dct_chan(channel_attn_out) 
#        channel_attn_convout = self.conv1(channel_attn_out)
#        channel_topk,_ = torch.topk(channel_attn_out,self.topk_chans,dim=1)
#        chan_topk_conv = self.conv2(channel_topk)
#        chan_topk_conv_act = self.sigmoid(chan_topk_conv)
#        final_out = channel_attn_convout * chan_topk_conv_act.expand_as(ip)
#        final_out = self.idct_chan(final_out)
#        return final_out
#

class FCA(nn.Module):
    '''
        Computes Attention based on the frequency spectrum
    '''
    #def __init__(self,channels,topk=1,reduction=16) -> None:
    def __init__(self,channels,topk=8,reduction=8,topk_chans=8) -> None: #best setting with 32 top k dct coeffs
    #def __init__(self,channels,topk=1,reduction=4) -> None:
        super().__init__()
        self.channels = channels
        self.dct_2d   = DCT_2D()
        self.topk=topk
        self.fc = nn.Sequential(
            nn.Linear(channels * topk, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        # print(self.fc)
        self.dct_chan = DCT_Channel()
        self.topk_chans = topk_chans
        #self.conv1 = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3,padding=1),nn.BatchNorm2d(channels),nn.ReLU())
        self.conv2 = nn.Conv2d(topk_chans,1,kernel_size=3,padding=1)
        self.sigmoid = nn.Sigmoid()    
        self.idct_chan = iDCT_Channel()

    def forward(self,ip):
        ip = ip.to(torch.float32)
        channel_attn_out = ip
        channel_attn_out = self.dct_chan(channel_attn_out) 
        #channel_attn_convout = self.conv1(channel_attn_out)
        channel_topk,_ = torch.topk(channel_attn_out,self.topk_chans,dim=1)
        chan_topk_conv = self.conv2(channel_topk)
        chan_topk_conv_act = self.sigmoid(chan_topk_conv)
        final_out = channel_attn_out * chan_topk_conv_act.expand_as(ip)
        ip = self.idct_chan(final_out)
        ip = ip.to(torch.float32)
        batches,channels,_,_ = ip.shape
        dct_coeffs = self.dct_2d(ip)
        dct_coeffs_re = dct_coeffs.reshape((batches,channels,-1))
        top_4_dct_coeffs,_ = torch.topk(dct_coeffs_re,self.topk,dim=-1)
        top_4_dct_coeffs = top_4_dct_coeffs.reshape((batches,-1))
        weights = self.fc(top_4_dct_coeffs).view(batches,channels,1,1)
        channel_attn_out = ip * weights.expand_as(ip)
 
        return channel_attn_out

if __name__ == '__main__':
    feats = torch.rand((4,16,128,128))
    fca   = FCA(16,topk=4,topk_chans=4)
    out = fca(feats)

    print(f'out shape = {out.shape}')
    
    
######### Corresponding Matlab implementation of iDCT in consistency with Sci py iDCT #################
# clc;clear all;close all; clearvars;format compact;
# rng(1);
# numElements = 10;

# %vX = randn(1, numElements);
# vX = [-0.6490 1.1812 -0.7585 -1.1096 -0.8456 -0.5727 -0.5587 0.1784 -0.1969 0.5864]

# disp('input signal');
# disp(vX); %<! Diusplay the input signal

# vDctRef = dct(vX);

# % Forward DCR using FFT
# vXX     = [fliplr(vX), vX]; %<! Mirroring
# vXDft   = fft(vXX);

# vGrid = [0:((2 * numElements) - 1)];

# vShiftGrid = exp(-1j * 2 * pi * (numElements - 0.5) * vGrid / (2 * numElements));

# vXDft2 = real(vXDft ./ vShiftGrid);

# % vDct    = vXDft2(1:numElements) / sqrt(2 * numElements);
# vDct    = vXDft2(1:numElements) / (2 * numElements);
# %vDct(1) = vDct(1) / sqrt(2);

# disp('ref')
# disp(vDctRef)
# disp('computed')
# disp(vDct)

# % Inverse DCT Using FFT
# %vDct(1) = vDct(1) * sqrt(2);

# %vDct    = vDct * sqrt(2 * numElements);
# vDct    = vDct * (2 * numElements);

# vXDft = [vDct, 0, -fliplr(vDct(2:numElements))] .* vShiftGrid;
# vXX = ifft(vXDft);

# vX = real(fliplr(vXX(1:numElements)));
# disp(vX);

####################################################################################
