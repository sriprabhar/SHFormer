import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable, grad
import numpy as np
import os 
import math
from torchvision import models
#from fca_layer2 import FCA
from fcbam import FCA
from dynamicHPF import DynamicHPF
from restormer import TransformerBlock

class cSELayerWithDCTHighPass(nn.Module):
    def __init__(self, channel):
        
        super(cSELayerWithDCTHighPass, self).__init__()
        self.fca = FCA(channel) 
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        #self.conv_du = nn.Sequential(
        #        nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
        #        nn.ReLU(inplace=True),
        #        nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        #        nn.Sigmoid()
        #)

    def forward(self, x):
        y =self.fca(x) 
        #y = self.avg_pool(x)
        #y = self.conv_du(y)
        
        return y


class cSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        
        super(cSELayer, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        
        y = self.avg_pool(x)
        y = self.conv_du(y)
        
        return x * y

class sSELayer(nn.Module):
    def __init__(self, channel):
        super(sSELayer, self).__init__()
        
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, 1, 1, padding=0, bias=True),
                nn.Sigmoid())


    def forward(self, x):
        
        y = self.conv_du(x)
        
        return x * y


class scSELayer(nn.Module):
    def __init__(self, channel,reduction=16):
        super(scSELayer, self).__init__()
        
        self.cSElayer = cSELayer(channel,reduction)
        self.sSElayer = sSELayer(channel)

    def forward(self, x):
        
        y1 = self.cSElayer(x)
        y2 = self.sSElayer(x)
        
        y  = torch.max(y1,y2)
        
        return y

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob,attention,attention_type,reduction): # cSE,scSE
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.attention = attention
        self.reduction = reduction

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )
        
        if self.attention:
            if attention_type == 'cSE':
                self.attention_layer = cSELayerWithDCTHighPass(channel=self.out_chans) 
                #self.attention_layer = cSELayer(channel=self.out_chans,reduction=reduction)
            if attention_type == 'scSE':
                self.attention_layer = scSELayer(channel=self.out_chans,reduction=reduction)
                  
    def forward(self, inp):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]

        """
        out = self.layers(inp)
        
        if self.attention:
            out = self.attention_layer(out)
        
        return out

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class CSEUnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob,attention_type='cSE',reduction=16):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.reduction = reduction

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob,attention=False,attention_type=attention_type,reduction=reduction)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob,attention=False,attention_type=attention_type,reduction=reduction)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob,attention=True,attention_type=attention_type,reduction=reduction)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob,attention=True,attention_type=attention_type,reduction=reduction)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob,attention=True,attention_type=attention_type,reduction=reduction)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input
        # Apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)
        output = output.float()

        # Apply up-sampling layers
        for layer in self.up_sample_layers:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
            output = torch.cat([output, stack.pop()], dim=1)
            output = layer(output)
            output = output.float()
        return self.conv2(output)

class restormer_layer(nn.Module):

    def __init__(self):
        super(restormer_layer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            TransformerBlock(dim=32, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'), 
            #nn.Conv2d(32, 32, 3, padding=1, bias=True),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(32, 32, 3, padding=1, bias=True),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(32, 32, 3, padding=1, bias=True),
            #nn.ReLU(inplace=True),
            nn.Conv2d(32, 1,  3, padding=1, bias=True)
        )

    def forward(self, x):
        x = self.conv(x)

        return x



class DataConsistencyLayer(nn.Module):

    def __init__(self):

        super(DataConsistencyLayer,self).__init__()

        #self.device = device

    def forward(self,predicted_img,us_kspace,us_mask):

#         us_mask_path = os.path.join(self.us_mask_path,dataset_string,mask_string,'mask_{}.npy'.format(acc_factor))

#         us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(self.device)
        #print(predicted_img.shape, us_kspace.shape, us_mask.shape)
        predicted_img = predicted_img[:,0,:,:]

        #print("predicted_img: ",predicted_img.shape)
        #print("us_kspace: ", us_kspace.shape, "us_mask: ",us_mask.shape)
        kspace_predicted_img = torch.fft.fft2(predicted_img,norm = "ortho")

        #print("kspace_predicted_img: ",kspace_predicted_img.shape)
        #kspace_predicted_img_real = torch.view_as_real(kspace_predicted_img)

        us_kspace_complex = us_kspace[:,:,:,0]+us_kspace[:,:,:,1]*1j

        updated_kspace1  = us_mask * us_kspace_complex

        updated_kspace2  = (1 - us_mask) * kspace_predicted_img
        #print("updated_kspace1: ", updated_kspace1.shape, "updated_kspace2: ",updated_kspace2.shape)

        updated_kspace = updated_kspace1 + updated_kspace2
        #print("updated_kspace: ", updated_kspace.shape)
        
        #updated_kspace = updated_kspace[:,:,:,0]+updated_kspace[:,:,:,1]*1j
        #print("updated_kspace: ", updated_kspace.shape)

        updated_img  = torch.fft.ifft2(updated_kspace,norm = "ortho")
        #print("updated_img: ", updated_img.shape)

        updated_img = torch.view_as_real(updated_img)
        #print("updated_img: ", updated_img.shape)
        
        update_img_abs = updated_img[:,:,:,0] # 

        update_img_abs = update_img_abs.unsqueeze(1)
        #print("updated_img_abs out of DC: ", update_img_abs.shape)

        return update_img_abs.float()


class DnCn(nn.Module):

    def __init__(self,args,n_channels=2, nc=5, nd=5,**kwargs):

        super(DnCn, self).__init__()

        self.nc = nc
        self.nd = nd

        #us_mask_path = os.path.join(args.usmask_path,'mask_{}.npy'.format(args.acceleration_factor))
        #us_mask = torch.from_numpy(np.load(us_mask_path)).unsqueeze(2).unsqueeze(0).to(args.device)

        print('Creating D{}C{}'.format(nd, nc))
        conv_blocks = []
        dcs = []

        for i in range(nc):
            print('Cascade: {}'.format(i))
            #conv_blocks.append(CSEUnetModel(in_chans=1,out_chans=1,chans=32,num_pool_layers=3,drop_prob = 0,attention_type='cSE',reduction=16))       
            conv_blocks.append(restormer_layer())       
            conv_blocks.append(CSEUnetModel(in_chans=1,out_chans=1,chans=32,num_pool_layers=3,drop_prob = 0,attention_type='cSE',reduction=16))       
            dcs.append(DataConsistencyLayer())

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dcs = dcs

    def forward(self,x,k,m):

        for i in range(self.nc):
            x_dct = self.conv_blocks[2*i](x)
            x_2 = x + x_dct
            x_2 = x_2.float()
            x_hpf = self.conv_blocks[(2*i)+1](x_2)
            x = x_2 + x_hpf
            #x = self.dcs[i](x,k)
            x = self.dcs[i](x,k,m)        
 
        return x



