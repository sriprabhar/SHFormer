import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2

# returns the padding type
def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class DynamicHPF(nn.Module):

    def __init__(self, in_channels, kernel_size=3, stride=1, pad_type='reflect', group=2,inv = False):
        super(DynamicHPF, self).__init__()
        self.pad = get_pad_layer(pad_type)(kernel_size//2)
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group
        self.inv = inv

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        #print("dynamic hpf x: ", x.shape)
        sigma = self.conv(self.pad(x))
        sigma = self.bn(sigma)

        # sigma1 = sigma[:,:self.kernel_size**2,:,:]
        for i in range(self.group):
            sigma[:,(self.kernel_size**2)*i:(self.kernel_size**2)*(i+1),:,:] = self.softmax(sigma[:,(self.kernel_size**2)*i:(self.kernel_size**2)*(i+1),:,:]) # in dim=1, each group of kernels are seperated and then softmax is applied
            sigma[:,(self.kernel_size**2)*i + (self.kernel_size**2)//2,:,:] -= 1 # add -1 from center value of the the k^2 values in dim=1. This gives the condition for HPF, base paper does LPF 

        n,c,h,w = sigma.shape

#         sum_axis = 1
#         print(f'sigma shape = {sigma.shape}')#' sum in axis {sum_axis} = {torch.sum(sigma,axis=sum_axis)}')

        sigma = sigma.reshape(n,1,c,h*w)

        # sum_axis = 2
        # print(f'after reshape \n sigma shape = {sigma.shape}' sum in axis {sum_axis} = {torch.sum(sigma,axis=sum_axis)}')


        n,c,h,w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))

        n,c1,p,q = x.shape
        x = x.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)

        n,c2,p,q = sigma.shape

        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)

#         print(f'sigma shape = {sigma.shape} x shape = {x.shape}')#sigma sum = {torch.sum(sigma,axis=[1,3])}')
       
#         print(f'x*sigma shape = {(x*sigma).shape} \n torch.sum(x*sigma, dim=3) shape = {torch.sum(x*sigma, dim=3).shape}')
        x = torch.sum(x*sigma, dim=3).reshape(n,c1,h,w)
#         print(f'x shape = {x.shape}')
        return x[:,:,torch.arange(h)%self.stride==0,:][:,:,:,torch.arange(w)%self.stride==0]
