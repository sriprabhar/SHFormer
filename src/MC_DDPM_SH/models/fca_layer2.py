import torch
import numpy as np
from torch import nn

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


class FCA(nn.Module):
    '''
        Computes Attention based on the frequency spectrum
    '''
    def __init__(self,channels,reduction=16) -> None:
        super().__init__()
        self.channels = channels
        self.dct_2d   = DCT_2D()

        self.fc = nn.Sequential(
            nn.Linear(channels * 1, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        print(self.fc)

    def forward(self,ip):
        batches,channels,_,_ = ip.shape
        dct_coeffs = self.dct_2d(ip)
        dct_coeffs_re = dct_coeffs.reshape((batches,channels,-1))
        top_4_dct_coeffs,_ = torch.topk(dct_coeffs_re,1,dim=-1)
        top_4_dct_coeffs = top_4_dct_coeffs.reshape((batches,-1))

        weights = self.fc(top_4_dct_coeffs).view(batches,channels,1,1)
        return ip * weights.expand_as(ip)
        

if __name__ == '__main__':
    feats = torch.rand((4,16,128,128))
    fca   = FCA(16,128,128)
    out = fca(feats)

    print(f'out shape = {out.shape}')





    

        
        


