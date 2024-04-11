import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from dynamicHPF import DynamicHPF
def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads, bias):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
#         self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
#         self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


#     def forward(self, x):
#         b,c,h,w = x.shape

#         qkv = self.qkv_dwconv(self.qkv(x))
#         q,k,v = qkv.chunk(3, dim=1)   
        
#         q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature
#         attn = attn.softmax(dim=-1)

#         out = (attn @ v)
        
#         out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

#         out = self.project_out(out)
#         return out

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Parameter(torch.rand(1,num_heads,dim//num_heads,48))

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)



    def forward(self, x):
        b,c,h,w = x.shape

        task_q = self.q
        task_q = task_q.repeat(b,1,1,1)

        kv = self.kv_dwconv(self.kv(x))
        k,v = kv.chunk(2, dim=1)   
        
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # q = torch.nn.functional.normalize(q, dim=-1)
        q = torch.nn.functional.interpolate(task_q,size=(k.shape[2],k.shape[3]))
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.dynHPF = DynamicHPF(dim)# added dynamic HPF after Attn 
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.dynHPF(self.attn(self.norm1(x))) # added dynamic HPF after Attn
        x = x + self.ffn(self.norm2(x))

        return x

###############################################################################
# Using Separate Restormer for encoder and decoder
###############################################################################
class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:

            # self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            # self.csff_dec = nn.Conv2d(in_size, out_size, 3, 1, 1)
            # self.phi = nn.Conv2d(out_size, out_size, 3, 1, 1)
            # self.gamma = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.enc_trans = Restormer(in_chans=out_size,out_chans=out_size,heads=16)
            self.dec_trans = Restormer(in_chans=out_size,out_chans=out_size,heads=16)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            # skip_ = F.leaky_relu(self.csff_enc(enc) + self.csff_dec(dec), 0.1, inplace=True)
            # out = out*F.sigmoid(self.phi(skip_)) + self.gamma(skip_) + out
            phi = F.leaky_relu(self.enc_trans(enc), 0.1, inplace=True)
            gamma = F.leaky_relu(self.dec_trans(dec), 0.1, inplace=True)
            out = out*F.sigmoid(phi) + gamma + out
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

class Restormer(nn.Module):

    def __init__(self, in_chans, out_chans, heads = 1, num_blocks = 4, ffn_expansion_factor = 2.66,
                 bias = False, LayerNorm_type = 'WithBias'   ## Other option 'BiasFree'
                ) -> None:
        super().__init__()

        self.pe = OverlapPatchEmbed(in_chans,out_chans)
        self.trans = nn.Sequential(*[TransformerBlock(dim=out_chans, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks)])

    def forward(self,x):
        patches = self.pe(x)
        out = self.trans(patches)

        return out


if __name__ == '__main__':
    # dim = 48
    # in_chans = 16
    # heads = 1
    # num_blocks = 4
    # ffn_expansion_factor = 2.66
    # bias = False
    # LayerNorm_type = 'WithBias'   ## Other option 'BiasFree'
    # B = 4 # Batches

    # pe = OverlapPatchEmbed(in_chans,dim)
    # trans = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for _ in range(num_blocks)])


    # feat = torch.rand((B,in_chans,256,256))

    # patches = pe(feat)
    # out = trans(patches)

    # print(f'input shape = {feat.shape} out shape = {out.shape}')

    in_chans = 16
    out_chans = 32
    ip = torch.rand((4,in_chans,256,256))

    rest = Restormer(in_chans=in_chans,out_chans=out_chans,heads = 16,ffn_expansion_factor=1.66)
    out = rest(ip)

    print(f'input shape = {ip.shape} out shape = {out.shape}')

    params = sum([p.numel() for p in rest.parameters()])
    print(f'params = {params}')

