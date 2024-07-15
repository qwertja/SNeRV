import torch
import torch.nn as nn
from .residual_block import ResidualBlocksWithInputConv as RB
from math import pi, sqrt, ceil
import numpy as np
from matplotlib.path import Path
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F

class HfrBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        if kargs['mode'] == 'conv':
            self.layer = ConvBlock(ngf1=kargs['in_chan'], ngf2=kargs['in_chan'], out=3, act='leaky01')
        elif kargs['mode'] == 'RB':
            self.layer = nn.Sequential(
                RB(in_channels=kargs['in_chan'], out_channels=kargs['in_chan'], num_blocks=kargs['RB_blocks']),
                nn.Conv2d(kargs['in_chan'], 3, 1, 1, 0)
            )

    def forward(self, x):
        return self.layer(x)


class UpsampleBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.Tconv1 = nn.ConvTranspose2d(kargs['in_chan'], kargs['out_chan'], kargs['strd'], kargs['strd'], 0)
        self.Tconv2 = nn.ConvTranspose2d(kargs['in_chan'], kargs['out_chan'], kargs['strd'], kargs['strd'], 0)
        self.Tconv3 = nn.ConvTranspose2d(kargs['in_chan'], kargs['out_chan'], kargs['strd'], kargs['strd'], 0)
        self.conv3d = nn.Conv3d(kargs['out_chan'], kargs['out_chan']*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
        self.conv3d_2 = nn.Conv3d(kargs['out_chan']*2, kargs['out_chan'], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1))
    
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x_curr, x_t):
        x_curr = self.Tconv1(x_curr)
        x_p = self.Tconv2(x_t[0:1])
        x_n = self.Tconv3(x_t[1:2])

        x_curr_ = self.act(self.conv3d(torch.stack([x_p, x_curr, x_n],2)))
        x_curr = self.act(self.conv3d_2(x_curr_)) + torch.stack([x_p, x_curr, x_n],2)

        return x_curr[:,:,1,:,:], torch.cat([x_curr[:,:,0,:,:], x_curr[:,:,2,:,:]],0)

class UpsampleBlock_3dto2d(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.Tconv1 = nn.ConvTranspose2d(kargs['in_chan'], kargs['out_chan'], kargs['strd'], kargs['strd'], 0)
        self.Tconv2 = nn.ConvTranspose2d(kargs['in_chan'], kargs['out_chan'], kargs['strd'], kargs['strd'], 0)
        self.Tconv3 = nn.ConvTranspose2d(kargs['in_chan'], kargs['out_chan'], kargs['strd'], kargs['strd'], 0)
        self.conv3d = nn.Conv2d(kargs['out_chan']*3, kargs['out_chan']*6, 3, 1, 1)
        self.conv3d_2 = nn.Conv2d(kargs['out_chan']*6, kargs['out_chan']*3, 3, 1, 1)
    
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x_curr, x_t):
        x_curr = self.Tconv1(x_curr)
        x_p = self.Tconv2(x_t[0:1])
        x_n = self.Tconv3(x_t[1:2])

        x_curr_ = self.act(self.conv3d(torch.cat([x_p, x_curr, x_n], 1)))
        x_curr = self.act(self.conv3d_2(x_curr_)) + torch.cat([x_p, x_curr, x_n], 1)

        x_p, x_curr, x_n = torch.chunk(x_curr, 3, dim=1)

        return x_curr, torch.cat([x_p, x_n],0)

class UpsampleBlock_last_3dto2d(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.Tconv1 = nn.ConvTranspose2d(kargs['in_chan'], kargs['out_chan'], kargs['strd'], kargs['strd'], 0)
        self.Tconv2 = nn.ConvTranspose2d(kargs['in_chan'], kargs['out_chan'], kargs['strd'], kargs['strd'], 0)
        self.Tconv3 = nn.ConvTranspose2d(kargs['in_chan'], kargs['out_chan'], kargs['strd'], kargs['strd'], 0)

        self.conv3d = nn.Conv2d(kargs['out_chan']*3, kargs['out_chan']*6, 3, 1, 1)
        self.conv3d_2 = nn.Conv2d(kargs['out_chan']*6, kargs['out_chan'], 3, 1, 1)
        
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x_curr, x_t):
        x_curr = self.Tconv1(x_curr)
        x_p = self.Tconv2(x_t[0:1])
        x_n = self.Tconv3(x_t[1:2])

        x_curr_ = self.act(self.conv3d(torch.cat([x_p, x_curr, x_n], 1)))
        x_curr = self.act(self.conv3d_2(x_curr_)) + x_curr

        return x_curr

class UpsampleBlock_2d(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.Tconv1 = nn.ConvTranspose2d(kargs['in_chan'], kargs['out_chan'], kargs['strd'], kargs['strd'], 0)
        self.conv2d = nn.Conv2d(kargs['out_chan'], kargs['out_chan']*2, kernel_size=3, stride=1, padding=1)
        self.conv2d_2 = nn.Conv2d(kargs['out_chan']*2, kargs['out_chan'], kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x_curr):
        x_curr_ = self.Tconv1(x_curr)
        x_curr = self.act(self.conv2d(x_curr_))
        x_curr = self.act(self.conv2d_2(x_curr)) + x_curr_

        return x_curr
    
class UpsampleBlock_last(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.Tconv1 = nn.ConvTranspose2d(kargs['in_chan'], kargs['out_chan'], kargs['strd'], kargs['strd'], 0)
        self.Tconv2 = nn.ConvTranspose2d(kargs['in_chan'], kargs['out_chan'], kargs['strd'], kargs['strd'], 0)
        self.Tconv3 = nn.ConvTranspose2d(kargs['in_chan'], kargs['out_chan'], kargs['strd'], kargs['strd'], 0)

        self.conv3d = nn.Conv3d(kargs['out_chan'], kargs['out_chan']*2, kernel_size=(3,3,3), stride=1, padding=1)
        self.conv3d_2 = nn.Conv3d(kargs['out_chan']*2, kargs['out_chan'], kernel_size=(3,3,3), stride=1, padding=(0,1,1))
        
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    def forward(self, x_curr, x_t):
        x_curr = self.Tconv1(x_curr)
        x_p = self.Tconv2(x_t[0:1])
        x_n = self.Tconv3(x_t[1:2])

        x_curr_ = self.act(self.conv3d(torch.stack([x_p, x_curr, x_n],2)))
        x_curr = self.act(self.conv3d_2(x_curr_).squeeze(2)) + x_curr

        return x_curr
    
class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        conv = UpConv if kargs['dec_block'] else DownConv
        self.conv = conv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], strd=kargs['strd'], ks=kargs['ks'], 
            conv_type=kargs['conv_type'], bias=kargs['bias'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    

class ConvBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.conv1 = nn.Conv2d(kargs['ngf1'], kargs['ngf2'], 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(kargs['ngf2'], kargs['out'], 3, 1, 1)
        self.norm = nn.Identity()
        if kargs['act'] == 'leaky01':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif kargs['act'] == 'tanh':
            self.act = nn.Tanh()
        elif kargs['act'] == 'gelu':
            self.act = nn.GELU()
        elif kargs['act'] == 'relu':
            self.act = nn.ReLU()
        elif kargs['act'] == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        x = self.conv2(x)
        return x
    
    
class FusionBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        # self.conv1 = nn.Conv2d(kargs['ngf1'], kargs['ngf2'], 3, 1, 1, bias=True)
        self.conv1 = nn.Conv2d(kargs['ngf1'], kargs['ngf2'], 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(kargs['ngf2'], kargs['out'], 3, 1, 1)
        self.norm = nn.Identity()
        if kargs['act'] == 'leaky01':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif kargs['act'] == 'tanh':
            self.act = nn.Tanh()
        elif kargs['act'] == 'gelu':
            self.act = nn.GELU()
        elif kargs['act'] == 'relu':
            self.act = nn.ReLU()
        elif kargs['act'] == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.norm(self.conv1(x)))
        x = self.conv2(x)
        return x
    

def OutImg(x, out_bias='tanh'):
    if out_bias == 'sigmoid':
        return torch.sigmoid(x)
    elif out_bias == 'tanh':
        return (torch.tanh(x) * 0.5) + 0.5
    else:
        return x + float(out_bias)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


###################################  Basic layers like position encoding/ downsample layers/ upscale blocks   ###################################
class PositionEncoding(nn.Module):
    def __init__(self, pe_embed):
        super(PositionEncoding, self).__init__()
        self.pe_embed = pe_embed
        if 'pe' in pe_embed:
            lbase, levels = [float(x) for x in pe_embed.split('_')[-2:]]
            self.pe_bases = lbase ** torch.arange(int(levels)) * pi

    def forward(self, pos):
        if 'pe' in self.pe_embed:
            value_list = pos * self.pe_bases.to(pos.device)
            pe_embed = torch.cat([torch.sin(value_list), torch.cos(value_list)], dim=-1)
            return pe_embed.view(pos.size(0), -1, 1, 1)
        else:
            return pos
    


class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)


def ActivationLayer(act_type):
    if act_type == 'relu':
        act_layer = nn.ReLU(True)
    elif act_type == 'leaky':
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == 'leaky01':
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == 'relu6':
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == 'gelu':
        act_layer = nn.GELU()
    elif act_type == 'sin':
        act_layer = Sin
    elif act_type == 'swish':
        act_layer = nn.SiLU(inplace=True)
    elif act_type == 'softplus':
        act_layer = nn.Softplus()
    elif act_type == 'hardswish':
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):    
    if norm_type == 'none':
        norm_layer = nn.Identity()
    elif norm_type == 'bn':
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == 'in':
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


class DownConv(nn.Module):
    def __init__(self, **kargs):
        super(DownConv, self).__init__()
        ks, ngf, new_ngf, strd = kargs['ks'], kargs['ngf'], kargs['new_ngf'], kargs['strd']
        if kargs['conv_type'] == 'pshuffel':
            self.downconv = nn.Sequential(
                nn.PixelUnshuffle(strd) if strd !=1 else nn.Identity(),
                nn.Conv2d(ngf * strd**2, new_ngf, ks, 1, ceil((ks - 1) // 2), bias=kargs['bias'])
            )
        elif kargs['conv_type'] == 'conv':
            self.downconv = nn.Conv2d(ngf, new_ngf, ks+strd, strd, ceil(ks / 2), bias=kargs['bias'])
        elif kargs['conv_type'] == 'interpolate':
            self.downconv = nn.Sequential(
                nn.Upsample(scale_factor=1. / strd, mode='bilinear',),
                nn.Conv2d(ngf, new_ngf, ks+strd, 1, ceil((ks + strd -1) / 2), bias=kargs['bias'])
            )
        
    def forward(self, x):
        return self.downconv(x)


class UpConv(nn.Module):
    def __init__(self, **kargs):
        super(UpConv, self).__init__()
        ks, ngf, new_ngf, strd = kargs['ks'], kargs['ngf'], kargs['new_ngf'], kargs['strd']
        if  kargs['conv_type']  == 'pshuffel':
            self.upconv = nn.Sequential(
                nn.Conv2d(ngf, new_ngf * strd * strd, ks, 1, ceil((ks - 1) // 2), bias=kargs['bias']),
                nn.PixelShuffle(strd) if strd !=1 else nn.Identity(),
            )
        elif  kargs['conv_type']  == 'conv':
            self.upconv = nn.ConvTranspose2d(ngf, new_ngf, ks+strd, strd, ceil(ks / 2))
        elif  kargs['conv_type']  == 'interpolate':
            self.upconv = nn.Sequential(
                nn.Upsample(scale_factor=strd, mode='bilinear',),
                nn.Conv2d(ngf, new_ngf, strd + ks, 1, ceil((ks + strd -1) / 2), bias=kargs['bias'])
            )
        elif  kargs['conv_type']  == 'customconv':
            self.upconv = nn.ConvTranspose2d(ngf, new_ngf, strd, strd, 0)

    def forward(self, x):
        return self.upconv(x)


class ModConv(nn.Module):
    def __init__(self, **kargs):
        super(ModConv, self).__init__()
        mod_ks, mod_groups, ngf = kargs['mod_ks'], kargs['mod_groups'], kargs['ngf']
        self.mod_conv_multi = nn.Conv2d(ngf, ngf, mod_ks, 1, (mod_ks - 1)//2, groups=(ngf if mod_groups==-1 else mod_groups))
        self.mod_conv_sum = nn.Conv2d(ngf, ngf, mod_ks, 1, (mod_ks - 1)//2, groups=(ngf if mod_groups==-1 else mod_groups))

    def forward(self, x):
        sum_att = self.mod_conv_sum(x)
        multi_att = self.mod_conv_multi(x)
        return torch.sigmoid(multi_att) * x + sum_att


###################################  Tranform input for denoising or inpainting   ###################################
def RandomMask(height, width, points_num, scale=(0, 1)):
    polygon = [(x, y) for x,y in zip(np.random.randint(height * scale[0], height * scale[1], size=points_num), 
                             np.random.randint(width * scale[0], width * scale[1], size=points_num))]
    poly_path=Path(polygon)

    x, y = np.mgrid[:height, :width]
    coors=np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (4000000,2)
    mask = poly_path.contains_points(coors).reshape(height, width)
    return 1 - torch.from_numpy(mask).float()


###################################  Code for ConvNeXt   ###################################
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, stage_blocks=0, strds=[2,2,2,2], dims=[96, 192, 384, 768], 
            in_chans=3, drop_path_rate=0., layer_scale_init_value=1e-6,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stage_num = len(dims)
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, stage_blocks*self.stage_num)] 
        cur = 0
        for i in range(self.stage_num):
            # Build downsample layers
            if i > 0:
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i-1], dims[i], kernel_size=strds[i], stride=strds[i]),
                )
            else:
                downsample_layer = nn.Sequential(
                    nn.Conv2d(in_chans, dims[0], kernel_size=strds[i], stride=strds[i]),
                    LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
                )                
            self.downsample_layers.append(downsample_layer)

            # Build more blocks
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(stage_blocks)]
            )
            self.stages.append(stage)
            cur += stage_blocks

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out_list = []
        for i in range(self.stage_num):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            out_list.append(x)
        return out_list[-1]


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
