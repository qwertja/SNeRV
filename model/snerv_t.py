import time
import torch
import torch.nn as nn
from math import sqrt
import numpy as np
from pytorch_wavelets import DWT, IDWT, DWT1D
from .layers import *

class SNeRV_T(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = args.embed
        ks_enc, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split('_')]
        enc_blks, dec_blks = [int(x) for x in args.num_blks.split('_')]

        # BUILD Encoder LAYERS
        if len(args.enc_strds):         #HNeRV
            enc_dim1, enc_dim2 = [int(x) for x in args.enc_dim.split('_')]
            c_in_list, c_out_list = [enc_dim1] * len(args.enc_strds), [enc_dim1] * len(args.enc_strds)
            c_out_list[-1] = enc_dim2
            c_out_list2 = [enc_dim1] * len(args.enc2_strds)
            c_out_list2[-1] = 3
            if args.conv_type[0] == 'convnext':
                encoder_layers = []
                encoder_layers.append(ConvNeXt(stage_blocks=enc_blks, strds=args.enc_strds, dims=c_out_list,
                    drop_path_rate=0, in_chans = 3))
                encoder_layers.append(ConvNeXt(stage_blocks=enc_blks, strds=args.enc2_strds, dims=c_out_list2,
                    drop_path_rate=0, in_chans = 3))
                encoder_layers.append(ConvNeXt(stage_blocks=enc_blks, strds=args.enc2_strds, dims=c_out_list2,
                    drop_path_rate=0, in_chans = 3))
                self.encoder = nn.ModuleList(encoder_layers)
            else:
                c_in_list[0] = 3
                encoder_layers = []
                for c_in, c_out, strd in zip(c_in_list, c_out_list, args.enc_strds):
                    encoder_layers.append(NeRVBlock(dec_block=False, conv_type=args.conv_type[0], ngf=c_in,
                     new_ngf=c_out, ks=ks_enc, strd=strd, bias=True, norm=args.norm, act=args.act))
                self.encoder = nn.Sequential(*encoder_layers)
            hnerv_hw = np.prod(args.enc_strds) // np.prod(args.dec_strds)
            self.fc_h, self.fc_w = hnerv_hw, hnerv_hw
            ch_in = enc_dim2
        else:
            ch_in = 2 * int(args.embed.split('_')[-1])
            self.pe_embed = PositionEncoding(args.embed)  
            self.encoder = nn.Identity()
            self.fc_h, self.fc_w = [int(x) for x in args.fc_hw.split('_')]

        # BUILD Decoder LAYERS  
        decoder_layers = []
        ngf_list = [] 
        ngf = args.fc_dim
        out_f = int(ngf * self.fc_h * self.fc_w)
        decoder_layer1 = NeRVBlock(dec_block=False, conv_type='conv', ngf=ch_in, new_ngf=out_f, ks=0, strd=1, 
            bias=True, norm=args.norm, act=args.act)
        decoder_layers.append(decoder_layer1)
        
        for i, strd in enumerate(args.dec_strds):                         
            reduction = sqrt(strd) if args.reduce==-1 else args.reduce
            new_ngf = int(max(round(ngf / reduction), args.lower_width))
            if i==0 or i==1:
                cur_blk = UpsampleBlock_2d(in_chan=ngf, out_chan=new_ngf, strd=strd)
            elif i==4:
                cur_blk = UpsampleBlock_last(in_chan=ngf, out_chan=new_ngf, strd=strd)
            else:
                cur_blk = UpsampleBlock(in_chan=ngf, out_chan=new_ngf, strd=strd)
            decoder_layers.append(cur_blk)
            ngf = new_ngf
            ngf_list.append(ngf)
        
        # For Time Embedding
        temp_up_layer = []     
        if args.crop_list == '480_960':
            if args.emb_size == 10:
                temp_strd = [3]
            elif args.emb_size == 5:
                temp_strd = [3,2]
        elif args.crop_list == '960_1920':
            if args.emb_size == 20:
                temp_strd = [2]
            elif args.emb_size == 10:
                temp_strd = [2,2]
            elif args.emb_size == 5:
                temp_strd = [2,2,2]
        elif args.crop_list == '640_1280':
            if args.emb_size == 20:
                temp_strd = [2]
            elif args.emb_size == 5:
                temp_strd = [2,2,2]
                
        for strd in temp_strd:
            temp_up_layer.append(nn.ConvTranspose2d(ngf_list[1], ngf_list[1], strd, strd, 0))
                                        
        temp_up_seq = nn.Sequential(*temp_up_layer)
        # For Time Embedding
        temp_emb_layer = nn.Sequential(
            NeRVBlock(dec_block=False, conv_type='conv', ngf=3, new_ngf=ngf_list[1], ks=0, strd=1, 
                bias=True, norm=args.norm, act=args.act),
            temp_up_seq
        )
        decoder_layers.append(temp_emb_layer)
        
        # HF Blocks
        decoder_layer2 = ConvBlock(ngf1=new_ngf, ngf2=new_ngf, out=3, act='leaky01')
        decoder_layer3 = ConvBlock(ngf1=new_ngf, ngf2=new_ngf, out=3, act='leaky01')
        decoder_layer4 = ConvBlock(ngf1=new_ngf, ngf2=new_ngf, out=3, act='leaky01')
        decoder_layers.extend([decoder_layer2, decoder_layer3, decoder_layer4])

        # Multi Resoltuion Fusion Blocks
        upsample_5 = nn.ConvTranspose2d(ngf_list[-3], ngf_list[-3], 2, 2, 0)
        decoder_layer5 = RB(in_channels=ngf_list[-3]+ngf_list[-2], out_channels=ngf_list[-2], num_blocks=args.num_blocks)
        upsample_6 = nn.ConvTranspose2d(ngf_list[-2], ngf_list[-2], 2, 2, 0)
        decoder_layer6 = RB(in_channels=ngf_list[-2]+new_ngf, out_channels=new_ngf, num_blocks=args.num_blocks)
        decoder_layers.extend([upsample_5, decoder_layer5, upsample_6, decoder_layer6])

        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = nn.Conv2d(ngf, 3, 3, 1, 1)
        self.out_bias = args.out_bias
        self.decoder_len = len(args.dec_strds) + 1 + 1

    def forward(self, input, input_p, input_n, input_embed=None):
        # ENCODER
        if input_embed != None:
            img_embed = input_embed
        else:          
            yl, _ = DWT(J=1, wave='haar', mode='periodization').cuda()(torch.cat([input, input_p, input_n],0))
            yl_norm = torch.as_tensor([yl.min(), yl.max()])
            embed = (yl-yl_norm[0])/(yl_norm[1]-yl_norm[0]) ### normalize
            
            n, c, h, w = embed[0:2].shape
            embed_lv_p, embed_hv_p = DWT1D(J=1, wave='haar', mode='periodization').cuda()(torch.cat([embed[0:1], embed[1:2]],0).reshape(n,c,h*w).permute(2,1,0))
            embed_lv_n, embed_hv_n = DWT1D(J=1, wave='haar', mode='periodization').cuda()(torch.cat([embed[0:1], embed[2:3]],0).reshape(n,c,h*w).permute(2,1,0))

            embed_curr = self.encoder[0](embed[0:1])
            embed_hv_p = self.encoder[1]((embed_lv_p.permute(2,1,0).reshape(1,c,h,w))/2)
            embed_hv_n = self.encoder[2]((embed_lv_n.permute(2,1,0).reshape(1,c,h,w))/2)
            img_embed = [embed_curr, torch.cat([embed_hv_p, embed_hv_n],1), yl_norm]        
        # DECODER
        idwt = IDWT(wave='haar', mode='periodization').cuda()

        embed_list = [img_embed]
        dec_start = time.time()
        emb_ch = (img_embed[1].size(1))//2
        output = self.decoder[0](img_embed[0])
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        embed_list.append(output)
        
        output_2 = self.decoder[self.decoder_len-1](torch.cat([img_embed[1][:,0:emb_ch,:,:], img_embed[1][:,emb_ch:,:,:]],0))
        n, c, h, w = output_2.shape
        output_2 = output_2.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        
        for i, layer in enumerate(self.decoder[1:self.decoder_len-1]):
            if i < 2:
                output = layer(output)
            elif i == self.decoder_len-3:
                output = layer(output, output_2)
            else: 
                output, output_2 = layer(output, output_2)
            embed_list.append(output)

        ### MFU
        up1 = self.decoder[self.decoder_len+3](embed_list[-3])
        unet1 = self.decoder[self.decoder_len+4](torch.cat([up1, embed_list[-2]], dim=1))
        unet1_up =self.decoder[self.decoder_len+5](unet1)
        pyr_out = self.decoder[self.decoder_len+6](torch.cat([unet1_up, embed_list[-1]], dim=1))

        img_yl = OutImg(self.head_layer(pyr_out), self.out_bias)
        yl_out = img_yl * (img_embed[2][1]-img_embed[2][0]) + img_embed[2][0] ### needed
        
        ### HFR
        HF_in = pyr_out
        lh_out = self.decoder[self.decoder_len](HF_in)
        hl_out = self.decoder[self.decoder_len+1](HF_in)
        hh_out = self.decoder[self.decoder_len+2](HF_in)
        
        yh_out = torch.stack([lh_out, hl_out, hh_out], dim=2)

        img_out = idwt([yl_out, [yh_out]])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start

        return  img_out, embed_list, dec_time, img_yl, yh_out
    
    
class SNeRV_T_2D(nn.Module): # SNeRV_T with 2Dconv
    def __init__(self, args):
        super().__init__()
        self.embed = args.embed
        ks_enc, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split('_')]
        enc_blks, dec_blks = [int(x) for x in args.num_blks.split('_')]

        # BUILD Encoder LAYERS
        if len(args.enc_strds):
            enc_dim1, enc_dim2 = [int(x) for x in args.enc_dim.split('_')]
            c_in_list, c_out_list = [enc_dim1] * len(args.enc_strds), [enc_dim1] * len(args.enc_strds)
            c_out_list[-1] = enc_dim2
            c_out_list2 = [enc_dim1] * len(args.enc2_strds)
            c_out_list2[-1] = 3
            if args.conv_type[0] == 'convnext':
                encoder_layers = []
                encoder_layers.append(ConvNeXt(stage_blocks=enc_blks, strds=args.enc_strds, dims=c_out_list,
                    drop_path_rate=0, in_chans = 3))
                encoder_layers.append(ConvNeXt(stage_blocks=enc_blks, strds=args.enc2_strds, dims=c_out_list2,
                    drop_path_rate=0, in_chans = 3))
                encoder_layers.append(ConvNeXt(stage_blocks=enc_blks, strds=args.enc2_strds, dims=c_out_list2,
                    drop_path_rate=0, in_chans = 3))
                self.encoder = nn.ModuleList(encoder_layers)
            else:
                c_in_list[0] = 3
                encoder_layers = []
                for c_in, c_out, strd in zip(c_in_list, c_out_list, args.enc_strds):
                    encoder_layers.append(NeRVBlock(dec_block=False, conv_type=args.conv_type[0], ngf=c_in,
                     new_ngf=c_out, ks=ks_enc, strd=strd, bias=True, norm=args.norm, act=args.act))
                self.encoder = nn.Sequential(*encoder_layers)
            hnerv_hw = np.prod(args.enc_strds) // np.prod(args.dec_strds)
            self.fc_h, self.fc_w = hnerv_hw, hnerv_hw
            # self.fc_h, self.fc_w = 1, 1
            ch_in = enc_dim2
        else:
            ch_in = 2 * int(args.embed.split('_')[-1])
            self.pe_embed = PositionEncoding(args.embed)  
            self.encoder = nn.Identity()
            self.fc_h, self.fc_w = [int(x) for x in args.fc_hw.split('_')]

        # BUILD Decoder LAYERS  
        decoder_layers = []
        ngf_list = [] 
        ngf = args.fc_dim
        out_f = int(ngf * self.fc_h * self.fc_w)
        decoder_layer1 = NeRVBlock(dec_block=False, conv_type='conv', ngf=ch_in, new_ngf=out_f, ks=0, strd=1, 
            bias=True, norm=args.norm, act=args.act)
        decoder_layers.append(decoder_layer1)
        
        for i, strd in enumerate(args.dec_strds):                         
            reduction = sqrt(strd) if args.reduce==-1 else args.reduce
            new_ngf = int(max(round(ngf / reduction), args.lower_width))
            if i==0 or i==1:
                cur_blk = UpsampleBlock_2d(in_chan=ngf, out_chan=new_ngf, strd=strd)
            elif i==4:
                cur_blk = UpsampleBlock_last_3dto2d(in_chan=ngf, out_chan=new_ngf, strd=strd)
            else:
                cur_blk = UpsampleBlock_3dto2d(in_chan=ngf, out_chan=new_ngf, strd=strd)
            decoder_layers.append(cur_blk)
            ngf = new_ngf
            ngf_list.append(ngf)
        
        # For Time Embedding
        temp_up_layer = []     
        if args.crop_list == '480_960':
            if args.emb_size == 10:
                temp_strd = [3]
            elif args.emb_size == 5:
                temp_strd = [3,2]
        elif args.crop_list == '960_1920':
            if args.emb_size == 20:
                temp_strd = [2]
            elif args.emb_size == 10:
                temp_strd = [2,2]
            elif args.emb_size == 5:
                temp_strd = [2,2,2]
        elif args.crop_list == '640_1280':
            if args.emb_size == 20:
                temp_strd = [2]
            elif args.emb_size == 10:
                temp_strd = [2,2]
            elif args.emb_size == 5:
                temp_strd = [2,2,2]
        else:
            raise NotImplementedError

        for strd in temp_strd:
            temp_up_layer.append(nn.ConvTranspose2d(ngf_list[1], ngf_list[1], strd, strd, 0))                                        
        temp_up_seq = nn.Sequential(*temp_up_layer)

        temp_emb_layer = nn.Sequential(
            NeRVBlock(dec_block=False, conv_type='conv', ngf=3, new_ngf=ngf_list[1], ks=0, strd=1, 
                bias=True, norm=args.norm, act=args.act),
            temp_up_seq
        )
        decoder_layers.append(temp_emb_layer)
        
        # HF Blocks
        decoder_layer2 = ConvBlock(ngf1=new_ngf, ngf2=new_ngf, out=3, act='leaky01')
        decoder_layer3 = ConvBlock(ngf1=new_ngf, ngf2=new_ngf, out=3, act='leaky01')
        decoder_layer4 = ConvBlock(ngf1=new_ngf, ngf2=new_ngf, out=3, act='leaky01')
        decoder_layers.extend([decoder_layer2, decoder_layer3, decoder_layer4])

        # Multi Resoltuion Fusion Blocks
        upsample_5 = nn.ConvTranspose2d(ngf_list[-3], ngf_list[-3], 2, 2, 0)
        decoder_layer5 = RB(in_channels=ngf_list[-3]+ngf_list[-2], out_channels=ngf_list[-2], num_blocks=args.num_blocks)
        upsample_6 = nn.ConvTranspose2d(ngf_list[-2], ngf_list[-2], 2, 2, 0)
        decoder_layer6 = RB(in_channels=ngf_list[-2]+new_ngf, out_channels=new_ngf, num_blocks=args.num_blocks)
        decoder_layers.extend([upsample_5, decoder_layer5, upsample_6, decoder_layer6])

        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = nn.Conv2d(ngf, 3, 3, 1, 1)
        self.out_bias = args.out_bias
        self.decoder_len = len(args.dec_strds) + 1 + 1

    def forward(self, input, input_p, input_n, input_embed=None):
        # ENCODER
        if input_embed != None:
            img_embed = input_embed
        else:          
            yl, _ = DWT(J=1, wave='haar', mode='periodization').cuda()(torch.cat([input, input_p, input_n],0))
            yl_norm = torch.as_tensor([yl.min(), yl.max()])
            embed = (yl-yl_norm[0])/(yl_norm[1]-yl_norm[0]) ### normalize
            
            n, c, h, w = embed[0:2].shape
            embed_lv_p, embed_hv_p = DWT1D(J=1, wave='haar', mode='periodization').cuda()(torch.cat([embed[0:1], embed[1:2]],0).reshape(n,c,h*w).permute(2,1,0))
            embed_lv_n, embed_hv_n = DWT1D(J=1, wave='haar', mode='periodization').cuda()(torch.cat([embed[0:1], embed[2:3]],0).reshape(n,c,h*w).permute(2,1,0))

            embed_curr = self.encoder[0](embed[0:1])
            embed_hv_p = self.encoder[1]((embed_lv_p.permute(2,1,0).reshape(1,c,h,w))/2)
            embed_hv_n = self.encoder[2]((embed_lv_n.permute(2,1,0).reshape(1,c,h,w))/2)
            img_embed = [embed_curr, torch.cat([embed_hv_p, embed_hv_n],1), yl_norm]
        
        # DECODER
        idwt = IDWT(wave='haar', mode='periodization').cuda()

        embed_list = [img_embed]
        dec_start = time.time()
        emb_ch = (img_embed[1].size(1))//2
        output = self.decoder[0](img_embed[0])
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        embed_list.append(output)
        
        output_2 = self.decoder[self.decoder_len-1](torch.cat([img_embed[1][:,0:emb_ch,:,:], img_embed[1][:,emb_ch:,:,:]],0))
        n, c, h, w = output_2.shape
        output_2 = output_2.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        
        for i, layer in enumerate(self.decoder[1:self.decoder_len-1]):
            if i < 2:
                output = layer(output)
            elif i == self.decoder_len-3:
                output = layer(output, output_2)
            else: 
                output, output_2 = layer(output, output_2)
            embed_list.append(output)

        ### MFU
        up1 = self.decoder[self.decoder_len+3](embed_list[-3])
        unet1 = self.decoder[self.decoder_len+4](torch.cat([up1, embed_list[-2]], dim=1))
        unet1_up =self.decoder[self.decoder_len+5](unet1)
        pyr_out = self.decoder[self.decoder_len+6](torch.cat([unet1_up, embed_list[-1]], dim=1))

        img_yl = OutImg(self.head_layer(pyr_out), self.out_bias)
        yl_out = img_yl * (img_embed[2][1]-img_embed[2][0]) + img_embed[2][0]
        
        ### HFR
        HF_in = pyr_out
        lh_out = self.decoder[self.decoder_len](HF_in)
        hl_out = self.decoder[self.decoder_len+1](HF_in)
        hh_out = self.decoder[self.decoder_len+2](HF_in)
        
        yh_out = torch.stack([lh_out, hl_out, hh_out], dim=2)

        img_out = idwt([yl_out, [yh_out]])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start

        return  img_out, embed_list, dec_time, img_yl, yh_out