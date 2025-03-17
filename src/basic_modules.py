import torch
import torch.nn as nn
from utils import mask_invalid_pixels
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        logits = self.conv(x)
        logits = self.bn(logits)
        logits = self.relu(logits)
        return logits

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layer = ConvLayer(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        out = self.conv_layer(x)
        logits, indices = self.pool(out)
        return logits, indices, out

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv_layer = ConvLayer(in_channels, out_channels)

    def forward(self, x, indices):
        logits = self.unpool(x, indices)
        up_layer = logits
        logits = self.conv_layer(logits)
        return logits, up_layer

class EncoderSH(nn.Module):
    def __init__(self, filter, mid_layers=0):
        super().__init__()
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        block = nn.Sequential(
            ConvLayer(3, filter[0]),
            *[ConvLayer(filter[0], filter[0]) for _ in range(mid_layers)]
        )
        self.enc_blocks.append(block)
        # self.enc_blocks.append(ConvLayer(3, filter[0]))
        self.down_blocks.append(DownSampleBlock(filter[0], filter[0]))
        for i in range(len(filter) - 1):
            # self.enc_blocks.append(ConvLayer(filter[i], filter[i+1]))
            # self.down_blocks.append(DownSampleBlock(filter[i+1], filter[i+1]))
            block = nn.Sequential(
                ConvLayer(filter[i], filter[i+1]),
                *[ConvLayer(filter[i+1], filter[i+1]) for _ in range(mid_layers)]
            )
            self.enc_blocks.append(block)
            # self.enc_blocks.append(ConvLayer(filter[i], filter[i+1]))
            self.down_blocks.append(DownSampleBlock(filter[i+1], filter[i+1]))

    def forward(self, x):
        down_indices = []
        down_layer = []
        enc_layer = []
        out = []
        logits = x
        for i in range(len(self.down_blocks)):
            logits = self.enc_blocks[i](logits)
            enc_layer.append(logits)
            logits, indices, down = self.down_blocks[i](logits)
            down_layer.append(down)    
            out.append(logits)
            down_indices.append(indices)
            
        return logits, enc_layer, down_layer, down_indices, out

class DecoderSH(nn.Module):
    def __init__(self, filter, mid_layers=0):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(filter) - 1):
            self.up_blocks.append(UpSampleBlock(filter[i], filter[i+1]))
            # self.dec_blocks.append(ConvLayer(filter[i+1], filter[i+1]))
            block = nn.Sequential(
                ConvLayer(filter[i+1], filter[i+1]),
                *[ConvLayer(filter[i+1], filter[i+1]) for _ in range(mid_layers)]
            )
            self.dec_blocks.append(block)
        self.up_blocks.append(UpSampleBlock(filter[-1], filter[-1]))
        block = nn.Sequential(
            ConvLayer(filter[-1], filter[-1]),
            *[ConvLayer(filter[-1], filter[-1]) for _ in range(mid_layers)]
        )
        self.dec_blocks.append(block)
        # self.dec_blocks.append(ConvLayer(filter[-1], filter[-1]))
            

    def forward(self, x, down_indices):
        up_layer = []
        dec_layer = []
        logits = x
        for i in range(len(self.up_blocks)):
            logits, up = self.up_blocks[i](logits, down_indices[-(i+1)])
            up_layer.append(up)
            logits = self.dec_blocks[i](logits)
            dec_layer.append(logits)
        return up_layer, dec_layer
    
class SharedNet(nn.Module):
    def __init__(self, filter, mid_layers=0):
        super().__init__()
        self.enc = EncoderSH(filter, mid_layers)
        self.dec = DecoderSH([filter[-(i+1)] for i in range(len(filter))], mid_layers)  

    def forward(self, x):
        logits, enc_layer, down_layer, down_indices, enc_out = self.enc(x)
        enc_dict = {'out': enc_layer, 'down': down_layer}
        up_layer, dec_layer = self.dec(logits, down_indices)
        dec_dict = {'out': dec_layer, 'up': up_layer}
        out_dict = {'enc': enc_out, 'dec': dec_layer}
        return enc_dict, dec_dict, down_indices, out_dict
    
class Encoder(nn.Module):
    def __init__(self, filter, mid_layers):
        super().__init__()
        start_block = nn.Sequential(
            # ConvLayer(3, filter[0]), 
            # ConvLayer(filter[0], filter[0]), 
            # ConvLayer(filter[0], filter[0])
            ConvLayer(3, filter[0]),
            *[ConvLayer(filter[0], filter[0]) for _ in range(mid_layers)]
        )
        self.enc_blocks = nn.ModuleList([start_block])
        self.down_blocks = nn.ModuleList([DownSampleBlock(filter[0], filter[0])])
        for i in range(len(filter) - 1):
            block = nn.Sequential(
                # ConvLayer(filter[i], filter[i+1]), 
                # ConvLayer(filter[i+1], filter[i+1]), 
                # ConvLayer(filter[i+1], filter[i+1]),
                ConvLayer(filter[i], filter[i+1]),
                *[ConvLayer(filter[i+1], filter[i+1]) for _ in range(mid_layers)]
            )
            self.enc_blocks.append(block)
            self.down_blocks.append(DownSampleBlock(filter[i+1], filter[i+1]))

    def forward(self, x):
        down_indices = []
        logits = x
        for i in range(len(self.down_blocks)):
            logits = self.enc_blocks[i](logits)
            logits, indices, _ = self.down_blocks[i](logits)
            down_indices.append(indices)
        return logits, down_indices

class Decoder(nn.Module):
    def __init__(self, filter, mid_layers=3):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(filter) - 1):
            self.up_blocks.append(UpSampleBlock(filter[i], filter[i+1]))
            block = nn.Sequential(
                # ConvLayer(filter[i+1], filter[i+1]), 
                # ConvLayer(filter[i+1], filter[i+1]),
                # ConvLayer(filter[i+1], filter[i+1])
                *[ConvLayer(filter[i+1], filter[i+1]) for _ in range(mid_layers)]
            )
            self.dec_blocks.append(block)
        self.up_blocks.append(UpSampleBlock(filter[-1], filter[-1]))
        block = nn.Sequential(
            # ConvLayer(filter[-1], filter[-1]), 
            # ConvLayer(filter[-1], filter[-1]),
            # ConvLayer(filter[-1], filter[-1])
            *[ConvLayer(filter[-1], filter[-1]) for _ in range(mid_layers)]
        )
        self.dec_blocks.append(block)

    def forward(self, x, down_indices):
        logits = x
        for i in range(len(self.up_blocks)):
            logits, _ = self.up_blocks[i](logits, down_indices[-(i+1)])
            logits = self.dec_blocks[i](logits)
        return logits
    
class EncDecNet(nn.Module):
    def __init__(self, filter, mid_layers=3):
        super().__init__()
        self.enc = Encoder(filter, mid_layers)
        # self.mid = nn.Sequential(*[ConvLayer(filter[-1], filter[-1]) for _ in range(mid_layers)])
        self.dec = Decoder([filter[-(i+1)] for i in range(len(filter))], mid_layers)  

    def forward(self, x):
        logits, down_indices = self.enc(x)
        # logits = self.mid(logits)
        logits = self.dec(logits, down_indices)
        return logits

class Normalize(nn.Module):
    def __init__(self, p=2, dim=1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)
    
class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # Adapt shape
        # x = x.squeeze(1)
        # Find the invalid pixels (depth 0)
        # mask = (y != 0).to(torch.float).to(x.device)
        mask = mask_invalid_pixels(y).to(torch.float).to(x.device)
        return torch.sum(torch.abs(x-y)*mask)/torch.sum(mask)
    
class DotProductLoss(nn.Module):
    # def __init__(self, reduction='mean'):
    def __init__(self):
        super().__init__()
        # self.reduction = reduction

    #TODO: check correctness
    def forward(self, x, y):
        # Normalize pixels
        #x_norm = F.normalize(x, p=2, dim=1)
        # Find the invalid pixels (points orthogonal to every axes)
        # mask = (torch.sum(y, dim=1, keepdim=True) != 0).to(torch.float).to(x.device)
        mask = mask_invalid_pixels(y).to(torch.float)
        # 1 - dot product to make it a loss with minimum 0 because both 
        return 1-torch.sum(x*y*mask) / torch.sum(mask)
        # B, C, H, W = x.shape
        # x_flat = x_norm.view(B, 1, C*H*W)
        # y_flat = y.view(B, C*H*W, 1)
        # loss = -(1/H*W)*torch.matmul(x_flat, y_flat).squeeze(1)
        # if self.reduction == 'mean':
        #     return torch.mean(loss)
        # elif self.reduction == 'sum':
        #     return torch.sum(loss)
        # else:
        #     return loss