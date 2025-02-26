import torch.nn as nn

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
    def __init__(self, filter):
        super().__init__()
        self.enc_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        self.enc_blocks.append(ConvLayer(3, filter[0]))
        self.down_blocks.append(DownSampleBlock(filter[0], filter[0]))
        for i in range(len(filter) - 1):
            self.enc_blocks.append(ConvLayer(filter[i], filter[i+1]))
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
    def __init__(self, filter):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(filter) - 1):
            self.up_blocks.append(UpSampleBlock(filter[i], filter[i+1]))
            self.dec_blocks.append(ConvLayer(filter[i+1], filter[i+1]))
        self.up_blocks.append(UpSampleBlock(filter[-1], filter[-1]))
        self.dec_blocks.append(ConvLayer(filter[-1], filter[-1]))
            

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
    def __init__(self, filter):
        super().__init__()
        self.enc = EncoderSH(filter)
        self.dec = DecoderSH([filter[-(i+1)] for i in range(len(filter))])  

    def forward(self, x):
        logits, enc_layer, down_layer, down_indices, enc_out = self.enc(x)
        enc_dict = {'out': enc_layer, 'down': down_layer}
        up_layer, dec_layer = self.dec(logits, down_indices)
        dec_dict = {'out': dec_layer, 'up': up_layer}
        out_dict = {'enc': enc_out, 'dec': dec_layer}
        return enc_dict, dec_dict, down_indices, out_dict
    
class Encoder(nn.Module):
    def __init__(self, filter):
        super().__init__()
        start_block = nn.Sequential(
            ConvLayer(3, filter[0]), 
            ConvLayer(filter[0], filter[0]), 
            ConvLayer(filter[0], filter[0])
        )
        self.enc_blocks = nn.ModuleList([start_block])
        self.down_blocks = nn.ModuleList([DownSampleBlock(filter[0], filter[0])])
        for i in range(len(filter) - 1):
            block = nn.Sequential(
                ConvLayer(filter[i], filter[i+1]), 
                ConvLayer(filter[i+1], filter[i+1]), 
                ConvLayer(filter[i+1], filter[i+1]),
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
    def __init__(self, filter):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(len(filter) - 1):
            block = nn.Sequential(
                ConvLayer(filter[i+1], filter[i+1]), 
                ConvLayer(filter[i+1], filter[i+1]),
                ConvLayer(filter[i+1], filter[i+1])
            )
            self.dec_blocks.append(block)
            self.up_blocks.append(UpSampleBlock(filter[i], filter[i+1]))
        self.up_blocks.append(UpSampleBlock(filter[-1], filter[-1]))
        block = nn.Sequential(
            ConvLayer(filter[-1], filter[-1]), 
            ConvLayer(filter[-1], filter[-1]),
            ConvLayer(filter[-1], filter[-1])
        )
        self.dec_blocks.append(block)

    def forward(self, x, down_indices):
        logits = x
        for i in range(len(self.up_blocks)):
            logits, _ = self.up_blocks[i](logits, down_indices[-(i+1)])
            logits = self.dec_blocks[i](logits)
        return logits
    
class EncDecNet(nn.Module):
    def __init__(self, filter, mid_layers):
        super().__init__()
        self.enc = Encoder(filter)
        self.mid = nn.Sequential(*[ConvLayer(filter[-1], filter[-1]) for _ in range(mid_layers)])
        self.dec = Decoder([filter[-(i+1)] for i in range(len(filter))])  

    def forward(self, x):
        logits, down_indices = self.enc(x)
        logits = self.mid(logits)
        logits = self.dec(logits, down_indices)
        return logits