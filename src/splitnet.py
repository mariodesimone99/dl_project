import torch.nn as nn
from basic_modules import ConvLayer, EncDecNet
from utils import init_weights

class SplitNet(nn.Module):
    def __init__(self, filter=[64, 128, 256, 512, 512], classes=7, mid_layers=4, tasks=['segmentation', 'depth']):
        super().__init__()
        self.name = "splitnet"
        self.classes = classes + 1
        self.tasks = tasks
        # self.enc_net = Encoder(filter)
        # self.mid_net = nn.Sequential(*[ConvLayer(filter[-1], filter[-1]) for _ in range(mid_layers)])
        # self.dec_net = Decoder([filter[-(i+1)] for i in range(len(filter))])
        self.enc_dec = EncDecNet(filter, mid_layers)
        self.seg_head = nn.Sequential(
            ConvLayer(filter[0], filter[0]),
            ConvLayer(filter[0], filter[0]),
            nn.Conv2d(filter[0], self.classes, kernel_size=1)
        )

        if len(self.tasks) == 2:
            self.depth_head = nn.Sequential(
            ConvLayer(filter[0], filter[0]),
            ConvLayer(filter[0], filter[0]),
            nn.Conv2d(filter[0], 1, kernel_size=1),
            nn.Sigmoid()
        )
        else:
            self.depth_head = nn.Sequential(
            ConvLayer(filter[0], filter[0]),
            ConvLayer(filter[0], filter[0]),
            nn.Conv2d(filter[0], 1, kernel_size=1),
            nn.Sigmoid()
            )
            self.normal_head = nn.Sequential(
                ConvLayer(filter[0], filter[0]),
                ConvLayer(filter[0], filter[0]),
                nn.Conv2d(filter[0], 3, kernel_size=1),
                nn.Tanh()
            )
        init_weights(self)

    def forward(self, x):
        # logits, down_indices = self.enc_net(x)
        # logits = self.mid_net(logits)
        # logits = self.dec_net(logits, down_indices)
        logits = self.enc_dec(x)
        logits_seg = self.seg_head(logits)
        logits_depth = self.depth_head(logits)
        if len(self.tasks) == 3:
            logits_normal = self.normal_head(logits)
            return logits_seg, logits_depth, logits_normal
        return logits_seg, logits_depth