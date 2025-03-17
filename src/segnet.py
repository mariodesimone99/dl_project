import torch.nn as nn
from basic_modules import ConvLayer, EncDecNet
from utils import init_weights

class SegNet(nn.Module):
    def __init__(self, filter=[64, 128, 256, 512, 512], mid_layers=4, classes=7):
        super().__init__()
        self.name = "segnet"
        self.tasks = ['segmentation']
        filter = [64, 128, 256, 512, 512]
        self.classes = classes + 1
        self.enc_dec = EncDecNet(filter, mid_layers)
        self.seg_head = nn.Sequential(
            ConvLayer(filter[0], filter[0]),
            nn.Conv2d(filter[0], self.classes, kernel_size=1)
        )
        init_weights(self)

    def forward(self, x):
        logits = self.enc_dec(x)
        logits = self.seg_head(logits)
        logits_dict = {self.tasks[0]: logits}
        return logits_dict
        # return logits