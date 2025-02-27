import torch.nn as nn
from segnet import SegNet

class DepthNet(SegNet):
    def __init__(self, filter=[64, 128, 256, 512, 512], mid_layers=4, activation='relu'):
        super(DepthNet, self).__init__(filter=filter, classes=0, mid_layers=mid_layers)
        self.name = "depthnet"
        self.tasks = ['depth']
        self.classes = 1
        self.activation = nn.ReLU() if activation == 'relu' else nn.Sigmoid()

    def forward(self, x):
        logits = super().forward(x)
        logits = self.activation(logits)
        logits_dict = {'depth': logits}
        return logits_dict
        #return self.activation(logits)