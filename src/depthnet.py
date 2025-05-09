import torch.nn as nn
from segnet import SegNet

class DepthNet(SegNet):
    def __init__(self, filter=[64, 128, 256, 512, 512], mid_layers=4):
        super(DepthNet, self).__init__(filter=filter, classes=0, mid_layers=mid_layers)
        self.name = "depthnet"
        self.tasks = ['depth']
        self.classes = 1
        self.activation = nn.ReLU()

    def forward(self, x):
        logits = super().forward(x)
        logits = self.activation(logits[self.tasks[0]])
        logits_dict = {self.tasks[0]: logits}
        return logits_dict