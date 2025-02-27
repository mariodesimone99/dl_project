import torch.nn as nn
from segnet import SegNet

class NormalNet(SegNet):
    def __init__(self, filter=[64, 128, 256, 512, 512], mid_layers=4):
        super(NormalNet, self).__init__(filter=filter, classes=2, mid_layers=mid_layers)
        self.name = "normalnet"
        self.task = ['normal']
        self.classes = 3
        self.activation = nn.Tanh()

    def forward(self, x):
        logits = super().forward(x)
        logits = self.activation(logits)
        logits_dict = {self.task[0]: logits}
        return logits_dict
        # return self.activation(logits)