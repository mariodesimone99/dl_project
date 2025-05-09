import torch.nn as nn
from segnet import SegNet
from basic_modules import Normalize

class NormalNet(SegNet):
    def __init__(self, filter=[64, 128, 256, 512, 512], mid_layers=4):
        super(NormalNet, self).__init__(filter=filter, classes=2, mid_layers=mid_layers)
        self.name = "normalnet"
        self.tasks = ['normal']
        self.classes = 3
        self.activation = nn.Sequential(
            nn.Tanh(),
            Normalize()
        )

    def forward(self, x):
        logits = super().forward(x)
        logits = self.activation(logits[self.tasks[0]])
        logits_dict = {self.tasks[0]: logits}
        return logits_dict