import torch.nn as nn
from mtan import AttNet
from basic_modules import SharedNet
from utils import init_weights

class STAN(nn.Module):
    def __init__(self, filter=[64, 128, 256, 512, 512], mid_layers=0, classes=7, task=['segmentation'], depth_activation='relu'):
        super().__init__()
        if task == ['depth']:
            self.name = "stan_dep"
            self.classes = 1
        elif task == ['segmentation']:
            self.name = "stan_seg"
            self.classes = classes + 1 # background
        elif task == ['normal']: # normals estimation
            self.classes = 3
            self.name = "stan_nor"
        else:
            raise ValueError("Invalid task")
        self.task = task
        self.sh_net = SharedNet(filter, mid_layers)
        self.attnet = AttNet(filter)
        if self.task == ['depth'] and depth_activation == 'relu':
            self.head = nn.Sequential(
            nn.Conv2d(filter[0], 1, kernel_size=1), 
            nn.ReLU()
        )
        elif self.task == ['depth'] and depth_activation == 'sigmoid':
            self.head = nn.Sequential(
            nn.Conv2d(filter[0], 1, kernel_size=1), 
            nn.Sigmoid()
        )
        elif self.task == ['segmentation']:
            self.head = nn.Conv2d(filter[0], self.classes, kernel_size=1)
        elif self.task == ['normal']: # normals estimation
            self.head = nn.Sequential(
            nn.Conv2d(filter[0], 3, kernel_size=1),
            nn.Tanh()
        )
        else:
            raise ValueError("Invalid task")


    def forward(self, x):
        enc_dict, dec_dict, _, _ = self.sh_net(x)
        logits = self.attnet(enc_dict, dec_dict)
        logits = self.head(logits)
        logits_dict = {self.task[0]: logits}
        return logits_dict
        #return logits