import torch.nn as nn
from basic_modules import ConvLayer, EncDecNet, Normalize
from utils import init_weights

class SplitNet(nn.Module):
    def __init__(self, filter=[64, 128, 256, 512, 512], mid_layers=2, classes=7, tasks=['segmentation', 'depth', 'normal'], depth_activation=nn.ReLU()): 
        super().__init__()
        task_str = '_'
        self.classes = classes + 1
        self.tasks = tasks
        self.enc_dec = EncDecNet(filter, mid_layers)
        self.heads = nn.ModuleDict()
        for task in self.tasks:
            if task == 'segmentation':
                self.heads[task] = nn.Sequential(
                    ConvLayer(filter[0], filter[0]),
                    ConvLayer(filter[0], filter[0]),
                    nn.Conv2d(filter[0], self.classes, kernel_size=1)
                )
                task_str += 'seg_'
            elif task == 'depth':
                self.heads[task] = nn.Sequential(
                    ConvLayer(filter[0], filter[0]),
                    ConvLayer(filter[0], filter[0]),
                    nn.Conv2d(filter[0], 1, kernel_size=1),
                    depth_activation
                )
                task_str += 'dep_'
            elif task == 'normal':
                self.heads[task] = self.normal_head = nn.Sequential(
                    ConvLayer(filter[0], filter[0]),
                    ConvLayer(filter[0], filter[0]),
                    nn.Conv2d(filter[0], 3, kernel_size=1),
                    nn.Tanh(), 
                    Normalize()
                )
                task_str += 'nor_'
            else:
                raise ValueError("Invalid Task")
        self.name = "splitnet" + task_str[:-1]
        init_weights(self)

    def forward(self, x):
        logits = self.enc_dec(x)
        logits_dict = {}
        for k in self.tasks:
            logits_dict[k] = self.heads[k](logits)
        return logits_dict