import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_modules import ConvLayer, SharedNet, Normalize
from utils import init_weights

class TaskNet(nn.Module):
    def __init__(self, filter, classes, activation):
        super().__init__()
        self.classes = classes
        self.start_conv = nn.Sequential(
            ConvLayer(3, filter[0]),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  
        self.dense_enc = nn.ModuleList()
        self.dense_dec = nn.ModuleList()
        for i in range(len(filter)-1):
            dense_block_enc = nn.Sequential(
                ConvLayer(2*filter[i], filter[i+1]),
                ConvLayer(filter[i+1], filter[i+1]),
                nn.MaxPool2d(kernel_size=2, stride=2)   
            )
            self.dense_enc.append(dense_block_enc)
        dense_block_enc = nn.Sequential(
            ConvLayer(2*filter[-1], filter[-1]),
            ConvLayer(filter[-1], filter[-1]),
            nn.Upsample(scale_factor=2)
        )
        self.dense_enc.append(dense_block_enc)
        for i in range(len(filter)-1):
            dense_block_dec = nn.Sequential(
                ConvLayer(filter[-i-1]+filter[-i-2], filter[-i-2]),
                ConvLayer(filter[-i-2], filter[-i-2]),
                nn.Upsample(scale_factor=2)
            )
            self.dense_dec.append(dense_block_dec)
        dense_block_dec = nn.Sequential(
            ConvLayer(filter[0]+filter[0], filter[0]),
            ConvLayer(filter[0], filter[0])
        )
        self.dense_dec.append(dense_block_dec)
        self.head = nn.Sequential(
            ConvLayer(filter[0], filter[0]),
            ConvLayer(filter[0], filter[0]),
            nn.Conv2d(filter[0], classes, kernel_size=1),
            activation
        )
        
    def forward(self, x, out_dict):
        logits = self.start_conv(x)
        for i in range(len(self.dense_enc)):
            feat_in = torch.cat((logits, out_dict['enc'][i]), dim=1)
            logits = self.dense_enc[i](feat_in)
        for i in range(len(self.dense_dec)):
            feat_in = torch.cat((logits, out_dict['dec'][i]), dim=1)
            logits = self.dense_dec[i](feat_in)
        logits = self.head(logits)
        return logits

class DenseNet(nn.Module):
    def __init__(self, filter = [64, 128, 256, 512, 512], mid_layers=0, classes=7, tasks=['segmentation', 'depth', 'normal']):
        super().__init__()
        task_str = '_'
        
        self.tasks = tasks
        self.classes = classes + 1
        self.sh_net = SharedNet(filter, mid_layers)
        self.tasks_net = nn.ModuleDict()
        for task in self.tasks:
            if task == 'segmentation':
                self.tasks_net[task] = TaskNet(filter, classes=self.classes, activation=nn.Identity())
                task_str += 'seg_'
            elif task == 'depth':
                self.tasks_net[task] = TaskNet(filter, classes=1, activation=nn.ReLU())
                task_str += 'dep_'
            elif task == 'normal':
                self.tasks_net[task] = TaskNet(filter, classes=3, activation=nn.Sequential(nn.Tanh(), Normalize()))
                task_str += 'nor_'
            else:
                raise ValueError("Invalid task")
        self.name = "densenet" + task_str[:-1]
        init_weights(self)

    def forward(self, x):
        _, _, _, out_dict = self.sh_net(x)
        logits_dict = {}
        for key in self.tasks:
            logits_dict[key] = self.tasks_net[key](x, out_dict)
        return logits_dict