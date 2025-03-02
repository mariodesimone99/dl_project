import torch
import torch.nn as nn
from utils import init_weights
from basic_modules import ConvLayer, SharedNet

class AttEncBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.att_layer_g = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )
        self.att_layer_h = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Sigmoid()
        )
        self.att_layer_f = ConvLayer(mid_channels, out_channels)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, enc_layer, down_layer, x=None):
        logits = enc_layer if x == None else torch.cat([enc_layer, x], dim=1)
        g = self.att_layer_g(logits)
        h = self.att_layer_h(g)
        p = h * down_layer
        logits = self.att_layer_f(p)
        logits = self.down(logits)
        return logits

class AttDecBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.att_layer_f = ConvLayer(in_channels, out_channels)
        self.att_layer_g = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.att_layer_h = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x, up_layer, dec_layer):
        logits = self.up(x)
        logits = self.att_layer_f(logits)
        logits = torch.cat([logits, up_layer], dim=1)
        g = self.att_layer_g(logits)
        h = self.att_layer_h(g)
        p = h * dec_layer
        logits = p
        return logits
    
class AttNet(nn.Module):
    def __init__(self, filter):
        super().__init__()
        self.enc_att = nn.ModuleList()
        self.dec_att = nn.ModuleList()

        self.enc_att.append(AttEncBlock(filter[0], filter[0], filter[1]))
        for i in range(1,len(filter)-1):
            self.enc_att.append(AttEncBlock(2*filter[i], filter[i], filter[i+1]))
        self.enc_att.append(AttEncBlock(2*filter[-1], filter[-1], filter[-1]))
        
        for i in range(1, len(filter)):
            self.dec_att.append(AttDecBlock(filter[-i], filter[-i]+filter[-i-1], filter[-i-1]))
        self.dec_att.append(AttDecBlock(filter[0], 2*filter[0], filter[0]))

    def forward(self, enc_dict, dec_dict):
        for i in range(len(self.enc_att)):
            if i == 0:
                logits = self.enc_att[i](enc_dict['out'][i], enc_dict['down'][i])
            else:
                logits = self.enc_att[i](enc_dict['out'][i], enc_dict['down'][i], logits)

        for i in range(len(self.dec_att)):
            logits = self.dec_att[i](logits, dec_dict['up'][i], dec_dict['out'][i])
        return logits
    
class MTAN(nn.Module):
    def __init__(self, filter=[64, 128, 256, 512, 512], mid_layers=0, classes=7, tasks=['segmentation', 'depth', 'normal'], depth_activation='relu'):
        super().__init__()
        task_str = '_'
        self.classes = classes + 1 #background
        self.tasks = tasks
        self.sh_net = SharedNet(filter, mid_layers=mid_layers)
        self.attnet_task = nn.ModuleDict(zip(self.tasks, [AttNet(filter) for _ in range(len(self.tasks))]))
        # self.attnet_task = nn.ModuleList([AttNet(filter) for _ in range(len(self.tasks))])
        # to train with cross entropy loss
        # self.seg_head = nn.Conv2d(filter[0], self.classes, kernel_size=1)
        # #to train with L1 loss
        # if 'normals' in self.tasks:
        #     self.normal_head = nn.Sequential(
        #         nn.Conv2d(filter[0], 3, kernel_size=1),
        #         nn.Tanh()
        #     )
        #     self.depth_head = nn.Sequential(
        #     nn.Conv2d(filter[0], 1, kernel_size=1), 
        #     nn.ReLU()
        # )
        # else:
        #     self.depth_head = nn.Sequential(
        #     nn.Conv2d(filter[0], 1, kernel_size=1), 
        #     nn.Sigmoid()
        # )
        self.heads = nn.ModuleDict()
        for task in self.tasks:
            if task == 'segmentation':
                self.heads[task] = nn.Conv2d(filter[0], self.classes, kernel_size=1)
                task_str += 'seg_'
            elif task == 'depth' and depth_activation == 'relu':
                self.heads[task] = nn.Sequential(
                    nn.Conv2d(filter[0], 1, kernel_size=1), 
                    nn.ReLU()
                )
                task_str += 'dep_'
            elif task == 'depth' and depth_activation == 'sigmoid':
                self.heads[task] = nn.Sequential(
                    nn.Conv2d(filter[0], 1, kernel_size=1), 
                    nn.Sigmoid()
                )
                task_str += 'dep_'
            elif task == 'normal':
                self.heads[task] = nn.Sequential(
                    nn.Conv2d(filter[0], 3, kernel_size=1),
                    nn.Tanh()
                )
                task_str += 'nor_'
            else:
                raise ValueError("Invalid Task")
        self.name = "mtan" + task_str[:-1]
        init_weights(self)

    def forward(self, x):
        enc_dict, dec_dict, _, _ = self.sh_net(x)
        # logits = []
        # for i in range(len(self.tasks)):
        #     logits.append(self.attnet_task[i](enc_dict, dec_dict))
        # logits_seg = self.seg_head(logits[0])
        # logits_depth = self.depth_head(logits[1])
        # logits_dict = {'segmentation': logits_seg, 'depth': logits_depth}
        # if len(self.tasks) == 3:
        #     logits_normal = self.normal_head(logits[2])
        #     #return logits_seg, logits_depth, logits_normal
        #     logits_dict['normal'] = logits_normal
        # #return logits_seg, logits_depth
        # return logits_dict
        logits_dict = {}
        for k in self.tasks:
            logits_dict[k] = self.heads[k](self.attnet_task[k](enc_dict, dec_dict))
        return logits_dict