import torch
import torch.nn as nn
from basic_modules import ConvLayer, DownSampleBlock, UpSampleBlock, Normalize
from utils import init_weights


class CrossStitchNet(nn.Module):
    def __init__(self, filter=[64, 128, 256, 512, 512], mid_layers=1, classes=7, tasks=['segmentation', 'depth', 'normal'], depth_activation=nn.ReLU()):
        super().__init__()
        task_str = '_'
        self.classes = classes + 1
        self.tasks = tasks
        self.alphas = nn.ParameterDict({task: nn.ParameterDict({task : torch.rand(1) for task in self.tasks}) for task in self.tasks})
        self.nets = nn.ModuleDict()
        for task in self.tasks:
            self.nets[task] = nn.ModuleList()
            self.nets[task].append(ConvLayer(3, filter[0]))
        for i in range(len(filter)-1):
            for task in self.tasks:
                self.nets[task].append(ConvLayer(filter[i], filter[i+1]))
                self.nets[task].append(nn.Sequential(*[ConvLayer(filter[i+1], filter[i+1]) for _ in range(mid_layers)]))
                self.nets[task].append(DownSampleBlock(filter[i+1], filter[i+1]))
        for i in range(len(filter)-1):
            for task in self.tasks:
                self.nets[task].append(UpSampleBlock(filter[-(i+1)], filter[-(i+2)]))
                self.nets[task].append(nn.Sequential(*[ConvLayer(filter[-(i+2)], filter[-(i+2)]) for _ in range(mid_layers)]))
                self.nets[task].append(ConvLayer(filter[-(i+2)], filter[-(i+2)]))
        for task in self.tasks:
            self.nets[task].append(ConvLayer(filter[0], filter[0]))
            if task == 'segmentation':
                self.nets[task].append(nn.Conv2d(filter[0], self.classes, kernel_size=1))
                task_str += 'seg_'
            elif task == 'depth':
                self.nets[task].append(nn.Sequential(
                    nn.Conv2d(filter[0], 1, kernel_size=1),
                    depth_activation)
                )
                task_str += 'dep_'
            elif task == 'normal':
                self.nets[task].append(nn.Sequential(
                    nn.Conv2d(filter[0], 3, kernel_size=1),
                    nn.Tanh(), 
                    Normalize())
                )
                task_str += 'nor_'
            else:
                raise ValueError('Invalid task')
        self.name = "cross_stitch" + task_str[:-1]
        init_weights(self)
    
    def _apply_cross_stitch(self, x):
        logits_dict = x
        for logits_task in logits_dict.keys():
            for alphas_task in self.alphas.keys():
                logits_dict[logits_task] = logits_dict[logits_task] + self.alphas[logits_task][alphas_task] * logits_dict[alphas_task]
        return logits_dict

    def forward(self, x):
        logits_dict = {task: x for task in self.tasks}
        idx_dict = {task: [] for task in self.tasks}
        
        # All the nets are build with same number of modules for each task
        for task in self.tasks:
            nmod = len(self.nets[task])
            break
        
        j = 1

        for i in range(nmod):
            apply_cross_stitch = False
            for task in self.tasks:
                if isinstance(self.nets[task][i], DownSampleBlock):
                    logits_task, idx, _ = self.nets[task][i](logits_dict[task])
                    idx_dict[task].append(idx)
                    logits_dict[task] = logits_task
                    apply_cross_stitch = True
                elif isinstance(self.nets[task][i], UpSampleBlock):
                    logits_task, _ = self.nets[task][i](logits_dict[task], idx_dict[task][-j])
                    logits_dict[task] = logits_task
                    j = j + 1 if task == self.tasks[-1] else j
                    apply_cross_stitch = True
                else:
                    logits_task = self.nets[task][i](logits_dict[task])
                    logits_dict[task] = logits_task
                    apply_cross_stitch = False
            logits_dict = self._apply_cross_stitch(logits_dict) if apply_cross_stitch else logits_dict
        return logits_dict