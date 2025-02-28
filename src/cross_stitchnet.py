import torch
import torch.nn as nn
from basic_modules import ConvLayer, DownSampleBlock, UpSampleBlock
from utils import init_weights


class CrossStitchNet(nn.Module):
    def __init__(self, filter=[64, 128, 256, 512, 512], classes=7, tasks=['segmentation', 'depth']):
        super().__init__()
        task_str = '_'
        self.classes = classes + 1
        self.tasks = tasks
        # self.alphas = nn.ParameterList([nn.Parameter(torch.rand(len(self.tasks))) for _ in range(len(self.tasks))])
        self.alphas = nn.ParameterDict({task: nn.ParameterDict({task : torch.rand(len(self.tasks)) for task in self.tasks}) for task in self.tasks})

        # self.nets = nn.ModuleList()
        # for _ in range(len(self.tasks)):
        #     self.nets.append(nn.ModuleList())

        # for i in range(len(self.tasks)):
        #     self.nets[i].append(ConvLayer(3, filter[0]))

        # for i in range(len(filter)-1):
        #     for j in range(len(self.tasks)):
        #         self.nets[j].append(ConvLayer(filter[i], filter[i+1]))
        #         self.nets[j].append(ConvLayer(filter[i+1], filter[i+1]))
        #         self.nets[j].append(DownSampleBlock(filter[i+1], filter[i+1]))

        # for i in range(len(filter)-1):
        #     for j in range(len(self.tasks)):
        #         self.nets[j].append(UpSampleBlock(filter[-(i+1)], filter[-(i+2)]))
        #         self.nets[j].append(ConvLayer(filter[-(i+2)], filter[-(i+2)]))
        #         self.nets[j].append(ConvLayer(filter[-(i+2)], filter[-(i+2)]))
        self.nets = nn.ModuleDict()
        for task in self.tasks:
            self.nets[task] = nn.ModuleList()
            self.nets[task].append(ConvLayer(3, filter[0]))
            if task == 'segmentation':
                task_str += 'seg_'
            elif task == 'depth':
                task_str += 'dep_'
            elif task == 'normal':
                task_str += 'nor_'
        for i in range(len(filter)-1):
            for task in self.tasks:
                self.nets[task].append(ConvLayer(filter[i], filter[i+1]))
                self.nets[task].append(ConvLayer(filter[i+1], filter[i+1]))
                self.nets[task].append(DownSampleBlock(filter[i+1], filter[i+1]))
        for i in range(len(filter)-1):
            for task in self.tasks:
                self.nets[task].append(UpSampleBlock(filter[-(i+1)], filter[-(i+2)]))
                self.nets[task].append(ConvLayer(filter[-(i+2)], filter[-(i+2)]))
                self.nets[task].append(ConvLayer(filter[-(i+2)], filter[-(i+2)]))
        for task in self.tasks:
            self.nets[task].append(ConvLayer(filter[0], filter[0]))
            if task == 'segmentation':
                self.nets[task].append(nn.Conv2d(filter[0], self.classes, kernel_size=1))
            elif task == 'depth' and len(self.tasks) == 2:
                self.nets[task].append(nn.Sequential(
                    nn.Conv2d(filter[0], 1, kernel_size=1),
                    nn.Sigmoid())
                )
            elif task == 'depth' and len(self.tasks) == 3:
                self.nets[task].append(nn.Sequential(
                    nn.Conv2d(filter[0], 1, kernel_size=1),
                    nn.ReLU())
                )
            elif task == 'normal':
                self.nets[task].append(nn.Sequential(
                    nn.Conv2d(filter[0], 3, kernel_size=1),
                    nn.Tanh())
                )

        # heads = nn.ModuleList()
        # if 'segmentation' in self.tasks:
        #     heads.append(nn.Conv2d(filter[0], self.classes, kernel_size=1))
        # if 'depth' in self.tasks and len(self.tasks) == 2:
        #     heads.append(nn.Sequential(
        #         nn.Conv2d(filter[0], 1, kernel_size=1),
        #         nn.Sigmoid())
        #     )#depth head
        # if 'depth' in self.tasks and len(self.tasks) == 3:
        #     heads.append(nn.Sequential(
        #         nn.Conv2d(filter[0], 1, kernel_size=1),
        #         nn.ReLU())
        #     )
        # if 'normal' in self.tasks:
        #     heads.append(nn.Sequential(
        #         nn.Conv2d(filter[0], 3, kernel_size=1),
        #         nn.Tanh())
        #     )#normal head
        # for i in range(len(self.tasks)):
        #     self.nets[i].append(ConvLayer(filter[0], filter[0]))
        #     self.nets[i].append(heads[i])
        self.name = "cross_stitch" + task_str[:-1]
        init_weights(self)
    
    def _apply_cross_stitch(self, x):
        logits_dict = x
        for logits_task in self.logits_dict.keys():
            for alphas_task in self.alphas.keys():
                logits_dict[logits_task] += self.alphas[logits_task][alphas_task] * self.logits_dict[alphas_task]
        return logits_dict

    def forward(self, x):
        logits_dict = {task: x for task in self.tasks}
        idx_dict = {}
        
        # All the nets are build with same number of modules
        for task in self.tasks:
            nmod = len(self.nets[task])
            break

        for i in range(nmod):
            apply_cross_stitch = False
            for task in self.tasks:
                if isinstance(self.nets[task][i], DownSampleBlock):
                    logits_task, idx, _ = self.nets[task][i](logits_dict[task])
                    idx_dict[task].append(idx)
                    logits_dict[task] = logits_task
                    apply_cross_stitch = True
                elif isinstance(self.nets[task][i], UpSampleBlock):
                    logits_task, _ = self.nets[task][i](logits_dict[task], idx_dict[task][-i-1])
                    logits_dict[task] = logits_task
                    apply_cross_stitch = True
                else:
                    logits_task = self.nets[task][i](logits_dict[task])
                    logits_dict[task] = logits_task
                    apply_cross_stitch = False
            if apply_cross_stitch:
                logits_dict = self._apply_cross_stitch(logits_dict)
        return logits_dict
        # logits_seg = x 
        # logits_depth = x
        # indices_A = []
        # indices_B = []
        # j = 1
        # if len(self.tasks) == 2:
        #     for modA, modB in zip(self.nets[0], self.nets[1]):
        #         if isinstance(modA, DownSampleBlock):
        #             logits_seg, idx_A, _ = modA(logits_seg)
        #             logits_depth, idx_B, _ = modB(logits_depth)
        #             indices_A.append(idx_A)
        #             indices_B.append(idx_B)
        #             logits_seg = self.alphas[0][0] * logits_seg + self.alphas[0][1] * logits_depth
        #             logits_depth = self.alphas[1][0] * logits_depth + self.alphas[1][1] * logits_seg
        #         elif isinstance(modA, UpSampleBlock):
        #             logits_seg, _ = modA(logits_seg, indices_A[-j])
        #             logits_depth, _ = modB(logits_depth, indices_B[-j])
        #             j += 1
        #             logits_seg = self.alphas[0][0] * logits_seg + self.alphas[0][1] * logits_depth
        #             logits_depth = self.alphas[1][0] * logits_depth + self.alphas[1][1] * logits_seg
        #         else: #isinstance(modA, ConvLayer) or head
        #             logits_seg = modA(logits_seg)
        #             logits_depth = modB(logits_depth)
        #     return logits_seg, logits_depth
        # else: # 3 tasks
        #     logits_normal = x
        #     indices_C = []
        #     for modA, modB, modC in zip(self.nets[0], self.nets[1], self.nets[2]):
        #         if isinstance(modA, DownSampleBlock):
        #             logits_seg, idx_A, _ = modA(logits_seg)
        #             logits_depth, idx_B, _ = modB(logits_depth)
        #             logits_normal, idx_C, _ = modC(logits_normal)
        #             indices_A.append(idx_A)
        #             indices_B.append(idx_B)
        #             indices_C.append(idx_C)
        #             logits_seg = self.alphas[0][0] * logits_seg + self.alphas[0][1] * logits_depth + self.alphas[0][2] * logits_normal
        #             logits_depth = self.alphas[1][0] * logits_seg + self.alphas[1][1] * logits_depth + self.alphas[1][2] * logits_normal
        #             logits_normal = self.alphas[2][0] * logits_seg + self.alphas[2][1] * logits_depth + self.alphas[2][2] * logits_normal
        #         elif isinstance(modA, UpSampleBlock):
        #             logits_seg, _ = modA(logits_seg, indices_A[-j])
        #             logits_depth, _ = modB(logits_depth, indices_B[-j])
        #             logits_normal, _ = modC(logits_normal, indices_C[-j])
        #             j += 1
        #             logits_seg = self.alphas[0][0] * logits_seg + self.alphas[0][1] * logits_depth + self.alphas[0][2] * logits_normal
        #             logits_depth = self.alphas[1][0] * logits_seg + self.alphas[1][1] * logits_depth + self.alphas[1][2] * logits_normal
        #             logits_normal = self.alphas[2][0] * logits_seg + self.alphas[2][1] * logits_depth + self.alphas[2][2] * logits_normal
        #         else:
        #             logits_seg = modA(logits_seg)
        #             logits_depth = modB(logits_depth)
        #             logits_normal = modC(logits_normal)
        #         logits_dict = {'segmentation': logits_seg, 'depth':logits_depth, 'normal':logits_normal}
        #     #return logits_seg, logits_depth, logits_normal
        #     return logits_dict