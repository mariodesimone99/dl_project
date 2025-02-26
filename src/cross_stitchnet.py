import torch
import torch.nn as nn
from basic_modules import ConvLayer, DownSampleBlock, UpSampleBlock
from utils import init_weights


class CrossStitchNet(nn.Module):
    def __init__(self, filter=[64, 128, 256, 512, 512], classes=7, tasks=['segmentation', 'depth']):
        super().__init__()
        self.name = "cross_stitch"
        self.classes = classes + 1
        self.tasks = tasks
        self.alphas = nn.ParameterList([nn.Parameter(torch.rand(len(self.tasks))) for _ in range(len(self.tasks))])

        self.nets = nn.ModuleList()
        for _ in range(len(self.tasks)):
            self.nets.append(nn.ModuleList())

        for i in range(len(self.tasks)):
            self.nets[i].append(ConvLayer(3, filter[0]))

        for i in range(len(filter)-1):
            for j in range(len(self.tasks)):
                self.nets[j].append(ConvLayer(filter[i], filter[i+1]))
                self.nets[j].append(ConvLayer(filter[i+1], filter[i+1]))
                self.nets[j].append(DownSampleBlock(filter[i+1], filter[i+1]))

        for i in range(len(filter)-1):
            for j in range(len(self.tasks)):
                self.nets[j].append(UpSampleBlock(filter[-(i+1)], filter[-(i+2)]))
                self.nets[j].append(ConvLayer(filter[-(i+2)], filter[-(i+2)]))
                self.nets[j].append(ConvLayer(filter[-(i+2)], filter[-(i+2)]))

        heads = nn.ModuleList()
        if 'segmentation' in self.tasks:
            heads.append(nn.Conv2d(filter[0], self.classes, kernel_size=1))
        if 'depth' in self.tasks and len(self.tasks) == 2:
            heads.append(nn.Sequential(
                nn.Conv2d(filter[0], 1, kernel_size=1),
                nn.Sigmoid())
            )#depth head
        if 'depth' in self.tasks and len(self.tasks) == 3:
            heads.append(nn.Sequential(
                nn.Conv2d(filter[0], 1, kernel_size=1),
                nn.ReLU())
            )
        if 'normal' in self.tasks:
            heads.append(nn.Sequential(
                nn.Conv2d(filter[0], 3, kernel_size=1),
                nn.Tanh())
            )#normal head
        for i in range(len(self.tasks)):
            self.nets[i].append(ConvLayer(filter[0], filter[0]))
            self.nets[i].append(heads[i])
        init_weights(self)

    def forward(self, x):
        logits_seg = x 
        logits_depth = x
        indices_A = []
        indices_B = []
        j = 1
        if len(self.tasks) == 2:
            for modA, modB in zip(self.nets[0], self.nets[1]):
                if isinstance(modA, DownSampleBlock):
                    logits_seg, idx_A, _ = modA(logits_seg)
                    logits_depth, idx_B, _ = modB(logits_depth)
                    indices_A.append(idx_A)
                    indices_B.append(idx_B)
                    logits_seg = self.alphas[0][0] * logits_seg + self.alphas[0][1] * logits_depth
                    logits_depth = self.alphas[1][0] * logits_depth + self.alphas[1][1] * logits_seg
                elif isinstance(modA, UpSampleBlock):
                    logits_seg, _ = modA(logits_seg, indices_A[-j])
                    logits_depth, _ = modB(logits_depth, indices_B[-j])
                    j += 1
                    logits_seg = self.alphas[0][0] * logits_seg + self.alphas[0][1] * logits_depth
                    logits_depth = self.alphas[1][0] * logits_depth + self.alphas[1][1] * logits_seg
                else: #isinstance(modA, ConvLayer) or head
                    logits_seg = modA(logits_seg)
                    logits_depth = modB(logits_depth)
            return logits_seg, logits_depth
        else: # 3 tasks
            logits_normal = x
            indices_C = []
            for modA, modB, modC in zip(self.nets[0], self.nets[1], self.nets[2]):
                if isinstance(modA, DownSampleBlock):
                    logits_seg, idx_A, _ = modA(logits_seg)
                    logits_depth, idx_B, _ = modB(logits_depth)
                    logits_normal, idx_C, _ = modC(logits_normal)
                    indices_A.append(idx_A)
                    indices_B.append(idx_B)
                    indices_C.append(idx_C)
                    logits_seg = self.alphas[0][0] * logits_seg + self.alphas[0][1] * logits_depth + self.alphas[0][2] * logits_normal
                    logits_depth = self.alphas[1][0] * logits_seg + self.alphas[1][1] * logits_depth + self.alphas[1][2] * logits_normal
                    logits_normal = self.alphas[2][0] * logits_seg + self.alphas[2][1] * logits_depth + self.alphas[2][2] * logits_normal
                elif isinstance(modA, UpSampleBlock):
                    logits_seg, _ = modA(logits_seg, indices_A[-j])
                    logits_depth, _ = modB(logits_depth, indices_B[-j])
                    logits_normal, _ = modC(logits_normal, indices_C[-j])
                    j += 1
                    logits_seg = self.alphas[0][0] * logits_seg + self.alphas[0][1] * logits_depth + self.alphas[0][2] * logits_normal
                    logits_depth = self.alphas[1][0] * logits_seg + self.alphas[1][1] * logits_depth + self.alphas[1][2] * logits_normal
                    logits_normal = self.alphas[2][0] * logits_seg + self.alphas[2][1] * logits_depth + self.alphas[2][2] * logits_normal
                else:
                    logits_seg = modA(logits_seg)
                    logits_depth = modB(logits_depth)
                    logits_normal = modC(logits_normal)
            return logits_seg, logits_depth, logits_normal