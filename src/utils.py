import torch
import torch.nn as nn
from torchmetrics import Metric
from torchmetrics.functional.segmentation import mean_iou
from torchmetrics.functional.classification import multiclass_accuracy
from torchmetrics.functional.regression import mean_absolute_error
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mean_absolute_relative_error(preds, target):
    sum_abs_target = torch.sum(torch.abs(target))
    return torch.sum(torch.abs(preds - target))/sum_abs_target

class MeanAbsoluteRelativeError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_rel_err", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.sum_rel_err += mean_absolute_relative_error(preds, target)
        self.num_obs += target.shape[0]

    def compute(self):
        return self.sum_rel_err / self.num_obs
    
# Check if we have to module the difference by pi or not
def angle_distance(preds, target):
    preds_angle = torch.arccos(preds)*180/torch.pi
    target_angle = torch.arccos(target)*180/torch.pi
    print(f"Preds: {preds_angle}")
    print(f"Targets: {target_angle}")
    angle_diff = torch.abs(preds_angle - target_angle)
    # print(f"Angle diff: {angle_diff}")
    # print(f"Mean {torch.mean(angle_diff)}")
    # print(f"Median {torch.median(angle_diff)}")
    # print(f"Tolls {[torch.sum(angle_diff <= toll)/angle_diff.numel() for toll in [11.25, 22.5, 30]]}")
    return angle_diff

class AngleDistance(Metric):
    def __init__(self, tolls=[11.25, 22.5, 30]):
        super().__init__()
        self.tolls = tolls
        self.add_state("angle_mean", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("angle_median", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("angle_tolls", default=torch.tensor([0.0 for _ in range(len(tolls))]), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        angle_diff = angle_distance(preds, target)
        self.angle_mean += torch.mean(angle_diff)
        self.angle_median += torch.median(angle_diff)
        self.angle_tolls += torch.tensor([torch.sum(angle_diff <= toll)/angle_diff.numel() for toll in self.tolls])
        self.num_obs += target.shape[0]

    def compute(self):
        return self.angle_mean/self.num_obs, self.angle_median/self.num_obs, self.angle_tolls
    
class DotProductLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x, y):
        x_norm = F.normalize(x, p=2, dim=1)
        B, C, H, W = x.shape
        x_flat = x_norm.view(B, 1, C*H*W)
        y_flat = y.view(B, C*H*W, 1)
        loss = -(1/H*W)*torch.matmul(x_flat, y_flat).squeeze(1)
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

def add_plt(plt, data):
    for k in data.keys():
        plt[k].append(data[k].compute().cpu()) if isinstance(data[k], Metric) else plt[k].append(data[k])

def plot_dict(plt_dict, path=None):
    nrows, ncols = len(plt_dict)//2, 2
    _, ax = plt.subplots(nrows, ncols)
    for i, k in enumerate(plt_dict.keys()):
        ax[i//ncols][i%ncols].plot(plt_dict[k])
        ax[i//ncols][i%ncols].set_title(k)
    plt.savefig(path) if path else None

def compute_lambdas(losses_seg, losses_depth, T, K):
    w_seg = np.mean(losses_seg['new']) / np.mean(losses_seg['old'])
    w_depth = np.mean(losses_depth['new']) / np.mean(losses_depth['old'])
    w = F.softmax(torch.tensor([w_seg/T, w_depth/T]), dim=0)*K
    return w

def update_stats(stats, x, y, stats_keys):
    for k in stats_keys:
        stats[k].update(x, y)

def reset_stats(stats):
    for k in stats.keys():
        stats[k].reset()

def update_losses(losses_seg, losses_depth):
    losses_seg['old'] = losses_seg['new']
    losses_depth['old'] = losses_depth['new']
    losses_seg['new'] = []
    losses_depth['new'] = []

def save_model_opt(model, opt, epochs):
    torch.save(model.state_dict(), f"./models/{model.name}/{model.name}_train{epochs}.pth")
    torch.save(opt.state_dict(), f"./models/{model.name}/{model.name}_opt_train{epochs}.pth")

def ignore_index_seg(preds_seg, y_seg):
    preds_seg_flat = preds_seg.view(-1)
    y_seg_flat = y_seg.view(-1)
    pos_idx = torch.where(y_seg_flat != -1)
    preds_seg_flat = preds_seg_flat[pos_idx[0]].unsqueeze(0)
    y_seg_flat = y_seg_flat[pos_idx[0]].unsqueeze(0)
    return preds_seg_flat, y_seg_flat

def save_results(model_name):
    if not os.path.exists(f"./models/{model_name}"): 
        os.makedirs(f"./models/{model_name}")
    plt.savefig(f"./models/{model_name}/{model_name}_results.png")
    
def visualize_results_singletask(model, img_x, img_y, device, save=False):
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        img_x = img_x.to(device).to(torch.float)
        output = model(img_x.unsqueeze(0))
        
        plt.imshow(img_x.cpu().permute(1, 2, 0))
        _, ax = plt.subplots(1, 2, figsize=(11, 7))
        if model.name == 'depthnet':
            pred = output.squeeze(0,1).cpu()
            img_y = img_y.to(torch.float)
            ax[0].imshow(img_y, cmap='gray')
            ax[0].set_title('Ground Truth Depth')
            ax[1].imshow(pred.detach().numpy(), cmap='gray')
            ax[1].set_title('Predicted Depth')
            plt.show()
            print(f"Mean Absolute Error: {mean_absolute_error(pred, img_y).item()}")
            print(f"Mean Absolute Relative Error: {mean_absolute_relative_error(pred, img_y).item()}")
        else:
            pred = torch.argmax(output, dim=1).squeeze(0).cpu()
            pred_seg_flat, img_seg_flat = ignore_index_seg(pred, img_y)
            idx = img_y==-1
            img_y = img_y.to(torch.long)
            img_y[idx] = 0
            ax[0].imshow(img_y)
            ax[0].set_title('Ground Truth Segmentation')
            ax[1].imshow(pred.detach().numpy())
            ax[1].set_title('Predicted Segmentation')
            plt.show()
            print(f"Accuracy: {multiclass_accuracy(pred, img_y, num_classes=model.classes, multidim_average='global', average='micro').item()}")
            print(f"Mean IoU: {mean_iou(pred_seg_flat, img_seg_flat, num_classes=model.classes, per_class=False, include_background=False, input_format='index').item()}")
        save_results(model.name) if save else None

def visualize_results_multitask(model, img, img_seg, img_dis, device, save=False):
    with torch.no_grad():
        model.eval()
        img = img.to(device).to(torch.float)
        img_seg = img_seg.to(torch.long)
        img_dis = img_dis.to(torch.float)
        output_seg, output_dis = model(img.unsqueeze(0))
        pred_seg = torch.argmax(output_seg, dim=1).squeeze(0).cpu()
        pred_dis = output_dis.squeeze(0, 1).cpu()
        pred_seg_flat, img_seg_flat = ignore_index_seg(pred_seg, img_seg)
        idx = img_seg==-1
        img_seg[idx] = 0

        plt.imshow(img.cpu().permute(1, 2, 0))

        _, ax = plt.subplots(2, 2, figsize=(11, 7))
        ax[0][0].imshow(img_seg)
        ax[0][0].set_title('Ground Truth Segmentation')
        ax[0][1].imshow(pred_seg.detach().numpy())
        ax[0][1].set_title('Predicted Segmentation')
        ax[1][0].imshow(img_dis, cmap='gray')
        ax[1][0].set_title('Ground Truth Depth')
        ax[1][1].imshow(pred_dis.detach().numpy(), cmap='gray')
        ax[1][1].set_title('Predicted Depth')
        plt.show()
        print(f"Accuracy: {multiclass_accuracy(pred_seg_flat, img_seg_flat, num_classes=model.classes, multidim_average='global', average='micro').item()}")
        print(f"Mean IoU: {mean_iou(pred_seg_flat, img_seg_flat, num_classes=model.classes, per_class=False, include_background=False, input_format='index').item()}")
        print(f"Mean Absolute Error: {mean_absolute_error(pred_dis, img_dis).item()}")
        print(f"Mean Absolute Relative Error: {mean_absolute_relative_error(pred_dis, img_dis).item()}")
        save_results(model.name) if save else None