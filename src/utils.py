import torch
import torch.nn as nn
from torchmetrics import Metric 
from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import MulticlassAccuracy 
from torchmetrics.regression import MeanAbsoluteError
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


#TODO: fix the application if the last layer has not a relu
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
    return torch.sum(torch.abs(preds - target)/target), target.shape[0]

class MeanAbsoluteRelativeError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("sum_rel_err", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        rel, obs = mean_absolute_relative_error(preds, target)
        self.sum_rel_err += rel
        self.num_obs += obs

    def compute(self):
        return self.sum_rel_err / self.num_obs
    
def angle_distance(preds, target):
    mask = mask_invalid_pixels(target)
    # preds_angle = torch.acos(preds)*180/torch.pi
    # target_angle = torch.acos(target)*180/torch.pi
    # # print(f"Preds: {preds_angle}")
    # # print(f"Targets: {target_angle}")
    # angle_diff = torch.abs(preds_angle - target_angle)
    # print(f"Angle diff: {angle_diff}")
    # print(f"Mean {torch.mean(angle_diff)}")
    # print(f"Median {torch.median(angle_diff)}")
    # print(f"Tolls {[torch.sum(angle_diff <= toll)/angle_diff.numel() for toll in [11.25, 22.5, 30]]}")
    dot_prod = torch.sum(preds*target, dim=1)
    angle_diff = torch.acos(torch.clamp(dot_prod.masked_select(mask), -1, 1)).rad2deg()
    return angle_diff, angle_diff.shape[0]

class AngleDistance(Metric):
    def __init__(self, tolls=[11.25, 22.5, 30]):
        super().__init__()
        self.tolls = tolls
        self.add_state("angle_mean", default=torch.tensor(0.0), dist_reduce_fx="sum")
        #self.add_state("angle_median", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("angle_median", default=[], dist_reduce_fx="sum")
        self.add_state("angle_tolls", default=torch.tensor([0.0 for _ in range(len(tolls))]), dist_reduce_fx="sum")
        self.add_state("num_obs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        angle_diff, obs = angle_distance(preds, target)
        #self.angle_mean += torch.mean(angle_diff)
        self.angle_mean += torch.sum(angle_diff)
        #self.angle_median += torch.median(angle_diff)
        self.angle_median.append(torch.median(angle_diff))
        #self.angle_tolls += torch.tensor([torch.sum(angle_diff <= toll)/angle_diff.numel() for toll in self.tolls]).to(self.angle_tolls.device)
        self.angle_tolls += torch.tensor([torch.sum(angle_diff <= toll) for toll in self.tolls]).to(self.angle_tolls.device)
        self.num_obs += obs

    def compute(self):
        #return {'mean':self.angle_mean/self.num_obs, 'median':self.angle_median/self.num_obs, 'tolls':self.angle_tolls}
        return {'mean':self.angle_mean/self.num_obs, 'median':torch.mean(torch.tensor(self.angle_median)), 'tolls':self.angle_tolls}

def plot_dict(plt_dict, path=None):
    for k in plt_dict.keys():
        nplots = len(plt_dict[k])
        if nplots == 1:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        elif nplots == 2:
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        else:
            ncols = 2
            nrows = nplots//2 if nplots%2 == 0 else nplots//2 + 1
            fig, ax = plt.subplots(nrows, ncols, figsize=(15, 15)) if nplots > 2 else plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(k)
        for i, t in enumerate(plt_dict[k].keys()):
            if nplots == 1:
                ax.plot(plt_dict[k][t])
                ax.set_title(t)
            elif nplots == 2:
                ax[i].plot(plt_dict[k][t]) 
                ax[i].set_title(t)
            else:
                if t == 'lambdas':
                    for l in plt_dict[k][t].keys():
                        ax[i//ncols][i%ncols].plot(plt_dict[k][t][l], label=l)
                    ax[i//ncols][i%ncols].legend(loc="upper left")
                else:
                    ax[i//ncols][i%ncols].plot(plt_dict[k][t])
                ax[i//ncols][i%ncols].set_title(t)
        plt.savefig(path + k + '.png') if path else None

def compute_lambdas(losses_new, losses_old, K, T=2):
    w = []
    for k in losses_new.keys():
        w_tmp = losses_new[k] / losses_old[k]
        w.append(w_tmp/T)
    w = F.softmax(torch.tensor(w), dim=0)*K
    return dict(zip(losses_new.keys(), w))

def update_stats(stats, x, y):
    for t in stats.keys():
        stats[t].update(x, y)

def reset_stats(stats):
    for k in stats.keys():
        for t in stats[k].keys():
            stats[k][t].reset()

# def ignore_index_seg(preds_seg, y_seg):
#     preds_seg_flat = preds_seg.view(-1)
#     y_seg_flat = y_seg.view(-1)
#     pos_idx = torch.where(y_seg_flat != -1)
#     preds_seg_flat = preds_seg_flat[pos_idx[0]].unsqueeze(0)
#     y_seg_flat = y_seg_flat[pos_idx[0]].unsqueeze(0)
#     return preds_seg_flat, y_seg_flat

def mask_invalid_pixels(y):
    mask = (y.sum(dim=1, keepdim=True) != 0).to(y.device) if len(y.shape) == 4 else (y != 0).to(y.device)
    return mask

def make_plt_dict(loss, stats, train, grad=None, lambdas=None):
    dict_str = 'train' if train else 'val'
    plt_dict = {f'loss_{dict_str}': loss}
    plt_dict[f'stats_{dict_str}'] = {}
    for k in stats.keys():
        for t in stats[k].keys():
            plt_dict[f'stats_{dict_str}'][t] = stats[k][t]
    if train:
        if grad:
            plt_dict[f'loss_{dict_str}']['grad'] = grad
        if lambdas:
            plt_dict[f'loss_{dict_str}']['lambdas'] = lambdas
    return plt_dict

def move_tensors(x, y_dict, device):
    x = x.to(device).to(torch.float)
    y_dict = {k: v.to(device).to(torch.float) for k, v in y_dict.items()}
    if 'segmentation' in y_dict.keys():
        y_dict['segmentation'] = y_dict['segmentation'].to(torch.long)
    return x, y_dict

def loss_handler(plt_losses, losses, writer, epoch, train=True, out=True):
    train_str = 'train' if train else 'val'
    for k in plt_losses.keys():
        print(f"{train_str} loss {k}: {losses[k]:.4f}") if out else None
        writer_string = f'{train_str}/loss/{k}'
        writer.add_scalar(writer_string, losses[k], epoch)
        plt_losses[k].append(losses[k])

def stats_handler(plt_stats, stats, writer, epoch, train=True, out=True):
    train_str = 'train' if train else 'val'
    for k in stats.keys():
        for t in stats[k].keys():
            stat_comp = stats[k][t].compute() if t != 'ad' else stats[k][t].compute()['mean']
            stat_comp = stat_comp.cpu().item()
            print(f"{t}: {stat_comp:.4f}") if out else None
            plt_stats[k][t].append(stat_comp)
            writer_string = f'{train_str}/stats/{t}'
            writer.add_scalar(writer_string, stat_comp, epoch)

def visualize_results(model, device, x, y, id_result, dwa_trained=False, save=True):
    with torch.no_grad():
        if save: 
            if len(model.tasks) == 1:
                path = f"../results/{model.name}"
            else:
                dwa_string = 'dwa' if dwa_trained else 'equal'
                path = f"../results/{model.name}_{dwa_string}"
            if not os.path.exists(path):
                os.makedirs(path)

        stats = {'depth':{'mae': MeanAbsoluteError(), 
                'mre': MeanAbsoluteRelativeError()},

                'segmentation':{'miou': MeanIoU(num_classes=model.classes, per_class=False, include_background=False, input_format='index'), 
                'pix_acc': MulticlassAccuracy(num_classes=model.classes, multidim_average='global', average='micro')},

                'normal':{'ad': AngleDistance()}}
        for t in stats.keys():
            for s in stats[t].keys():
                stats[t][s] = stats[t][s].to(device)
        model.to(device)
        B, _, _, _ = x.shape
        model.eval()
        # x = x.to(device).to(torch.float)
        x, y = move_tensors(x, y, device)
        output = model(x)
        if 'segmentation' in model.tasks:
            #out_seg_flat, y_seg_flat = ignore_index_seg(output['segmentation'], y['segmentation'])
            output['segmentation'] = torch.argmax(output['segmentation'], dim=1)
            mask = (y['segmentation'] != -1).to(torch.long).to(device)
            output_stats_seg = output['segmentation']*mask
            y_stats_seg = y['segmentation']*mask
            y['segmentation'] *= mask
        for i in range(B):
            for t in model.tasks:
                _, ax = plt.subplots(1, 2, figsize=(11, 7))
                y_plt = y[t][i].cpu().permute(1, 2, 0) if len(y[t].shape) == 4 else y[t][i].cpu()
                ax[0].imshow(y_plt)
                ax[0].set_title(f'Ground Truth {t}')
                out_plt = output[t][i].cpu().permute(1, 2, 0) if len(output[t].shape) == 4 else output[t][i].cpu()
                ax[1].imshow(out_plt)
                ax[1].set_title(f'Predicted {t}')
                plt.show()
                if save:
                    plt.savefig(f"{path}/{t}_results{id_result+i}.png")
                if t == 'segmentation':
                    output_stats, y_stats = output_stats_seg[i], y_stats_seg[i]
                else:
                    mask = mask_invalid_pixels(y[t])
                    output_stats, y_stats = output[t][i].masked_select(mask[i]), y[t][i].masked_select(mask[i])
                for s in stats[t].keys():
                    if s == 'ad':
                        for k in stats[t][s].keys():
                            print(f"{s} {k}: {stats[t][s][k](output_stats, y_stats).item()}")
                    else:
                        print(f"{s}: {stats[t][s](output_stats, y_stats).item()}")

# def save_results(model_name, dataset_name):
#     if not os.path.exists(f"../models/{dataset_name}/{model_name}"): 
#         os.makedirs(f"../models/{dataset_name}/{model_name}")
#     plt.savefig(f"../models/{model_name}/{model_name}_results.png")
    
# def visualize_results_singletask(model, img_x, img_y, device, save=False):
#     with torch.no_grad():
#         model.eval()
#         model = model.to(device)
#         img_x = img_x.to(device).to(torch.float)
#         output = model(img_x.unsqueeze(0))
        
#         plt.imshow(img_x.cpu().permute(1, 2, 0))
#         _, ax = plt.subplots(1, 2, figsize=(11, 7))
#         if model.name == 'depthnet':
#             pred = output.squeeze(0,1).cpu()
#             img_y = img_y.to(torch.float)
#             ax[0].imshow(img_y, cmap='gray')
#             ax[0].set_title('Ground Truth Depth')
#             ax[1].imshow(pred.detach().numpy(), cmap='gray')
#             ax[1].set_title('Predicted Depth')
#             plt.show()
#             print(f"Mean Absolute Error: {mean_absolute_error(pred, img_y).item()}")
#             print(f"Mean Absolute Relative Error: {mean_absolute_relative_error(pred, img_y).item()}")
#         else:
#             pred = torch.argmax(output, dim=1).squeeze(0).cpu()
#             pred_seg_flat, img_seg_flat = ignore_index_seg(pred, img_y)
#             idx = img_y==-1
#             img_y = img_y.to(torch.long)
#             img_y[idx] = 0
#             ax[0].imshow(img_y)
#             ax[0].set_title('Ground Truth Segmentation')
#             ax[1].imshow(pred.detach().numpy())
#             ax[1].set_title('Predicted Segmentation')
#             plt.show()
#             print(f"Accuracy: {multiclass_accuracy(pred, img_y, num_classes=model.classes, multidim_average='global', average='micro').item()}")
#             print(f"Mean IoU: {mean_iou(pred_seg_flat, img_seg_flat, num_classes=model.classes, per_class=False, include_background=False, input_format='index').item()}")
#         save_results(model.name) if save else None

# def visualize_results_multitask(model, img, img_seg, img_dis, device, save=False):
#     with torch.no_grad():
#         model.eval()
#         img = img.to(device).to(torch.float)
#         img_seg = img_seg.to(torch.long)
#         img_dis = img_dis.to(torch.float)
#         output_seg, output_dis = model(img.unsqueeze(0))
#         pred_seg = torch.argmax(output_seg, dim=1).squeeze(0).cpu()
#         pred_dis = output_dis.squeeze(0, 1).cpu()
#         pred_seg_flat, img_seg_flat = ignore_index_seg(pred_seg, img_seg)
#         idx = img_seg==-1
#         img_seg[idx] = 0

#         plt.imshow(img.cpu().permute(1, 2, 0))

#         _, ax = plt.subplots(2, 2, figsize=(11, 7))
#         ax[0][0].imshow(img_seg)
#         ax[0][0].set_title('Ground Truth Segmentation')
#         ax[0][1].imshow(pred_seg.detach().numpy())
#         ax[0][1].set_title('Predicted Segmentation')
#         ax[1][0].imshow(img_dis, cmap='gray')
#         ax[1][0].set_title('Ground Truth Depth')
#         ax[1][1].imshow(pred_dis.detach().numpy(), cmap='gray')
#         ax[1][1].set_title('Predicted Depth')
#         plt.show()
#         print(f"Accuracy: {multiclass_accuracy(pred_seg_flat, img_seg_flat, num_classes=model.classes, multidim_average='global', average='micro').item()}")
#         print(f"Mean IoU: {mean_iou(pred_seg_flat, img_seg_flat, num_classes=model.classes, per_class=False, include_background=False, input_format='index').item()}")
#         print(f"Mean Absolute Error: {mean_absolute_error(pred_dis, img_dis).item()}")
#         print(f"Mean Absolute Relative Error: {mean_absolute_relative_error(pred_dis, img_dis).item()}")
#         save_results(model.name) if save else None