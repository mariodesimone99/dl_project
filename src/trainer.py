import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.segmentation import MeanIoU
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.regression import MeanAbsoluteError
from utils import MeanAbsoluteRelativeError, AngleDistance, DotProductLoss, add_plt, plot_dict, compute_lambdas, reset_stats, update_losses, save_model_opt, ignore_index_seg, update_stats
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

#TODO: check if the loss should take into account all the pixels or only the ones that are not -1 (see paper)

class Trainer:
    def __init__(self, model, opt, dataset_name, device):
        self.device = device
        self.model = model.to(self.device)
        self.opt = opt
        self.dataset_name = dataset_name
        self.loss_fn = {}
        # losses_keys = []
        stats_keys = []
        stats_values = []
        # if len(model.tasks) == 2: 
        #     self.loss_fn = [nn.CrossEntropyLoss(ignore_index=-1), nn.L1Loss()]
        # elif len(model.tasks) == 3:
        #     self.loss_fn = [nn.CrossEntropyLoss(ignore_index=-1), nn.L1Loss(), DotProductLoss()]

        # for t in model.tasks:
            # losses_keys.append(f"new_{t}")
            # losses_keys.append(f"old_{t}")
        for t in model.tasks:
            # if 'segmentation' in model.tasks:
            if t == 'segmentation':
                stats_keys.append("miou")
                stats_keys.append("pix_acc")
                stats_values.append(MeanIoU(num_classes=self.model.classes, per_class=False, include_background=False, input_format='index').to(self.device))
                stats_values.append(MulticlassAccuracy(num_classes=self.model.classes, multidim_average='global', average='micro').to(self.device))
                self.loss_fn[t] = nn.CrossEntropyLoss(ignore_index=-1)
                # self.loss_fn.append(nn.CrossEntropyLoss(ignore_index=-1))
            elif t =='depth':
                stats_keys.append("mae")
                stats_keys.append("mre")
                stats_values.append(MeanAbsoluteError().to(self.device))
                stats_values.append(MeanAbsoluteRelativeError().to(self.device))
                self.loss_fn[t] = nn.L1Loss()
                # self.loss_fn.append(nn.L1Loss())
            # if 'normal' in model.tasks:
            elif t == 'normal':
                stats_keys.append("ad")
                stats_values.append(AngleDistance().to(self.device))
                self.loss_fn[t] = DotProductLoss()
                # self.loss_fn.append(DotProductLoss())
            else:
                raise ValueError("Invalid task")
        self.track_losses_old = {t: [] for t in model.tasks}
        self.track_losses_new = {t: [] for t in model.tasks}
        self.plt_loss_train = {k: [] for k in self.model.tasks + ['total']}
        self.plt_stats = {k: [] for k in stats_keys}
        self.plt_stats_train = {k: [] for k in stats_keys}
        self.plt_grad = []
        self.plt_lambdas = {k:1 for k in self.model.tasks}

        # self.track_losses = {k: [] for k in losses_keys}
        self.stats = {k: v for k, v in zip(stats_keys, stats_values)}
        self.writer = SummaryWriter(f'../runs/{dataset_name}/{self.model.name}')
        self.lambdas = {k: 1 for k in self.model.tasks}
        #self.lambdas = np.array([1, 1]) if len(self.model.tasks) == 2 else np.array([1, 1, 1])
        
    # def _compute_loss_singletask(self, x, y, stats):
    #     x = x.to(self.device).to(torch.float)
    #     y = y.to(self.device)
    #     output = self.model(x)
        
    #     if isinstance(self.loss_fn, nn.CrossEntropyLoss):
    #         y = y.to(torch.long)
    #         loss = self.loss_fn(output, y)
    #         preds = torch.argmax(output, dim=1)
    #         preds_flat, y_flat = ignore_index_seg(preds, y)
    #         update_stats(stats, preds_flat, y_flat)
    #     else:
    #         y = y.to(torch.float)
    #         loss = self.loss_fn(output.squeeze(1), y)
    #         preds = output.squeeze(1)
    #         update_stats(stats, preds, y)
    #     return loss  

    #def _compute_loss_multitask(self, x, y_seg, y_dis, stats_seg, stats_depth):
    # Convention: y_list = [y_seg, y_dis, y_normal]
    def _compute_loss_multitask(self, x, y_dict, stats):
        x = x.to(self.device).to(torch.float)
        y_dict = {k: v.to(self.device) for k, v in y_dict.items()}
        if 'segmentation' in y_dict.keys():
            y_dict['segmentation'] = y_dict['segmentation'].to(torch.long)
        if 'depth' in y_dict.keys():
            y_dict['depth'] = y_dict['depth'].to(torch.float)
        if 'normal' in y_dict.keys():
            y_dict['normal'] = y_dict['normal'].to(torch.float)

        # y_seg = y_seg.to(self.device).to(torch.long)
        # y_dis = y_dis.to(self.device).to(torch.float)
        # loss_fn_seg = nn.CrossEntropyLoss(ignore_index=-1)
        # loss_fn_depth = nn.L1Loss()
        y_preds = self.model(x)
        #TODO: is it possible to add the following line to the model class?
        if 'depth' in y_preds.keys():
            y_preds['depth'] = y_preds['depth'].squeeze(1)
        losses = {k: l(y_preds[k], y_dict[k]) for k, l in self.loss_fn.items()}

        # output_seg, output_depth = self.model(x)
        # loss_seg = loss_fn_seg(output_seg, y_seg)
        # loss_depth = loss_fn_depth(output_depth.squeeze(1), y_dis)

        for t in y_preds.keys():
            if t == 'segmentation':
                preds_seg = torch.argmax(y_preds[t], dim=1)
                preds_seg_flat = preds_seg.view(-1)
                y_seg_flat = y_dict[t].view(-1)
                pos_idx = torch.where(y_seg_flat != -1)
                preds_seg_flat = preds_seg_flat[pos_idx[0]].unsqueeze(0)
                y_seg_flat = y_seg_flat[pos_idx[0]].unsqueeze(0)
                update_stats(stats, preds_seg_flat, y_seg_flat, ['miou', 'pix_acc'])
            elif t == 'depth':
                update_stats(stats, y_preds[t], y_dict[t], ['mae', 'mre'])
            else: # t == 'normal'
                update_stats(stats, y_preds[t], y_dict[t], ['ad'])
        #update_stats(stats_seg, preds_seg_flat, y_seg_flat)
        #update_stats(stats_depth, output_depth.squeeze(1), y_dis)
        return losses

    # def _train_singletask(self, train_dl, val_dl=None, epochs=10, save=False, check=5, grad=False):
    #     # plt_loss_train = []
    #     # plt_grad = []
    #     # if isinstance(self.loss_fn, nn.CrossEntropyLoss):
    #     #     miou = MeanIoU(num_classes=self.model.classes, per_class=False, include_background=False, input_format='index').to(self.device)
    #     #     pix_acc = MulticlassAccuracy(num_classes=self.model.classes, multidim_average='global', average='micro').to(self.device)
    #     #     stats = {'miou':miou, 'pix_acc':pix_acc}
    #     # else:
    #     #     mae = MeanAbsoluteError().to(self.device)
    #     #     mre = MeanAbsoluteRelativeError().to(self.device)
    #     #     stats = {'mae':mae, 'mre':mre}
    #     # stats_str = list(stats.keys())
    #     # plt_stats_train = {stats_str[0]: [], stats_str[1]: []}
    #     if val_dl != None:
    #         plt_loss_val = []
    #         plt_stats_val = {stats_str[0]: [], stats_str[1]: []}

    #     if save and not os.path.exists(f"./models/{self.model.name}"): 
    #         os.makedirs(f"./models/{self.model.name}")
    #     for epoch in range(epochs):
    #         self.model.train()
    #         reset_stats(stats)

    #         total_loss = 0
    #         for x, y_seg, y_dis in tqdm(train_dl):
    #             y = y_seg.squeeze(dim=1) if isinstance(self.loss_fn, nn.CrossEntropyLoss) else y_dis

    #             self.opt.zero_grad()
    #             loss = self._compute_loss_singletask(x, y, stats)
    #             loss.backward()
    #             self.opt.step()
                
    #             total_loss += loss.item()
    #         total_loss /= len(train_dl)
    #         writer_string = 'Train/Loss/Segmentation' if isinstance(self.loss_fn, nn.CrossEntropyLoss) else 'Train/Loss/Depth'
    #         self.writer.add_scalar(writer_string, total_loss, epoch)
    #         plt_loss_train.append(total_loss)
    #         add_plt(plt_stats_train, stats)
    #         if grad:
    #             grad_norm = self._compute_grad()
    #             plt_grad.append(grad_norm)
    #             self.writer.add_scalar('Train/Gradient', grad_norm, epoch)
    #         if epoch % check == 0:
    #             print(f"Epoch {epoch}/{epochs} - Train Loss: {total_loss:.4f}")
    #             for k in stats.keys():
    #                 print(f"{k}: {stats[k].compute().cpu()}")
    #             print(f"Gradient Norm: {grad_norm}\n")
    #         for k in stats.keys():
    #             self.writer.add_scalar(f'Train/{k}', stats[k].compute().cpu(), epoch)
    #         save_model_opt(self.model, self.opt, epoch) if save else None
                                    
    #         if val_dl != None and epoch % check == 0:
    #             losses_tmp, stats_tmp = self._val_epoch_singletask(val_dl, epoch)
    #             plt_loss_val.append(losses_tmp)
    #             for k in stats_tmp.keys():
    #                 plt_stats_val[k].append(stats_tmp[k].compute().cpu())

    #     _, ax = plt.subplots(2, 2, figsize=(40, 40))
    #     ax[0][0].plot(plt_loss_train)
    #     ax[0][0].set_title('Loss')
    #     ax[0][1].plot(plt_stats_train[stats_str[0]])
    #     ax[0][1].set_title(stats_str[0])
    #     ax[1][0].plot(plt_stats_train[stats_str[1]])
    #     ax[1][0].set_title(stats_str[1])
    #     if grad:
    #         ax[1][1].plot(plt_grad)
    #         ax[1][1].set_title('Gradient Norm')
    #     if save:
    #         plt.savefig(f"./models/{self.model.name}/{self.model.name}_train{epochs}.png")
    #         torch.save(self.model.state_dict(), f"./models/{self.model.name}/{self.model.name}_train{epochs}.pth")

    #     if val_dl != None:
    #         _, ax = plt.subplots(3, 1, figsize=(20, 20))
    #         ax[0].plot(plt_loss_val)
    #         ax[0].set_title('Loss')
    #         for i, k in enumerate(stats.keys()):
    #             ax[i+1].plot(plt_stats_val[k])
    #             ax[i+1].set_title(k)
    #         plt.savefig(f"./models/{self.model.name}/{self.model.name}_val{epochs}.png") if save else None

    # def _val_epoch_singletask(self, dl, epoch):
    #     with torch.no_grad():
    #         self.model.eval()
    #         total_loss = 0

    #         if isinstance(self.loss_fn, nn.CrossEntropyLoss):
    #             miou = MeanIoU(num_classes=self.model.classes, per_class=False, include_background=False, input_format='index').to(self.device)
    #             pix_acc = MulticlassAccuracy(num_classes=self.model.classes, multidim_average='global', average='micro').to(self.device)
    #             stats = {'miou': miou, 'pix_acc': pix_acc}
    #         else:
    #             mae = MeanAbsoluteError().to(self.device)
    #             mre = MeanAbsoluteRelativeError().to(self.device)
    #             stats = {'mae': mae, 'mre': mre}
    #         for x, y_seg, y_dis in tqdm(dl):
    #             y = y_seg.squeeze(dim=1) if isinstance(self.loss_fn, nn.CrossEntropyLoss) else y_dis
    #             loss = self._compute_loss_singletask(x, y, stats)
                    
    #             total_loss += loss.item()
    #         total_loss /= len(dl)
    #         writer_string = 'Val/Loss/Segmentation' if isinstance(self.loss_fn, nn.CrossEntropyLoss) else 'Val/Loss/Depth'
    #         self.writer.add_scalar(writer_string, total_loss, epoch)
    #         print("Val Loss: ", total_loss)
    #         for k in stats.keys():
    #             print(f"{k}: {stats[k].compute().cpu()}")
    #             self.writer.add_scalar(f'Val/{k}', stats[k].compute().cpu(), epoch)
    #         print("\n")
    #     return total_loss, stats

    def _train_multitask(self, train_dl, val_dl=None, epochs=10, save=False, check=5, grad=False, dwa=None):
        # lambdas = np.array([1, 1])
        # losses_seg = {'new': [], 'old': []}
        # losses_depth = {'new': [], 'old': []}
        # plt_losses_train = {'seg': [], 'depth': [], 'total': []}
        # plt_stats_train = {'miou': [], 'pix_acc': [], 'mae': [], 'mre': []}
        # plt_lambdas = {'lambda0': [], 'lambda1': []}
        # plt_grad = []

        # miou = MeanIoU(num_classes=self.model.classes, per_class=False, include_background=False, input_format='index').to(self.device)
        # pix_acc = MulticlassAccuracy(num_classes=self.model.classes, multidim_average='global', average='micro').to(self.device)
        # stats_seg = {'miou':miou, 'pix_acc':pix_acc}
        # mae = MeanAbsoluteError().to(self.device)
        # mre = MeanAbsoluteRelativeError().to(self.device)
        # stats_depth = {'mae':mae, 'mre':mre}
        # if len(self.model.tasks) == 3:
        #     ad = AngleDistance().to(self.device)
        #     stats_normal = {'ad': ad}
        update_lambdas = dwa if dwa else None
        if val_dl != None:
            # plt_losses_val = {'seg': [], 'depth': [], 'total': []}
            # plt_stats_val = {'miou': [], 'pix_acc': [], 'mae': [], 'mre': []}
            plt_losses_val = {k: [] for k in self.plt_loss_train.keys()}
            plt_stats_val = {k: [] for k in self.plt_stats.keys()}
            stats_val = self.stats.copy()
            
        if save and not os.path.exists(f"../models/{self.dataset_name}/{self.model.name}"): 
            os.makedirs(f"../models/{self.dataset_name}/{self.model.name}")
        old_losses = True
        count_losses = 0
        for epoch in range(epochs):
            self.model.train()

            # reset_stats(stats_seg)
            # reset_stats(stats_depth)
            reset_stats(self.stats)
            # if len(self.model.tasks) == 3:
            #     reset_stats(stats_normal)
        
            # total_loss = 0
            # total_loss_seg = 0
            # total_loss_depth = 0
            losses_epoch = {k: 0 for k in self.plt_loss_train}
            for x, y_dict in tqdm(train_dl):
                self.opt.zero_grad()
                
                # losses = self._compute_loss_multitask(x, y_seg, y_dis, stats_seg, stats_depth)
                losses = self._compute_loss_multitask(x, y_dict, self.stats)
                loss = torch.sum([self.lambdas[k]*losses[k] for k in losses.keys()]) if dwa else torch.sum(torch.tensor(list(losses.values()))) 
                # if len(self.model.tasks) == 3:
                #     loss_seg, loss_depth, loss_normal = self._compute_loss_multitask(x, y_seg, y_dis, stats_seg, stats_depth, stats_normal)
                #     loss = lambdas[0]*loss_seg + lambdas[1]*loss_depth + lambdas[2]*loss_normal
                # else:
                #     loss_seg, loss_depth = self._compute_loss_multitask(x, y_seg, y_dis, stats_seg, stats_depth)
                #     loss = lambdas[0]*loss_seg + lambdas[1]*loss_depth
                loss.backward()
                self.opt.step()

                #losses_seg['new'].append(loss_seg.item())
                #losses_depth['new'].append(loss_depth.item())
                if dwa:
                    for k in losses.keys():
                        self.track_losses_new[k].append(losses[k].item())
                    count_losses += 1
                    if count_losses == 2*update_lambdas and old_losses:
                        for k in self.track_losses_old.keys():
                            self.track_losses_old[k] = self.track_losses_new[k][0:update_lambdas]
                            self.track_losses_new[k] = self.track_losses_new[k][update_lambdas:]
                        self.lambdas = compute_lambdas(self.track_losses_new, self.track_losses_old, self.model.classes)
                        update_losses(self.track_losses_new, self.track_losses_old)
                        old_losses = False
                        count_losses = 0
                    elif count_losses == update_lambdas and not old_losses:
                        self.lambdas = compute_lambdas(self.track_losses_new, self.track_losses_old, self.model.classes)
                        update_losses(self.track_losses_new, self.track_losses_old)
                        count_losses = 0
                    for k in self.plt_lambdas.keys():
                        self.plt_lambdas[k].append(self.lambdas[k].item())

                # if len(losses_seg['new']) == 2*update_lambdas and len(losses_seg['old']) == 0:
                #     losses_seg['old'] = losses_seg['new'][0:update_lambdas]
                #     losses_depth['old'] = losses_depth['new'][0:update_lambdas]
                #     losses_seg['new'] = losses_seg['new'][update_lambdas:]
                #     losses_depth['new'] = losses_depth['new'][update_lambdas:]

                #     self.lambdas = compute_lambdas(losses_seg, losses_depth, T, self.model.classes)
                #     update_losses(losses_seg, losses_depth)

                # if len(losses_seg['new']) == update_lambdas and len(losses_seg['old']) == update_lambdas:
                #     self.lambdas = compute_lambdas(losses_seg, losses_depth, T, self.model.classes)
                #     update_losses(losses_seg, losses_depth)

                losses_epoch = {k: losses_epoch[k] + losses[k].item() for k in losses.keys}
                losses_epoch['total'] += loss.item()
        
                # total_loss += loss.item()
                # total_loss_seg += loss_seg.item()
                # total_loss_depth += loss_depth.item()

            # plt_lambdas['lambda0'].append(lambdas[0].item())
            # plt_lambdas['lambda1'].append(lambdas[1].item())
            
            losses_epoch = {k: losses_epoch[k]/len(train_dl) for k in losses_epoch.keys}
            # total_loss /= len(train_dl)
            # total_loss_seg /= len(train_dl)
            # total_loss_depth /= len(train_dl)
            for k in self.plt_loss_train.keys():
                self.writer.add_scalar(f'Train/Loss/{k}', losses_epoch[k], epoch)
                self.plt_loss_train[k].append(losses_epoch[k])
            # plt_losses_train['seg'].append(total_loss_seg)
            # plt_losses_train['depth'].append(total_loss_depth)
            # plt_losses_train['total'].append(total_loss)
            # print_stats = dict(stats_seg, **stats_depth)
            # add_plt(plt_stats_train, print_stats)
            for k in self.plt_stats.keys():
                self.plt_stats[k].append(self.stats[k].compute().cpu())
                self.writer.add_scalar(f'Train/{k}', self.stats[k].compute().cpu(), epoch)
            if grad:
                grad_norm = self._compute_grad()
                self.plt_grad.append(grad_norm)
                self.writer.add_scalar('Train/Gradient', grad_norm, epoch) 
            if epoch % check == 0:
                print(f"Epoch {epoch}/{epochs} - Train Total Loss: {losses_epoch['total']:.4f}")
                if dwa:
                    for k1, k2 in zip(self.lambdas.keys(), losses):
                        print(f"lambda_{k1} : {self.lambdas[k1]} - Train Loss {k2}: {losses_epoch[k2]:.4f}")
                # print(f"Lambda_0: {lambdas[0]} - Train Loss Segmentation: {total_loss_seg:.4f}")
                # print(f"Lambda_1: {lambdas[1]} - Train Loss Depth: {total_loss_depth:.4f}")
                # for k in print_stats.keys():
                #     print(f"{k}: {print_stats[k].compute().cpu()}")
                for k in self.stats:
                    print(f"{k}: {self.stats[k].compute().cpu()}")
                print(f"Gradient Norm: {grad_norm}\n") if grad else print("\n")
                save_model_opt(self.model, self.opt, self.dataset_name, epoch) if save else None
            # self.writer.add_scalar('Train/Loss/Total', total_loss, epoch)
            # self.writer.add_scalar('Train/Loss/Segmentation', total_loss_seg, epoch)
            # self.writer.add_scalar('Train/Loss/Depth', total_loss_depth, epoch)
            # for k in print_stats.keys():
            #     self.writer.add_scalar(f'Train/{k}', print_stats[k].compute().cpu(), epoch)
                    
            if val_dl != None and epoch % check == 0:
                # losses_tmp, stats_tmp = self._val_epoch_multitask(val_dl, epoch)
                # add_plt(plt_losses_val, losses_tmp)
                # add_plt(plt_stats_val, stats_tmp)
                losses, stats_tmp = self._val_epoch_multitask(val_dl, stats_val) 
                for k in losses.keys():
                    print(f"Val Loss {k}: {losses[k]:.4f}")
                    self.writer.add_scalar(f'Val/Loss/{k}', losses_epoch[k], epoch)
                for k in stats_tmp.keys():
                    plt_stats_val[k].append(stats_tmp[k].compute().cpu())
                    self.writer.add_scalar(f'Val/{k}', stats_val[k].compute().cpu(), epoch)
                    print(f"{k}: {stats_tmp[k].compute().cpu()}")
                print("\n")

        plt_train_dict = {**self.plt_loss_train, **self.plt_stats}
        plt_train_dict = {**plt_train_dict, **self.plt_lambdas} if dwa else plt_train_dict
        plt_train_dict = {**plt_train_dict, 'grad': self.plt_grad} if grad else plt_train_dict
        train_path = f"./models/{self.model.name}/{self.model.name}_train{epochs}.png"
        plot_dict(plt_train_dict, train_path)
        # nrows, ncols = len(plt_train_dict)//2, 2
        # _, ax = plt.subplots(nrows, ncols)
        # for i, k in enumerate(plt_train_dict.keys()):
        #     ax[i//ncols][i%ncols].plot(plt_train_dict[k])
        #     ax[i//ncols][i%ncols].set_title(k)
        # if save:
        #     plt.savefig(f"./models/{self.model.name}/{self.model.name}_train{epochs}.png")
        torch.save(self.model.state_dict(), f"../models/{self.dataset_name}/{self.model.name}/{self.model.name}_train{epochs}.pth") if save else None
        
        if val_dl != None:
            plt_val_dict = {**plt_losses_val, **plt_stats_val}
            val_path = f"./models/{self.model.name}/{self.model.name}_val{epochs}.png"
            plot_dict(plt_val_dict, val_path)
            # nrows, ncols = len(plt_val_dict)//2, 2
            # _, ax = plt.subplots(nrows, ncols)
            # for i, k in enumerate(plt_val_dict.keys()):
            #     ax[i//ncols][i%ncols].plot(plt_val_dict[k])
            #     ax[i//ncols][i%ncols].set_title(k)
            # plt.savefig(f"./models/{self.model.name}/{self.model.name}_val{epochs}.png") if save else None
            
        # fig, ax = plt.subplots(4, 2, figsize=(40, 40))

        # _, ax = plt.subplots(4, 2, figsize=(40, 40)) if not grad else plt.subplots(5, 2, figsize=(50, 50))
        # ax[0][0].plot(plt_losses_train['seg'])
        # ax[0][0].set_title('Segmentation Loss')
        # ax[0][1].plot(plt_losses_train['depth'])
        # ax[0][1].set_title('Depth Loss')
        # ax[1][0].plot(plt_lambdas['lambda0'])
        # ax[1][0].plot(plt_lambdas['lambda1'])
        # ax[1][0].set_title('Lambdas')
        # ax[1][1].plot(plt_losses_train['total'])
        # ax[1][1].set_title('Total Loss')
        # ax[2][0].plot(plt_stats_train['miou'])
        # ax[2][0].set_title('Mean IoU')
        # ax[2][1].plot(plt_stats_train['pix_acc'])
        # ax[2][1].set_title('Pixel Accuracy')
        # ax[3][0].plot(plt_stats_train['mae'])
        # ax[3][0].set_title('Mean Absolute Error')
        # ax[3][1].plot(plt_stats_train['mre'])
        # ax[3][1].set_title('Mean Absolute Relative Error')
        # if grad:
        #     ax[4][0].plot(plt_grad)
        #     ax[4][0].set_title('Gradient Norm')
        # if save:
        #     plt.savefig(f"./models/{self.model.name}/{self.model.name}_train{epochs}.png")
        #     torch.save(self.model.state_dict(), f"./models/{self.model.name}/{self.model.name}_train{epochs}.pth")

        # if val_dl != None:
        #     _, ax = plt.subplots(3, 1, figsize=(20, 20))
        #     ax[0].plot(plt_losses_val['seg'])
        #     ax[0].set_title('Segmentation Loss')
        #     ax[1].plot(plt_losses_val['depth'])
        #     ax[1].set_title('Depth Loss')
        #     ax[2].plot(plt_losses_val['total'])
        #     ax[2].set_title('Total Loss')
        #     plt.savefig(f"./models/{self.model.name}/{self.model.name}_val{epochs}.png") if save else None

    def _val_epoch_multitask(self, val_dl, stats_val):
        with torch.no_grad():
            self.model.eval()
            # total_loss = 0
            # total_loss_seg = 0
            # total_loss_depth = 0
            losses_epoch = {k: 0 for k in self.plt_loss_train}
            reset_stats(stats_val)

            # miou = MeanIoU(num_classes=self.model.classes, per_class=False, include_background=False, input_format='index').to(self.device)
            # pix_acc = MulticlassAccuracy(num_classes=self.model.classes, multidim_average='global', average='micro').to(self.device)
            # stats_seg = {'miou':miou, 'pix_acc':pix_acc}
            # mae = MeanAbsoluteError().to(self.device)
            # mre = MeanAbsoluteRelativeError().to(self.device)
            # stats_depth = {'mae':mae, 'mre':mre}
            for x, y_dict in tqdm(val_dl):
                # loss_seg, loss_depth = self._compute_loss_multitask(x, y_seg, y_dis, stats_seg, stats_depth)
                losses = self._compute_loss_multitask(x, y_dict, stats_val)
                loss = torch.sum(torch.tensor(list(losses.values()))) 
                for k in losses.keys():
                    losses_epoch[k] += losses[k].item()
                losses_epoch['total'] += loss.item()

                # total_loss += loss.item()
                # total_loss_seg += loss_seg.item()
                # total_loss_depth += loss_depth.item()
            for k in losses_epoch.keys():
                losses_epoch[k] /= len(val_dl)
                # print(f"Val Loss {k}: {losses_epoch[k]:.4f}")
            # total_loss /= len(val_dl)
            # total_loss_seg /= len(val_dl)
            # total_loss_depth /= len(val_dl)
            # self.writer.add_scalar('Val/Loss/Total', total_loss, epoch)
            # self.writer.add_scalar('Val/Loss/Segmentation', total_loss_seg, epoch)
            # self.writer.add_scalar('Val/Loss/Depth', total_loss_depth, epoch)
            # print(f"Val Total Loss: {total_loss:.4f}")
            # print(f"Val Loss Segmentation: {total_loss_seg:.4f}")
            # print(f"Val Loss Depth: {total_loss_depth:.4f}")
            # losses = {'total': total_loss, 'seg': total_loss_seg, 'depth': total_loss_depth}
            # stats_comp = dict(stats_seg, **stats_depth)
            # for k in stats_comp.keys():
            #     print(f"{k}: {stats_comp[k].compute().cpu()}")
            #     self.writer.add_scalar(f'Val/{k}', stats_comp[k].compute().cpu(), epoch)
            # print("\n")
            return losses, stats_val
        
    def _compute_grad(self):
        params = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        grad_norm = 0
        for p in params:
            p_grad = p.grad.detach().data.norm(2).item()
            grad_norm += p_grad**2
        return grad_norm**0.5

    def train(self, train_dl, val_dl=None, epochs=10, save=False, check=5, grad=True, dwa=None):
        if len(self.model.tasks) > 1:
           self._train_multitask(train_dl, val_dl, epochs, save, check, grad, dwa)
        else:
            self._train_singletask(train_dl, val_dl, epochs, save, check, grad)
       