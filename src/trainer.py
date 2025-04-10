import torch
import torch.nn as nn
from utils import plot_dict, compute_lambdas, reset_stats, update_stats, mask_invalid_pixels, make_plt_dict, loss_handler, stats_handler, move_tensors, build_stats_dict
from basic_modules import L1Loss, DotProductLoss
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, opt, dataset_name, device, dwa=False, save_path='./'):
        self.path = save_path + 'models/'
        self.device = device
        self.model = model.to(self.device)
        self.multitask = True if len(self.model.tasks) > 1 else False
        self.opt = opt
        self.dataset_name = dataset_name
        self.loss_fn = {}
        self.dwa = dwa
        self.stats = build_stats_dict(self.model, self.device)
        
        for t in model.tasks:
            if t == 'segmentation':
                self.loss_fn[t] = nn.CrossEntropyLoss(ignore_index=-1)
            elif t =='depth':
                self.loss_fn[t] = L1Loss()
            elif t == 'normal':
                self.loss_fn[t] = DotProductLoss()
            else:
                raise ValueError("Invalid task")
            
        self.track_losses_old = {t: None for t in model.tasks}
        self.track_losses_new = {t: None for t in model.tasks}
        self.plt_loss_train = {k: [] for k in self.model.tasks + ['total']} if self.multitask else {k: [] for k in self.model.tasks}
        self.plt_stats = {k: {t: [] for t in self.stats[k].keys()} for k in self.stats.keys()}
        self.plt_grad = []
        self.plt_lambdas = {k:[1] for k in self.model.tasks}
        self.dwa_string = '_dwa' if self.dwa else '_equal'
        self.dwa_string = '' if not self.multitask else self.dwa_string
        self.writer = SummaryWriter(save_path + f'/runs/{dataset_name}/{self.model.name}{self.dwa_string}')
        self.lambdas = {k: torch.tensor(1).to(self.device).to(torch.float) for k in self.model.tasks}

    # Convention: y_list = [y_seg, y_dis, y_normal]
    def _compute_loss(self, x, y_dict, stats):
        y_preds = self.model(x)
        if 'depth' in y_preds.keys():
            y_preds['depth'] = y_preds['depth'].squeeze(1)
        losses = {k: l(y_preds[k], y_dict[k]) for k, l in self.loss_fn.items()}

        for t in y_preds.keys():
            if t == 'segmentation':
                preds_seg = torch.argmax(y_preds[t], dim=1)
                mask = (y_dict[t] != -1).to(torch.long).to(y_dict[t].device)
                update_stats(stats[t], preds_seg*mask, y_dict[t]*mask)
            elif t == 'depth':
                mask = mask_invalid_pixels(y_dict[t])
                update_stats(stats[t], y_preds[t].masked_select(mask), y_dict[t].masked_select(mask))
            else: # t == 'normal'
                update_stats(stats[t], y_preds[t], y_dict[t])
        return losses

    def _val_epoch(self, val_dl, stats_val):
        with torch.no_grad():
            self.model.eval()
            losses_epoch = {k: 0 for k in self.plt_loss_train}
            reset_stats(stats_val)

            for x, y_dict in tqdm(val_dl):
                x, y_dict = move_tensors(x, y_dict, self.device)
                losses = self._compute_loss(x, y_dict, stats_val)
                loss = sum(losses.values())
                for k in losses.keys():
                    losses_epoch[k] += losses[k].item()
                if self.multitask:
                    losses_epoch['total'] += loss.item()

            for k in losses_epoch.keys():
                losses_epoch[k] /= len(val_dl)

            return losses_epoch
        
    def _compute_grad(self):
        params = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]
        grad_norm = 0
        for p in params:
            p_grad = p.grad.detach().data.norm(2).item()
            grad_norm += p_grad**2
        return grad_norm**0.5

    def train(self, train_dl, val_dl=None, epochs=10, save=False, check=5, grad=True):
        if val_dl != None:
            plt_loss_val = {k: [] for k in self.plt_loss_train.keys()}
            plt_stats_val = {k: {t:[] for t in self.plt_stats[k].keys()} for k in self.plt_stats.keys()}
            stats_val = self.stats.copy()
            
        self.path += self.dataset_name
        self.path += f"/{self.model.name}{self.dwa_string}/"
        if save and not os.path.exists(self.path): 
            os.makedirs(self.path)
        for epoch in range(epochs):
            self.model.train()
            reset_stats(self.stats)
            losses_epoch = {k: 0 for k in self.plt_loss_train}

            for x, y_dict in tqdm(train_dl):
                self.opt.zero_grad()
                x, y_dict = move_tensors(x, y_dict, self.device)
                losses = self._compute_loss(x, y_dict, self.stats)
                loss = sum([self.lambdas[k]*losses[k] for k in losses.keys()])
                loss.backward()
                self.opt.step()
                if self.multitask:
                    losses_epoch['total'] += loss.item()
                for k in losses.keys():
                    losses_epoch[k] += losses[k].item()

            losses_epoch = {k: losses_epoch[k]/len(train_dl) for k in losses_epoch.keys()}

            if self.dwa:
                if epoch > 0:
                    for k in self.track_losses_old.keys():
                        self.track_losses_old[k] = self.track_losses_new[k]
                        self.track_losses_new[k] = losses_epoch[k]
                    self.lambdas = compute_lambdas(self.track_losses_new, self.track_losses_old, len(self.model.tasks))
                else:
                    for k in self.lambdas.keys():
                        self.track_losses_new[k] = losses_epoch[k]
                        self.lambdas[k] = torch.tensor(1).to(self.device).to(torch.float)
                for k in self.plt_lambdas.keys():
                    self.plt_lambdas[k].append(self.lambdas[k].item())

            loss_handler(self.plt_loss_train, losses_epoch, self.writer, epoch, train=True, out=False)
            stats_handler(self.plt_stats, self.stats, self.writer, epoch, train=True, out=False)
                
            if grad:
                grad_norm = self._compute_grad()
                self.plt_grad.append(grad_norm)
                writer_string = 'train/loss/grad'
                self.writer.add_scalar(writer_string, grad_norm, epoch) 

            #Print subroutine
            if epoch % check == 0:
                if self.multitask:
                    print(f"Epoch {epoch+1}/{epochs} - train total loss: {losses_epoch['total']:.4f}")
                    if self.dwa:
                        for k1, k2 in zip(self.lambdas.keys(), losses.keys()):
                            print(f"lambda_{k1} : {self.lambdas[k1]} - train loss {k2}: {losses_epoch[k2]:.4f}")
                    else:
                        for k in losses.keys():
                            print(f"train loss {k}: {losses_epoch[k]:.4f}")
                else:
                    task = self.model.tasks[0]
                    print(f"Epoch {epoch+1}/{epochs} - train loss {task}: {losses_epoch[task]:.4f}")

                for k in self.stats.keys():
                    for t in self.stats[k].keys():
                        stat_comp = self.stats[k][t].compute().cpu() if t != 'ad' else self.stats[k][t].compute()['mean'].cpu()
                        print(f"{t}: {stat_comp:.4f}")
                print(f"gradient norm: {grad_norm:.4f}\n") if grad else print("\n")
                torch.save(self.model.state_dict(), self.path + f"{self.model.name}_{epoch}.pth") if save else None
                    
            if val_dl != None and epoch % check == 0:
                losses = self._val_epoch(val_dl, stats_val)
                loss_handler(plt_loss_val, losses, self.writer, epoch, train=False, out=True)
                stats_handler(plt_stats_val, stats_val, self.writer, epoch, train=False, out=True)
                print("\n")

        plt_train_dict = make_plt_dict(self.plt_loss_train, self.plt_stats, train=True, lambdas=self.plt_lambdas if self.dwa else None, grad=self.plt_grad if grad else None)
        plot_dict(plt_train_dict, self.path)
        train_path = self.path + f"{self.model.name}_{epochs}.pth"
        torch.save(self.model.state_dict(), train_path) if save else None
        
        if val_dl != None:
            plt_val_dict = make_plt_dict(plt_loss_val, plt_stats_val, train=False)
            plot_dict(plt_val_dict, self.path)