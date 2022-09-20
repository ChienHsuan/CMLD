import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..evaluation_metrics import accuracy
from ..loss import CrossEntropyLabelSmooth, SoftEntropy, SoftTripletLoss, MoCo
from ..utils.meters import AverageMeter


class MoCoTrainer(object):
    def __init__(self, model_1, model_2, model_1_ema, model_2_ema, target_loader,
                 optimizer, num_cluster=500, alpha=0.999, temperature=0.07):
        super(MoCoTrainer, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_1_ema = model_1_ema
        self.model_2_ema = model_2_ema

        self.num_cluster = num_cluster
        self.alpha = alpha

        self.target_loader = target_loader

        self.optimizer = optimizer

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
        self.moco_1 = MoCo(model_1, model_1_ema, dim=num_cluster, T=temperature).cuda()
        self.moco_2 = MoCo(model_2, model_2_ema, dim=num_cluster, T=temperature).cuda()

    def train(self, epoch, print_freq=1, train_iters=200):
        batch_time = AverageMeter()
        losses_ce_teacher_meter = AverageMeter()
        losses_tri_teacher_meter = AverageMeter()
        losses_tri_soft_teacher_meter = AverageMeter()
        losses_contrast_teacher_meter = AverageMeter()
        precisions_t1 = AverageMeter()
        precisions_t2 = AverageMeter()

        self.target_loader.new_epoch()
        self.model_1.train()
        self.model_2.train()
        self.model_1_ema.train()
        self.model_2_ema.train()

        print(f'Epoch[{epoch}] start')
        end = time.time()
        for i in range(train_iters):
            # two augmented view inputs
            target_inputs = self.target_loader.next()
            inputs_1, inputs_2, _, labels, _ = self._parse_data(target_inputs)
            
            # average models forward
            f_ema_1, p_ema_1 = self.model_1_ema(inputs_1)
            f_ema_2, p_ema_2 = self.model_2_ema(inputs_2)
            p_ema_1 = p_ema_1[:,:self.num_cluster]
            p_ema_2 = p_ema_2[:,:self.num_cluster]

            # peer networks forward
            f_1, p_1 = self.model_1(inputs_1)
            f_2, p_2 = self.model_2(inputs_2)
            p_1 = p_1[:,:self.num_cluster]
            p_2 = p_2[:,:self.num_cluster]
            
            # peer networks loss
            loss_ce = self.criterion_ce(p_1, labels) + self.criterion_ce(p_2, labels)
            loss_ce_soft = self.criterion_ce_soft(p_1, p_ema_2) + self.criterion_ce_soft(p_2, p_ema_1)
            loss_tri = self.criterion_tri(f_1, f_1, labels) + self.criterion_tri(f_2, f_2, labels)
            loss_tri_soft = self.criterion_tri_soft(f_1, f_ema_2, labels) + self.criterion_tri_soft(f_2, f_ema_1, labels)

            output_1, target_1 = self.moco_1(inputs_1, inputs_2)
            output_2, target_2 = self.moco_2(inputs_1, inputs_2)
            loss_contrast_teacher = self.criterion_ce(output_1, target_1) + self.criterion_ce(output_2, target_2)

            loss = 0.5*loss_ce + 0.5*loss_ce_soft + 0.2*loss_tri + 0.8*loss_tri_soft + loss_contrast_teacher

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update average models
            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(self.target_loader)+i)
            self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(self.target_loader)+i)

            t_prec_t1, = accuracy(p_1.data, labels)
            t_prec_t2, = accuracy(p_2.data, labels)
            losses_ce_teacher_meter.update(0.5*loss_ce.item() + 0.5*loss_ce_soft.item())
            losses_tri_teacher_meter.update(0.2*loss_tri.item())
            losses_tri_soft_teacher_meter.update(0.8*loss_tri_soft.item())
            losses_contrast_teacher_meter.update(loss_contrast_teacher.item())
            precisions_t1.update(t_prec_t1[0])
            precisions_t2.update(t_prec_t2[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f}\t'
                      'Loss_ce {:.3f}\t'
                      'Loss_tri {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Loss_contrast {:.3f}\t'
                      'Prec {:.2%}/{:.2%}\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val,
                              losses_ce_teacher_meter.avg,
                              losses_tri_teacher_meter.avg,
                              losses_tri_soft_teacher_meter.avg,
                              losses_contrast_teacher_meter.avg,
                              precisions_t1.avg, precisions_t2.avg))

        return precisions_t1.avg, precisions_t2.avg

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, fnames, pids, camid = inputs
        imgs_1 = imgs_1.cuda()
        imgs_2 = imgs_2.cuda()
        labels = pids.cuda()
        return imgs_1, imgs_2, fnames, labels, camid


class CAPTrainer(object):
    def __init__(self, model_1, model_2, model_1_ema, model_2_ema, cam_memory,
                 target_loader, optimizer, num_cluster=500, alpha=0.999):
        super(CAPTrainer, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_1_ema = model_1_ema
        self.model_2_ema = model_2_ema
        self.cam_memory = cam_memory

        self.num_cluster = num_cluster
        self.alpha = alpha

        self.target_loader = target_loader

        self.optimizer = optimizer

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

    def train(self, epoch, intra_id_features=None, all_img_cams=None,
              all_pseudo_label=None, print_freq=1, train_iters=200):
        batch_time = AverageMeter()
        losses_ce_teacher_meter = AverageMeter()
        losses_tri_teacher_meter = AverageMeter()
        losses_tri_soft_teacher_meter = AverageMeter()
        losses_contrast_teacher_meter = AverageMeter()
        precisions_t1 = AverageMeter()
        precisions_t2 = AverageMeter()

        self.target_loader.new_epoch()
        self.model_1.train()
        self.model_2.train()
        self.model_1_ema.train()
        self.model_2_ema.train()

        print(f'Epoch[{epoch}] start')
        end = time.time()
        for i in range(train_iters):
            # two augmented view inputs
            target_inputs = self.target_loader.next()
            inputs_1, inputs_2, _, labels, cids = self._parse_data(target_inputs)
            
            # average models forward
            f_ema_1, p_ema_1 = self.model_1_ema(inputs_1)
            f_ema_2, p_ema_2 = self.model_2_ema(inputs_2)
            p_ema_1 = p_ema_1[:,:self.num_cluster]
            p_ema_2 = p_ema_2[:,:self.num_cluster]

            # peer networks forward
            f_1, p_1 = self.model_1(inputs_1)
            f_2, p_2 = self.model_2(inputs_2)
            p_1 = p_1[:,:self.num_cluster]
            p_2 = p_2[:,:self.num_cluster]
            
            # peer networks loss
            loss_ce = self.criterion_ce(p_1, labels) + self.criterion_ce(p_2, labels)
            loss_ce_soft = self.criterion_ce_soft(p_1, p_ema_2) + self.criterion_ce_soft(p_2, p_ema_1)
            loss_tri = self.criterion_tri(f_1, f_1, labels) + self.criterion_tri(f_2, f_2, labels)
            loss_tri_soft = self.criterion_tri_soft(f_1, f_ema_2, labels) + self.criterion_tri_soft(f_2, f_ema_1, labels)

            loss_contrast_teacher = self.cam_memory(f_1, labels, cams=cids, epoch=epoch,
                                                    all_pseudo_label=all_pseudo_label, all_img_cams=all_img_cams,
                                                    batch_ind=i, init_intra_id_feat=intra_id_features) \
                                    + self.cam_memory(f_2, labels, cams=cids, epoch=epoch,
                                                    all_pseudo_label=all_pseudo_label, all_img_cams=all_img_cams,
                                                    batch_ind=i, init_intra_id_feat=intra_id_features)

            loss = 0.5*loss_ce + 0.5*loss_ce_soft + 0.2*loss_tri + 0.8*loss_tri_soft + loss_contrast_teacher

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update average models
            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(self.target_loader)+i)
            self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(self.target_loader)+i)

            t_prec_t1, = accuracy(p_1.data, labels)
            t_prec_t2, = accuracy(p_2.data, labels)
            losses_ce_teacher_meter.update(0.5*loss_ce.item() + 0.5*loss_ce_soft.item())
            losses_tri_teacher_meter.update(0.2*loss_tri.item())
            losses_tri_soft_teacher_meter.update(0.8*loss_tri_soft.item())
            losses_contrast_teacher_meter.update(loss_contrast_teacher.item())
            precisions_t1.update(t_prec_t1[0])
            precisions_t2.update(t_prec_t2[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f}\t'
                      'Loss_ce {:.3f}\t'
                      'Loss_tri {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Loss_contrast {:.3f}\t'
                      'Prec {:.2%}/{:.2%}\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val,
                              losses_ce_teacher_meter.avg,
                              losses_tri_teacher_meter.avg,
                              losses_tri_soft_teacher_meter.avg,
                              losses_contrast_teacher_meter.avg,
                              precisions_t1.avg, precisions_t2.avg))

        return precisions_t1.avg, precisions_t2.avg

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, fnames, pids, camid = inputs
        imgs_1 = imgs_1.cuda()
        imgs_2 = imgs_2.cuda()
        labels = pids.cuda()
        camid = camid.cuda()
        return imgs_1, imgs_2, fnames, labels, camid
