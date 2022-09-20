import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..evaluation_metrics import accuracy
from ..loss import CrossEntropyLabelSmooth, SoftEntropy, SoftTripletLoss, ContrastiveLoss, DistillKL, CRDLoss
from ..utils.meters import AverageMeter


class KDTrainer(object):
    def __init__(self, model_1, model_2, model_1_ema, model_2_ema, model_student, xbm_teacher_1,
                 xbm_teacher_2, target_loader, optimizer, num_cluster=500, alpha=0.999, num_negative=10,
                 contrastive_loss_margin=0.3, contrastive_loss_temperature_teacher=0.7):
        super(KDTrainer, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_1_ema = model_1_ema
        self.model_2_ema = model_2_ema
        self.model_student = model_student
        self.xbm_teacher_1 = xbm_teacher_1
        self.xbm_teacher_2 = xbm_teacher_2

        self.num_cluster = num_cluster
        self.alpha = alpha

        self.target_loader = target_loader

        self.optimizer = optimizer

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
        self.criterion_contrastive_teacher = ContrastiveLoss(num_positive=4, num_negative=num_negative,
                                                             margin=contrastive_loss_margin,
                                                             T=contrastive_loss_temperature_teacher)
        self.criterion_kd = DistillKL()

    def train(self, epoch, print_freq=1, train_iters=200):
        batch_time = AverageMeter()
        losses_ce_teacher_meter = AverageMeter()
        losses_tri_teacher_meter = AverageMeter()
        losses_tri_soft_teacher_meter = AverageMeter()
        losses_contrast_teacher_meter = AverageMeter()
        losses_ce_student_meter = AverageMeter()
        losses_kd_meter = AverageMeter()
        precisions_t1 = AverageMeter()
        precisions_t2 = AverageMeter()
        precisions_s1 = AverageMeter()

        self.target_loader.new_epoch()
        self.model_1.train()
        self.model_2.train()
        self.model_1_ema.train()
        self.model_2_ema.train()
        self.model_student.train()

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

            # student network forward
            f_stud_1, p_stud_1 = self.model_student(inputs_1)
            f_stud_2, p_stud_2 = self.model_student(inputs_2)
            p_stud_1 = p_stud_1[:,:self.num_cluster]
            p_stud_2 = p_stud_2[:,:self.num_cluster]
            
            # peer networks loss
            loss_ce = self.criterion_ce(p_1, labels) + self.criterion_ce(p_2, labels)
            loss_ce_soft = self.criterion_ce_soft(p_1, p_ema_2) + self.criterion_ce_soft(p_2, p_ema_1)
            loss_tri = self.criterion_tri(f_1, f_1, labels) + self.criterion_tri(f_2, f_2, labels)
            loss_tri_soft = self.criterion_tri_soft(f_1, f_ema_2, labels) + self.criterion_tri_soft(f_2, f_ema_1, labels)

            self.xbm_teacher_1.enqueue_dequeue(f_1.detach(), labels)
            xbm_feats_teacher_1, xbm_targets_teacher_1 = self.xbm_teacher_1.get_feats()
            self.xbm_teacher_2.enqueue_dequeue(f_2.detach(), labels)
            xbm_feats_teacher_2, xbm_targets_teacher_2 = self.xbm_teacher_2.get_feats()
            loss_contrast_teacher = self.criterion_contrastive_teacher(f_1, labels, xbm_feats_teacher_1, xbm_targets_teacher_1) \
                                    + self.criterion_contrastive_teacher(f_2, labels, xbm_feats_teacher_2, xbm_targets_teacher_2)

            loss_teacher = 0.5*loss_ce + 0.5*loss_ce_soft + 0.2*loss_tri + 0.8*loss_tri_soft + loss_contrast_teacher

            # student network loss
            loss_ce_stud = self.criterion_ce(p_stud_1, labels) + self.criterion_ce(p_stud_2, labels)
            loss_kd_stud = self.criterion_kd(p_stud_1, p_ema_1) + self.criterion_kd(p_stud_2, p_ema_2)

            loss_student = 0.3*loss_ce_stud + 2*loss_kd_stud
            loss = loss_teacher + loss_student

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update average models
            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(self.target_loader)+i)
            self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(self.target_loader)+i)

            t_prec_t1, = accuracy(p_1.data, labels)
            t_prec_t2, = accuracy(p_2.data, labels)
            t_prec_s1, = accuracy(p_stud_1.data, labels)
            losses_ce_teacher_meter.update(0.5*loss_ce.item() + 0.5*loss_ce_soft.item())
            losses_tri_teacher_meter.update(0.2*loss_tri.item())
            losses_tri_soft_teacher_meter.update(0.8*loss_tri_soft.item())
            losses_contrast_teacher_meter.update(loss_contrast_teacher.item())
            losses_ce_student_meter.update(0.3*loss_ce_stud.item())
            losses_kd_meter.update(2*loss_kd_stud.item())
            precisions_t1.update(t_prec_t1[0])
            precisions_t2.update(t_prec_t2[0])
            precisions_s1.update(t_prec_s1[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f}\t'
                      'Loss_ce {:.3f}/{:.3f}\t'
                      'Loss_tri {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Loss_contrast {:.3f}\t'
                      'Loss_kd {:.3f}\t'
                      'Prec {:.2%}/{:.2%}/{:.2%}\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val,
                              losses_ce_teacher_meter.avg, losses_ce_student_meter.avg,
                              losses_tri_teacher_meter.avg,
                              losses_tri_soft_teacher_meter.avg,
                              losses_contrast_teacher_meter.avg,
                              losses_kd_meter.avg,
                              precisions_t1.avg, precisions_t2.avg, precisions_s1.avg))

        return precisions_s1.avg

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


class CRDTrainer(object):
    def __init__(self, model_1, model_2, model_1_ema, model_2_ema, model_student, xbm_teacher_1,
                 xbm_teacher_2, target_loader, optimizer, num_cluster=500, alpha=0.999, num_negative=10,
                 contrastive_loss_margin=0.3, contrastive_loss_temperature_teacher=0.7, n_data=None,
                 feat_dim=512):
        super(CRDTrainer, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_1_ema = model_1_ema
        self.model_2_ema = model_2_ema
        self.model_student = model_student
        self.xbm_teacher_1 = xbm_teacher_1
        self.xbm_teacher_2 = xbm_teacher_2

        self.num_cluster = num_cluster
        self.alpha = alpha

        self.target_loader = target_loader

        self.optimizer = optimizer

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()
        self.criterion_contrastive_teacher = ContrastiveLoss(num_positive=4, num_negative=num_negative,
                                                             margin=contrastive_loss_margin,
                                                             T=contrastive_loss_temperature_teacher)
        self.criterion_kd = DistillKL()
        self.criterion_crd = CRDLoss(n_data=n_data, feat_dim=feat_dim)

    def train(self, epoch, print_freq=1, train_iters=200):
        batch_time = AverageMeter()
        losses_ce_teacher_meter = AverageMeter()
        losses_tri_teacher_meter = AverageMeter()
        losses_tri_soft_teacher_meter = AverageMeter()
        losses_contrast_teacher_meter = AverageMeter()
        losses_ce_student_meter = AverageMeter()
        losses_kd_meter = AverageMeter()
        precisions_t1 = AverageMeter()
        precisions_t2 = AverageMeter()
        precisions_s1 = AverageMeter()

        self.target_loader.new_epoch()
        self.model_1.train()
        self.model_2.train()
        self.model_1_ema.train()
        self.model_2_ema.train()
        self.model_student.train()

        print(f'Epoch[{epoch}] start')
        end = time.time()
        for i in range(train_iters):
            # two augmented view inputs
            target_inputs = self.target_loader.next()
            inputs_1, inputs_2, fnames, labels, _ = self._parse_data(target_inputs)
            
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

            # student network forward
            f_stud_1, p_stud_1 = self.model_student(inputs_1)
            f_stud_2, p_stud_2 = self.model_student(inputs_2)
            p_stud_1 = p_stud_1[:,:self.num_cluster]
            p_stud_2 = p_stud_2[:,:self.num_cluster]
            
            # peer networks loss
            loss_ce = self.criterion_ce(p_1, labels) + self.criterion_ce(p_2, labels)
            loss_ce_soft = self.criterion_ce_soft(p_1, p_ema_2) + self.criterion_ce_soft(p_2, p_ema_1)
            loss_tri = self.criterion_tri(f_1, f_1, labels) + self.criterion_tri(f_2, f_2, labels)
            loss_tri_soft = self.criterion_tri_soft(f_1, f_ema_2, labels) + self.criterion_tri_soft(f_2, f_ema_1, labels)

            self.xbm_teacher_1.enqueue_dequeue(f_1.detach(), labels)
            xbm_feats_teacher_1, xbm_targets_teacher_1 = self.xbm_teacher_1.get_feats()
            self.xbm_teacher_2.enqueue_dequeue(f_2.detach(), labels)
            xbm_feats_teacher_2, xbm_targets_teacher_2 = self.xbm_teacher_2.get_feats()
            loss_contrast_teacher = self.criterion_contrastive_teacher(f_1, labels, xbm_feats_teacher_1, xbm_targets_teacher_1) \
                                    + self.criterion_contrastive_teacher(f_2, labels, xbm_feats_teacher_2, xbm_targets_teacher_2)

            loss_teacher = 0.5*loss_ce + 0.5*loss_ce_soft + 0.2*loss_tri + 0.8*loss_tri_soft + loss_contrast_teacher

            # student network loss
            loss_ce_stud = self.criterion_ce(p_stud_1, labels) + self.criterion_ce(p_stud_2, labels)
            loss_kd_stud = self.criterion_kd(p_stud_1, p_ema_1) + self.criterion_kd(p_stud_2, p_ema_2)
            loss_crd_stud = self.criterion_crd(f_stud_1, f_ema_1.detach(), labels, fnames) \
                            + self.criterion_crd(f_stud_2, f_ema_2.detach(), labels, fnames)

            loss_student = 0.3*loss_ce_stud + 2*loss_kd_stud + 2*loss_crd_stud
            loss = loss_teacher + loss_student

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update average models
            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(self.target_loader)+i)
            self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(self.target_loader)+i)

            t_prec_t1, = accuracy(p_1.data, labels)
            t_prec_t2, = accuracy(p_2.data, labels)
            t_prec_s1, = accuracy(p_stud_1.data, labels)
            losses_ce_teacher_meter.update(0.5*loss_ce.item() + 0.5*loss_ce_soft.item())
            losses_tri_teacher_meter.update(0.2*loss_tri.item())
            losses_tri_soft_teacher_meter.update(0.8*loss_tri_soft.item())
            losses_contrast_teacher_meter.update(loss_contrast_teacher.item())
            losses_ce_student_meter.update(0.3*loss_ce_stud.item())
            losses_kd_meter.update(2*loss_kd_stud.item() + 2*loss_crd_stud.item())
            precisions_t1.update(t_prec_t1[0])
            precisions_t2.update(t_prec_t2[0])
            precisions_s1.update(t_prec_s1[0])

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f}\t'
                      'Loss_ce {:.3f}/{:.3f}\t'
                      'Loss_tri {:.3f}\t'
                      'Loss_tri_soft {:.3f}\t'
                      'Loss_contrast {:.3f}\t'
                      'Loss_kd {:.3f}\t'
                      'Prec {:.2%}/{:.2%}/{:.2%}\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val,
                              losses_ce_teacher_meter.avg, losses_ce_student_meter.avg,
                              losses_tri_teacher_meter.avg,
                              losses_tri_soft_teacher_meter.avg,
                              losses_contrast_teacher_meter.avg,
                              losses_kd_meter.avg,
                              precisions_t1.avg, precisions_t2.avg, precisions_s1.avg))

        return precisions_s1.avg

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
