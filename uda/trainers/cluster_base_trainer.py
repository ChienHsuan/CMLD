import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..evaluation_metrics import accuracy
from ..loss import CrossEntropyLabelSmooth, SoftTripletLoss
from ..utils.meters import AverageMeter


class ClusterBaseTrainer(object):
    def __init__(self, model, target_loader, optimizer, num_cluster=500):
        super(ClusterBaseTrainer, self).__init__()
        self.model = model
        self.num_cluster = num_cluster

        self.target_loader = target_loader

        self.optimizer = optimizer

        self.criterion_ce = CrossEntropyLabelSmooth(num_cluster).cuda()
        self.criterion_tri = SoftTripletLoss(margin=0.0).cuda()

    def train(self, epoch, print_freq=1, train_iters=200):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        precisions = AverageMeter()

        self.target_loader.new_epoch()
        self.model.train()

        print(f'Epoch[{epoch}] start')
        end = time.time()
        for i in range(train_iters):
            target_inputs = self.target_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, targets = self._parse_data(target_inputs)

            # forward
            f_out_t, p_out_t = self.model(inputs)
            p_out_t = p_out_t[:,:self.num_cluster]

            loss_ce = self.criterion_ce(p_out_t, targets)
            loss_tri = self.criterion_tri(f_out_t, f_out_t, targets)
            loss = loss_ce + loss_tri

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            prec, = accuracy(p_out_t.data, targets.data)

            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            precisions.update(prec[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tri {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              precisions.val, precisions.avg))
        
        return precisions.avg

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets
