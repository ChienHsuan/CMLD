import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..evaluation_metrics import accuracy
from ..loss import CrossEntropyLabelSmooth, MultiSimilarityLoss
from ..utils.meters import AverageMeter


class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_ms = MultiSimilarityLoss().cuda()

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_ms = AverageMeter()
        transfer_loss_ = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)
            s_features, s_cls_out = self.model(s_inputs)
            # target samples: only forward
            t_features, _ = self.model(t_inputs)

            ############################################
            target_softmax = F.softmax(s_cls_out, dim=1)
            transfer_loss = -torch.norm(target_softmax,'nuc')/target_softmax.shape[0]
            ############################################

            # backward main #
            loss_ce, loss_ms, prec1 = self._forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_ms + 2 * transfer_loss

            losses_ce.update(loss_ce.item())
            losses_ms.update(loss_ms.item())
            transfer_loss_.update(transfer_loss.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_ms {:.3f} ({:.3f})\t'
                      'transfer_loss_ {:.3f}\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_ms.val, losses_ms.avg,
                              transfer_loss_.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        loss_ms = self.criterion_ms(s_features, targets)

        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_ms, prec
