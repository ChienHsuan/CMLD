import math
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


eps = 1e-7


class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)

        # sample index
        self.smaples_idx_dict = collections.defaultdict(lambda:-1)
        self.labels_list = np.ones(outputSize, dtype=np.int32) * -1
        self.times = 0

        # teacher
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize, device=torch.device('cuda')).mul_(2 * stdv).add_(-stdv))

        # student
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize, device=torch.device('cuda')).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, labels, fnames):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # y: the indices of these positive samples in the dataset, size [batch_size]
        # idx: the indice of one positive sample and the indices of negative samples, size [batch_size, nce_k+1]
        y, idx = self.get_index(labels, fnames)

        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2

    def get_index(self, labels, fnames):
        pos_index = []
        pos_neg_index = []
        for i, f_name in enumerate(fnames):
            label = labels[i].item()
            # positive idx
            p_idx = self.smaples_idx_dict[f_name]
            if p_idx == -1:
                p_idx = self.times
                self.smaples_idx_dict[f_name] = p_idx
                self.labels_list[p_idx] = label
                self.times += 1
            pos_index.append(p_idx)

            # negative idx
            n_idx = np.random.choice(np.nonzero(self.labels_list != label)[0],
                        size=int(self.params[0].item()), replace=False)
            pos_neg_index.append(np.concatenate(([p_idx], n_idx)))

        pos_index = torch.tensor(pos_index).cuda()
        pos_neg_index = torch.tensor(pos_neg_index).cuda()
        return pos_index, pos_neg_index


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side
    Args:
        feat_dim: the dimension of the projection space
        nce_k: number of negatives paired with each positive
        nce_t: the temperature
        nce_m: the momentum for updating the memory buffer
        n_data: the number of samples in the training set, therefor the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, n_data=None, nce_k=None, feat_dim=512, nce_t=0.07, nce_m=0.5):
        super(CRDLoss, self).__init__()
        if nce_k == None:
            nce_k = n_data // 3
        self.contrast = ContrastMemory(feat_dim, n_data, nce_k, nce_t, nce_m).cuda()
        self.criterion_t = ContrastLoss(n_data).cuda()
        self.criterion_s = ContrastLoss(n_data).cuda()

    def forward(self, f_s, f_t, labels, fnames):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            labels: size [batch_size]
            fnames: size [batch_size]
        Returns:
            The contrastive loss
        """

        f_s = F.normalize(f_s, p=2, dim=1)
        f_t = F.normalize(f_t, p=2, dim=1)
        out_s, out_t = self.contrast(f_s, f_t, labels, fnames)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss
