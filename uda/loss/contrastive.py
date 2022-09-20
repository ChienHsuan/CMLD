import torch
from torch import nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, num_positive=4, num_negative=10, num_negative_min=10, margin=0.3, T=1.0):
        super(ContrastiveLoss, self).__init__()
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.num_negative_min = num_negative_min
        self.margin = margin
        self.T = T

    def forward(self, out_feats, out_labels, mem_feats, mem_label, feat_normalize=True):
        B = out_feats.shape[0]
        N = mem_feats.shape[0]
        if feat_normalize:
            out_feats = F.normalize(out_feats, dim=1)
            mem_feats = F.normalize(mem_feats, dim=1)
        mat_sim = torch.matmul(out_feats, mem_feats.transpose(0, 1))
        mat_eq = mem_label.expand(B, N).eq(out_labels.expand(N, B).t()).float()
        
        # negative sample num safety check
        if (N - N/B*self.num_positive) < self.num_negative:
            num_negative = self.num_negative_min
        else:
            num_negative = self.num_negative

        # batch hard
        hard_p, hard_n = self.batch_hard(mat_sim, mat_eq, self.num_positive, num_negative, indice=False)
        hard_p = hard_p.view(B, self.num_positive)
        hard_n = hard_n.view(B, num_negative)

        total_loss = 0
        for pos_num in range(self.num_positive):
            per_h_pos = hard_p[:, pos_num].view(B, 1) - self.margin
            out = torch.cat((per_h_pos, hard_n), dim=1) / self.T
            triple_dist = F.log_softmax(out, dim=1)
            triple_dist_ref = torch.zeros_like(triple_dist)
            triple_dist_ref[:, 0] = 1.0
            loss = (- triple_dist_ref * triple_dist).sum(1).mean()
            total_loss = total_loss + loss / self.num_positive

        return total_loss

    def batch_hard(self, mat_sim, mat_eq, num_positive, num_negative, indice=False):
        # hardest positive
        sorted_mat_sim, positive_indices = torch.sort(mat_sim + (9999999.) * (1 - mat_eq), dim=1,
                                                           descending=False)
        hard_p = sorted_mat_sim[:, :num_positive]
        hard_p_indice = positive_indices[:, :num_positive]

        # some hard negatives
        sorted_mat_distance, negative_indices = torch.sort(mat_sim + (-9999999.) * (mat_eq), dim=1,
                                                           descending=True)
        hard_n = sorted_mat_distance[:, :num_negative]
        hard_n_indice = negative_indices[:, :num_negative]

        if indice:
            return hard_p, hard_n, hard_p_indice, hard_n_indice
        return hard_p, hard_n


class PNContrastiveLoss(nn.Module):
    def __init__(self, num_positive=4, num_negative=10, num_negative_min=10, T=1.0):
        super(PNContrastiveLoss, self).__init__()
        self.num_positive = num_positive
        self.num_negative = num_negative
        self.num_negative_min = num_negative_min
        self.T = T

    def forward(self, out_feats, out_labels, mem_feats, mem_label, feat_normalize=True):
        B = out_feats.shape[0]
        N = mem_feats.shape[0]
        if feat_normalize:
            out_feats = F.normalize(out_feats, dim=1)
            mem_feats = F.normalize(mem_feats, dim=1)
        mat_sim = torch.matmul(out_feats, mem_feats.transpose(0, 1))
        mat_eq = mem_label.expand(B, N).eq(out_labels.expand(N, B).t()).float()
        
        # negative sample num safety check
        if (N - N/B*self.num_positive) < self.num_negative:
            num_negative = self.num_negative_min
        else:
            num_negative = self.num_negative

        # batch hard
        hard_p, hard_n = self.batch_hard(mat_sim, mat_eq, self.num_positive, num_negative, indice=False)
        hard_p = hard_p.view(B, self.num_positive)
        hard_n = hard_n.view(B, num_negative)

        total_loss = 0
        for pos_num in range(self.num_positive):
            per_h_pos = hard_p[:, pos_num].view(B, 1)
            out = torch.cat((per_h_pos, hard_n), dim=1) / self.T
            triple_dist = F.log_softmax(out, dim=1)
            triple_dist_ref = torch.zeros_like(triple_dist)
            triple_dist_ref[:, 0] = 1.0
            loss = (- triple_dist_ref * triple_dist).sum(1).mean()
            total_loss = total_loss + loss / self.num_positive

        for neg_num in range(self.num_negative):
            per_h_neg = hard_n[:, neg_num].view(B, 1)
            out = torch.cat((hard_p, per_h_neg), dim=1) / self.T
            triple_dist = F.log_softmax(out, dim=1)
            triple_dist_ref = torch.zeros_like(triple_dist)
            triple_dist_ref[:, 0:self.num_positive] = 1.0
            loss = (- triple_dist_ref * triple_dist).sum(1).mean()
            total_loss = total_loss + loss / self.num_negative

        return total_loss

    def batch_hard(self, mat_sim, mat_eq, num_positive, num_negative, indice=False):
        # hardest positive
        sorted_mat_sim, positive_indices = torch.sort(mat_sim + (9999999.) * (1 - mat_eq), dim=1,
                                                           descending=False)
        hard_p = sorted_mat_sim[:, :num_positive]
        hard_p_indice = positive_indices[:, :num_positive]

        # some hard negatives
        sorted_mat_distance, negative_indices = torch.sort(mat_sim + (-9999999.) * (mat_eq), dim=1,
                                                           descending=True)
        hard_n = sorted_mat_distance[:, :num_negative]
        hard_n_indice = negative_indices[:, :num_negative]

        if indice:
            return hard_p, hard_n, hard_p_indice, hard_n_indice
        return hard_p, hard_n
