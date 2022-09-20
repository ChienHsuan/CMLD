import torch
from torch import nn
import torch.nn.functional as F


def euclidean_dist(x, y):
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(x, y.t(), beta=1, alpha=-2)
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist

def cosine_dist(x, y):
    x_normed = F.normalize(x, p=2, dim=-1)
    y_normed = F.normalize(y, p=2, dim=-1)
    return 1 - torch.mm(x_normed, y_normed.t())

def _batch_hard(mat_distance, mat_similarity, indice=False):
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, 0]
	hard_p_indice = positive_indices[:, 0]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	hard_n = sorted_mat_distance[:, 0]
	hard_n_indice = negative_indices[:, 0]
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n


class TripletLoss(nn.Module):
    
    def __init__(self, margin=None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, global_feat, labels):
        dist_mat = euclidean_dist(global_feat, global_feat)
        assert dist_mat.size(0) == dist_mat.size(1)

        N = dist_mat.size(0)
        mat_sim = labels.expand(N, N).eq(labels.expand(N, N).t()).float()
        dist_ap, dist_an = _batch_hard(dist_mat, mat_sim)
        assert dist_an.size(0) == dist_ap.size(0)

        y = torch.ones_like(dist_ap)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss


class SoftTripletLoss(nn.Module):

	def __init__(self, margin=None, normalize_feature=False):
		super(SoftTripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature

	def forward(self, emb1, emb2, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb1 = F.normalize(emb1)
			emb2 = F.normalize(emb2)

		mat_dist = euclidean_dist(emb1, emb1)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
		assert dist_an.size(0)==dist_ap.size(0)
		triple_dist = torch.stack((dist_ap, dist_an), dim=1)
		triple_dist = F.log_softmax(triple_dist, dim=1)
		if (self.margin is not None):
			loss = (- self.margin * triple_dist[:,0] - (1 - self.margin) * triple_dist[:,1]).mean()
			return loss

		mat_dist_ref = euclidean_dist(emb2, emb2)
		dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N,1).expand(N,N))[:,0]
		dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N,1).expand(N,N))[:,0]
		triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
		triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

		loss = (- triple_dist_ref * triple_dist).mean(0).sum()
		return loss


class SoftConsistencyLoss(nn.Module):

	def __init__(self, temparature=1.):
		super(SoftConsistencyLoss, self).__init__()
		self.temparature = temparature
		self.criterion_kl = nn.KLDivLoss(reduction='batchmean')

	def forward(self, emb1, emb2, label):
		mat_dist_inputs = euclidean_dist(emb1, emb1)
		assert mat_dist_inputs.size(0) == mat_dist_inputs.size(1)
		inputs = F.log_softmax(mat_dist_inputs/self.temparature, dim=1)

		mat_dist_targets = euclidean_dist(emb2, emb2)
		assert mat_dist_targets.size(0) == mat_dist_targets.size(1)
		targets = F.softmax(mat_dist_targets/self.temparature, dim=1).detach()

		loss = self.criterion_kl(inputs, targets)
		return loss
