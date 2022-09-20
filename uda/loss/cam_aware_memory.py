import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def update_cam_feats(memory, cam_id, inputs, targets, alpha):
    for x, y in zip(inputs, targets):
        memory[cam_id][y] = alpha * memory[cam_id][y] + (1.0 - alpha) * x
        memory[cam_id][y] /= memory[cam_id][y].norm()


class CAPMemory(nn.Module):
    def __init__(self, beta=0.05, alpha=0.01, crosscam_epoch=5, bg_knn=50):
        super(CAPMemory, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = alpha  # Memory update rate
        self.beta = beta  # Temperature factor
        self.all_pseudo_label = None
        self.all_img_cams = None
        self.unique_cams = None
        self.crosscam_epoch = crosscam_epoch
        self.bg_knn = bg_knn
    
    def forward(self, features, targets, cams=None, epoch=None, all_pseudo_label=None,
                all_img_cams=None, batch_ind=-1, init_intra_id_feat=None):

        loss = torch.tensor([0.]).to(device='cuda')
        self.all_pseudo_label = torch.tensor(all_pseudo_label).to(self.device)
        self.all_img_cams = torch.tensor(all_img_cams).to(self.device)
        self.unique_cams = torch.unique(self.all_img_cams)
        self.init_intra_id_feat = init_intra_id_feat

        loss = self.loss_using_pseudo_percam_proxy(features, targets, cams, batch_ind, epoch)

        return loss


    def loss_using_pseudo_percam_proxy(self, features, targets, cams, batch_ind, epoch):
        if batch_ind == 0:
            # initialize proxy memory
            self.percam_memory = []
            self.memory_class_mapper = []
            self.concate_intra_class = []
            for cc in self.unique_cams:
                percam_ind = torch.nonzero(self.all_img_cams == cc).squeeze(-1)
                uniq_class = torch.unique(self.all_pseudo_label[percam_ind], sorted=True)
                uniq_class = uniq_class[uniq_class >= 0]
                self.concate_intra_class.append(uniq_class)
                cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
                self.memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

                if len(self.init_intra_id_feat) > 0:
                    # print('initializing ID memory from updated embedding features...')
                    proto_memory = self.init_intra_id_feat[cc]
                    proto_memory = proto_memory.to(self.device)
                    self.percam_memory.append(proto_memory.detach())
            self.concate_intra_class = torch.cat(self.concate_intra_class)

        if epoch >= self.crosscam_epoch:
            all_cam_feats = []
            for ii in self.unique_cams:
                all_cam_feats.append(self.percam_memory[ii].detach().clone())
            all_cam_feats = torch.cat(all_cam_feats, dim=0).to(self.device)

        loss = torch.tensor([0.]).to(self.device)
        for cc in torch.unique(cams):  # every unique camera id in mini-batch
            inds = torch.nonzero(cams == cc).squeeze(-1)
            percam_targets = targets[inds]
            percam_feat = features[inds]

            # intra-camera loss
            mapped_targets = [self.memory_class_mapper[cc][int(k)] for k in percam_targets]
            mapped_targets = torch.tensor(mapped_targets).to(self.device)
            update_cam_feats(self.percam_memory, cc, percam_feat, mapped_targets, self.alpha)
            percam_inputs_similarity = F.normalize(percam_feat, dim=1).mm(F.normalize(self.percam_memory[cc].t(), dim=0))
            percam_inputs_similarity /= self.beta  # similarity score before softmax
            loss += F.cross_entropy(percam_inputs_similarity, mapped_targets)

            # global loss
            if epoch >= self.crosscam_epoch:
                associate_loss = 0
                target_inputs = torch.matmul(F.normalize(percam_feat, dim=1), F.normalize(all_cam_feats.t().clone(), dim=0))
                temp_sims = target_inputs.detach().clone()
                target_inputs /= self.beta

                for k in range(len(percam_feat)):
                    ori_asso_ind = torch.nonzero(self.concate_intra_class == percam_targets[k]).squeeze(-1)
                    temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                    sel_ind = torch.sort(temp_sims[k])[1][-self.bg_knn:]
                    concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                    concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(self.device)
                    concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                    associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
                loss += 0.5 * associate_loss / len(percam_feat)
        return loss