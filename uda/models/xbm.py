import torch


class XBM:
    def __init__(self, memory_size, feature_size):
        self.K = memory_size
        self.D = feature_size
        self.feats = torch.zeros(self.K, self.D).cuda()
        self.targets = torch.zeros(self.K, dtype=torch.long).cuda()
        self.ptr = 0
        self._is_full = False

    @property
    def is_full(self):
        return self._is_full

    def get_feats(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
            self._is_full = True
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size
            