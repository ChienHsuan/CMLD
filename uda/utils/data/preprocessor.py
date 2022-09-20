import os
import os.path as osp
import random
import math

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, transform_weak=None, transform_strong=None, mutual=False, triplet=False):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
        self.mutual = mutual
        self.triplet = triplet

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.mutual:
            return self._get_mutual_item(indices)
        elif self.triplet:
            return self._get_triplet_item(indices)
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid

    def _get_mutual_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_1 = Image.open(fpath).convert('RGB')
        img_2 = img_1.copy()

        if self.transform_weak is not None and self.transform_strong is not None:
            img_1 = self.transform_weak(img_1)
            img_2 = self.transform_strong(img_2)

        return img_1, img_2, fpath, pid, camid

    def _get_triplet_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_1 = Image.open(fpath).convert('RGB')
        img_2 = img_1.copy()
        img_3 = img_1.copy()

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)
            img_3 = self.transform(img_3)

        return img_1, img_2, img_3, fpath, pid, camid
