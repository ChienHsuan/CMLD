import os.path as osp
import glob
import re

import imagesize

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json

class MTA(BaseImageDataset):
    """
    MTA(Multi Camera Track Auto) dataset
    Reference:
    The MTA Dataset for Multi Target Multi Camera Pedestrian Tracking by Weighted Distance Aggregation. CVPRW 2020
    """
    dataset_dir = 'mta/MTA_reid/'

    def __init__(self, root, verbose=True, height_threshold=50, **kwargs):
        super(MTA, self).__init__()
        self.height_threshold = height_threshold

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> MTA loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'camid_(\d)_pid_(\d+)')

        pid_container = set()
        for i, img_path in enumerate(img_paths):
            width, height = imagesize.get(img_path)
            if height > self.height_threshold:
                _, pid = map(int, pattern.search(img_path).groups())
                pid_container.add(pid)
            else:
                img_paths[i] = None
        img_paths = list(filter(lambda x: x is not None, img_paths))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            camid, pid = map(int, pattern.search(img_path).groups())
            assert 0 <= pid
            assert 0 <= camid <= 5
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
