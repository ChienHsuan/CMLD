import os.path as osp
import glob
import re
import urllib
import zipfile

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class Lab314(BaseImageDataset):
    dataset_dir = 'LAB_data'

    def __init__(self, root, verbose=True, **kwargs):
        super(Lab314, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir_cam_0 = osp.join(self.dataset_dir, 'cam_0')
        self.train_dir_cam_1 = osp.join(self.dataset_dir, 'cam_1')
        self.train_dir_cam_2 = osp.join(self.dataset_dir, 'cam_2')

        self._check_before_run()

        train = self._process_dir(self.train_dir_cam_0, self.train_dir_cam_1, self.train_dir_cam_2)

        self.train = train

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        print("=> Lab314 loaded")
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(self.num_train_pids, self.num_train_imgs, self.num_train_cams))
        print("  ----------------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir_cam_0):
            raise RuntimeError("'{}' is not available".format(self.train_dir_cam_1))
        if not osp.exists(self.train_dir_cam_1):
            raise RuntimeError("'{}' is not available".format(self.train_dir_cam_2))
        if not osp.exists(self.train_dir_cam_2):
            raise RuntimeError("'{}' is not available".format(self.train_dir_cam_3))

    def _process_dir(self, dir_path_0, dir_path_1, dir_path_2):
        img_paths_0 = glob.glob(osp.join(dir_path_0, '*.jpg'))
        img_paths_1 = glob.glob(osp.join(dir_path_1, '*.jpg'))
        img_paths_2 = glob.glob(osp.join(dir_path_2, '*.jpg'))

        dataset = []
        pid = 0

        camid = 0
        for img_path in img_paths_0:
            dataset.append((img_path, pid, camid))

        camid = 1
        for img_path in img_paths_1:
            dataset.append((img_path, pid, camid))

        camid = 2
        for img_path in img_paths_2:
            dataset.append((img_path, pid, camid))

        return dataset
