# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import json
import warnings
from collections import OrderedDict, defaultdict
from pycocotools.coco import COCO

import mmcv
import numpy as np
from mmcv import Config

from mmpose.core.evaluation import keypoint_mpjpe
from mmpose.datasets.datasets.base import Kpt3dSviewKpt2dDataset
from mmpose.datasets.datasets.body3d import Body3DH36MModifiedDataset
from ...builder import DATASETS

@DATASETS.register_module(name="Body3DCombinedDataset")
class Body3DCombinedDataset(Kpt3dSviewKpt2dDataset):
    """Human3.6M dataset for 3D human pose estimation.

    `Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human
    Sensing in Natural Environments' TPAMI`2014
    More details can be found in the `paper
    <http://vision.imar.ro/human3.6m/pami-h36m.pdf>`__.

    Human3.6M keypoint indexes::
        0: 'root (pelvis)',
        1: 'right_hip',
        2: 'right_knee',
        3: 'right_foot',
        4: 'left_hip',
        5: 'left_knee',
        6: 'left_foot',
        7: 'spine',
        8: 'thorax',
        9: 'neck_base',
        10: 'head',
        11: 'left_shoulder',
        12: 'left_elbow',
        13: 'left_wrist',
        14: 'right_shoulder',
        15: 'right_elbow',
        16: 'right_wrist'


    Args:
        ann_prefix (str): Path to the annotation files folder.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (DatasetInfo): A class containing all dataset info.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    JOINT_NAMES = [
        'Root', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine',
        'Thorax', 'NeckBase', 'Head', 'LShoulder', 'LElbow', 'LWrist',
        'RShoulder', 'RElbow', 'RWrist'
    ]

    # 2D joint source options:
    # "gt": from the annotation file
    # "detection": from a detection result file of 2D keypoint
    # "pipeline": will be generate by the pipeline
    SUPPORTED_JOINT_2D_SRC = {'gt', 'detection', 'pipeline'}

    # metric
    ALLOWED_METRICS = {'mpjpe', 'p-mpjpe', 'n-mpjpe'}

    def __init__(self,
                 child_types,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/h36m.py')
            dataset_info = cfg._cfg_dict['dataset_info']
        
        self.protocol = 2
        self.lshoulder_idx = 11
        self.rshoulder_idx = 14
        self.thorax_idx = 8
        self.root_idx = 0
        self.joint_num = 17
        self.old_to_new_coords = [0, 1, 2, 3, 4, 5, 6, 7, 0, 8, 10, 11, 12, 13, 14, 15, 16]

        self.child_datasets = []
        for i in range(len(child_types)):
            obj_cls = DATASETS.get(child_types[i])
            self.child_datasets.append(
                obj_cls(
                    ann_file[0],
                    img_prefix[0],
                    data_cfg[0],
                    pipeline,
                    dataset_info=dataset_info,
                    test_mode=test_mode

                )
            )
        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)


    def load_config(self, data_cfg_list):
        self.child_datasets[0].load_config(data_cfg_list)

    def load_annotations(self):
        return self.child_datasets[0].load_annotations()

    @staticmethod
    def _parse_h36m_imgname(imgname):
        """Parse imgname to get information of subject, action and camera.

        A typical h36m modified image filename is like:
        s_11_act_16_subact_02_ca_04_001360.jpg
        """
        name_parts = osp.basename(imgname).split('_')
        return name_parts[1], name_parts[3]+"_"+name_parts[5], name_parts[7]

    def build_sample_indices(self):
        return self.child_datasets[0].build_sample_indices()

    def evaluate(self,
                 outputs,
                 res_folder,
                 metric='mpjpe',
                 logger=None,
                 **kwargs):

        return self.child_datasets[0].evaluate(outputs, res_folder, metrix, logger, **kwargs)

    def get_camera_param(self, imgname):
        return self.child_datasets[0].get_camera_param(imgname)